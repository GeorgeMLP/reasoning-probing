"""
Find reasoning-correlated features in SAE layers.

This script identifies SAE features that show differential activation
between reasoning and non-reasoning text, and analyzes which tokens
these features rely on.

## Metrics for Reasoning Feature Detection

1. **ROC-AUC**: Area under ROC curve for binary classification (reasoning vs not)
   - Threshold: >= 0.6 (better than random)

2. **Cohen's d**: Effect size (standardized mean difference)
   - Threshold: >= 0.3 (small-to-medium effect)

3. **Mann-Whitney U**: Non-parametric test for distribution differences
   - Threshold: p <= 0.01 after Bonferroni correction

4. **Composite Reasoning Score**: Weighted combination of all metrics

## Metrics for Token Dependency Analysis

1. **Mean Activation**: Average feature activation when token is present
2. **PMI**: Pointwise Mutual Information between token and feature
3. **Token Concentration**: Fraction of high activations from top tokens
   - High (>50%): Feature relies on shallow token cues
   - Low (<50%): Feature may capture deeper patterns

## Usage

```bash
# Basic usage
python find_reasoning_features.py --layer 8

# With specific reasoning dataset
python find_reasoning_features.py --layer 8 --reasoning-dataset s1k

# Save results
python find_reasoning_features.py --layer 8 --save-dir results/layer8
```
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reasoning_features.datasets import PileDataset, get_reasoning_dataset
from reasoning_features.features import (
    FeatureCollector,
    ReasoningFeatureDetector,
    TopTokenAnalyzer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find reasoning-correlated features in SAE layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        default="google/gemma-2-9b",
        help="HuggingFace model name (default: google/gemma-2-9b)",
    )
    parser.add_argument(
        "--sae-name",
        default="gemma-scope-9b-pt-res-canonical",
        help="SAE release name (default: gemma-scope-9b-pt-res-canonical)",
    )
    parser.add_argument(
        "--sae-id-format",
        default="layer_{layer}/width_16k/canonical",
        help="SAE ID format string",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Layer index to analyze (default: 8)",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--reasoning-dataset",
        choices=["s1k", "general_inquiry_cot", "combined"],
        default="s1k",
        help="Reasoning dataset to use (default: s1k)",
    )
    parser.add_argument(
        "--reasoning-samples",
        type=int,
        default=500,
        help="Number of reasoning samples (default: 500)",
    )
    parser.add_argument(
        "--nonreasoning-samples",
        type=int,
        default=500,
        help="Number of non-reasoning samples (default: 500)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )
    
    # Feature detection parameters
    parser.add_argument(
        "--feature-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of feature indices to analyze (default: all)",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.6,
        help="Minimum ROC-AUC for reasoning features (default: 0.6)",
    )
    parser.add_argument(
        "--max-pvalue",
        type=float,
        default=0.01,
        help="Maximum p-value for statistical significance (default: 0.01)",
    )
    parser.add_argument(
        "--min-effect-size",
        type=float,
        default=0.3,
        help="Minimum Cohen's d effect size (default: 0.3)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip threshold filtering; just rank by score and save top-k features",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=100,
        help="Number of top features to analyze (default: 100)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum features to collect from SAE (default: all)",
    )
    
    # Token analysis parameters
    parser.add_argument(
        "--top-k-tokens",
        type=int,
        default=30,
        help="Number of top tokens per feature (default: 30)",
    )
    parser.add_argument(
        "--min-token-occurrences",
        type=int,
        default=5,
        help="Minimum token occurrences for analysis (default: 5)",
    )
    parser.add_argument(
        "--top-k-bigrams",
        type=int,
        default=20,
        help="Number of top bigrams per feature (default: 20)",
    )
    parser.add_argument(
        "--min-bigram-occurrences",
        type=int,
        default=3,
        help="Minimum bigram occurrences for analysis (default: 3)",
    )
    parser.add_argument(
        "--top-k-trigrams",
        type=int,
        default=10,
        help="Number of top trigrams per feature (default: 10)",
    )
    parser.add_argument(
        "--min-trigram-occurrences",
        type=int,
        default=2,
        help="Minimum trigram occurrences for analysis (default: 2)",
    )
    
    # Reasoning score weights
    parser.add_argument(
        "--score-weight-auc",
        type=float,
        default=0.3,
        help="Weight for AUC contribution in reasoning score (default: 0.3)",
    )
    parser.add_argument(
        "--score-weight-effect",
        type=float,
        default=0.25,
        help="Weight for effect size contribution in reasoning score (default: 0.25)",
    )
    parser.add_argument(
        "--score-weight-pvalue",
        type=float,
        default=0.25,
        help="Weight for p-value contribution in reasoning score (default: 0.25)",
    )
    parser.add_argument(
        "--score-weight-freq",
        type=float,
        default=0.2,
        help="Weight for frequency contribution in reasoning score (default: 0.2)",
    )
    
    # Output
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--save-activations",
        action="store_true",
        help="Save activations to file",
    )
    parser.add_argument(
        "--load-activations",
        type=Path,
        default=None,
        help="Load pre-computed activations from file",
    )
    
    # Runtime
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for activation collection (default: 8)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("REASONING FEATURE DETECTION")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"SAE: {args.sae_name}")
    print(f"Layer: {args.layer}")
    print(f"Reasoning dataset: {args.reasoning_dataset}")
    print("=" * 60)
    
    # Create save directory if needed
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or collect activations
    if args.load_activations and args.load_activations.exists():
        print(f"\nLoading activations from {args.load_activations}")
        from reasoning_features.features.collector import FeatureActivations
        activations = FeatureActivations.load(args.load_activations)
    else:
        print("\n--- Collecting Activations ---")
        
        # Initialize collector
        collector = FeatureCollector(
            model_name=args.model_name,
            sae_name=args.sae_name,
            sae_id_format=args.sae_id_format,
            device=args.device,
        )
        
        # Load datasets
        print(f"\nLoading reasoning dataset: {args.reasoning_dataset}")
        reasoning_data = get_reasoning_dataset(
            args.reasoning_dataset,
            max_samples=args.reasoning_samples,
        )
        
        print(f"Loading non-reasoning dataset: Pile")
        nonreasoning_data = PileDataset(
            max_samples=args.nonreasoning_samples,
        )
        
        # Collect activations
        activations = collector.collect_from_datasets(
            reasoning_dataset=reasoning_data,
            nonreasoning_dataset=nonreasoning_data,
            layer_index=args.layer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            max_features=args.max_features,
        )
        
        # Save activations if requested
        if args.save_dir and args.save_activations:
            act_path = args.save_dir / "activations.pt"
            activations.save(act_path)
            print(f"Saved activations to {act_path}")
    
    print(f"\nActivations shape: {activations.activations.shape}")
    print(f"Reasoning samples: {sum(activations.is_reasoning)}")
    print(f"Non-reasoning samples: {sum(not r for r in activations.is_reasoning)}")
    
    # Detect reasoning features
    print("\n--- Detecting Reasoning Features ---")
    score_weights = {
        "auc": args.score_weight_auc,
        "effect": args.score_weight_effect,
        "pvalue": args.score_weight_pvalue,
        "freq": args.score_weight_freq,
    }
    detector = ReasoningFeatureDetector(
        activations, 
        aggregation="max",
        score_weights=score_weights
    )
    
    # Get all feature statistics (or subset if feature_indices specified)
    all_stats = detector.compute_all_stats(
        verbose=True,
        feature_indices=args.feature_indices,
    )
    
    # Apply Bonferroni correction
    detector.apply_bonferroni_correction(feature_indices=args.feature_indices)
    
    # Get reasoning features
    if args.no_filter:
        # Skip threshold filtering, just rank by score
        reasoning_features = detector.get_top_features_by_score(
            top_k=args.top_k_features,
            feature_indices=args.feature_indices,
        )
        print(f"\nTop {len(reasoning_features)} features by reasoning_score (no filtering):")
    else:
        reasoning_features = detector.get_reasoning_features(
            min_auc=args.min_auc,
            max_p_value=args.max_pvalue,
            min_effect_size=args.min_effect_size,
            top_k=args.top_k_features,
            feature_indices=args.feature_indices,
        )
        print(f"\nFound {len(reasoning_features)} reasoning features meeting criteria:")
        print(f"  - ROC-AUC >= {args.min_auc}")
        print(f"  - p-value <= {args.max_pvalue} (Bonferroni corrected)")
        print(f"  - Cohen's d >= {args.min_effect_size}")
    
    # Print summary
    summary = detector.summary(feature_indices=args.feature_indices)
    print(f"\n--- Summary ---")
    print(f"Total features analyzed: {summary['total_features']}")
    print(f"Reasoning features found: {summary['reasoning_features_count']} ({summary['percentage_reasoning']:.1f}%)")
    
    if reasoning_features:
        print(f"\nTop 10 reasoning features:")
        print("-" * 80)
        print(f"{'Index':>8} {'Score':>8} {'AUC':>8} {'Cohen d':>8} {'p-value':>12} {'Mean(R)':>10} {'Mean(NR)':>10}")
        print("-" * 80)
        for stat in reasoning_features[:10]:
            print(
                f"{stat.feature_index:>8} "
                f"{stat.reasoning_score:>8.3f} "
                f"{stat.roc_auc:>8.3f} "
                f"{stat.cohens_d:>8.3f} "
                f"{stat.mannwhitney_p:>12.2e} "
                f"{stat.mean_reasoning:>10.4f} "
                f"{stat.mean_nonreasoning:>10.4f}"
            )
    
    # Token analysis
    print("\n--- Token Dependency Analysis ---")
    collector = FeatureCollector(
        model_name=args.model_name,
        sae_name=args.sae_name,
        sae_id_format=args.sae_id_format,
        device=args.device,
    )
    collector.load_model()
    
    token_analyzer = TopTokenAnalyzer(
        activations,
        collector.model.tokenizer,
    )
    
    feature_token_analyses = []
    
    # Analyze top features
    features_to_analyze = reasoning_features[:min(args.top_k_features, len(reasoning_features))]
    if not features_to_analyze:
        # If no features meet criteria, analyze top by score anyway
        features_to_analyze = detector.get_top_features_by_score(
            args.top_k_features,
            feature_indices=args.feature_indices,
        )
    
    print(f"\nAnalyzing token dependencies for {len(features_to_analyze)} features...")
    
    for stat in features_to_analyze:
        analysis = token_analyzer.analyze_feature_token_dependency(
            stat.feature_index,
            top_k_tokens=args.top_k_tokens,
        )
        analysis["feature_stats"] = stat.to_dict()
        
        # Add top bigrams
        top_bigrams = token_analyzer.get_top_ngrams_for_feature(
            stat.feature_index,
            n=2,
            top_k=args.top_k_bigrams,
            min_occurrences=args.min_bigram_occurrences,
        )
        analysis["top_bigrams"] = [b.to_dict() for b in top_bigrams]
        
        # Add top trigrams
        top_trigrams = token_analyzer.get_top_ngrams_for_feature(
            stat.feature_index,
            n=3,
            top_k=args.top_k_trigrams,
            min_occurrences=args.min_trigram_occurrences,
        )
        analysis["top_trigrams"] = [t.to_dict() for t in top_trigrams]
        
        feature_token_analyses.append(analysis)
        
        # Print summary
        top_tokens = [t["token_str"] for t in analysis["top_tokens"][:5]]
        top_bi_strs = [b["ngram_str"] for b in analysis["top_bigrams"][:3]]
        print(
            f"Feature {stat.feature_index:>5}: "
            f"Token concentration: {analysis['token_concentration']:.2%}, "
            f"Top tokens: {top_tokens}"
        )
        if top_bi_strs:
            print(f"                      Top bigrams: {top_bi_strs}")
    
    # Save results
    if args.save_dir:
        # Save feature statistics
        stats_path = args.save_dir / "feature_stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                [s.to_dict() for s in all_stats],
                f,
                indent=2,
            )
        print(f"\nSaved feature statistics to {stats_path}")
        
        # Save reasoning features
        reasoning_path = args.save_dir / "reasoning_features.json"
        config = {
            "model_name": args.model_name,
            "sae_name": args.sae_name,
            "layer": args.layer,
            "reasoning_dataset": args.reasoning_dataset,
            "no_filter": args.no_filter,
        }
        if not args.no_filter:
            config.update({
                "min_auc": args.min_auc,
                "max_pvalue": args.max_pvalue,
                "min_effect_size": args.min_effect_size,
            })
        
        with open(reasoning_path, "w") as f:
            json.dump(
                {
                    "config": config,
                    "summary": summary,
                    "feature_indices": [s.feature_index for s in reasoning_features],
                    "features": [s.to_dict() for s in reasoning_features],
                },
                f,
                indent=2,
            )
        print(f"Saved reasoning features to {reasoning_path}")
        
        # Save token analysis
        tokens_path = args.save_dir / "token_analysis.json"
        
        # Compute summary statistics
        token_analysis_summary = {
            "total_features_analyzed": len(feature_token_analyses),
            "high_token_dependency_count": sum(
                1 for a in feature_token_analyses if a["token_concentration"] > 0.5
            ),
            "high_token_dependency_percentage": (
                sum(1 for a in feature_token_analyses if a["token_concentration"] > 0.5) 
                / len(feature_token_analyses) * 100
                if feature_token_analyses else 0
            ),
            "mean_token_concentration": (
                sum(a["token_concentration"] for a in feature_token_analyses) 
                / len(feature_token_analyses)
                if feature_token_analyses else 0
            ),
            "mean_normalized_entropy": (
                sum(a["normalized_entropy"] for a in feature_token_analyses) 
                / len(feature_token_analyses)
                if feature_token_analyses else 0
            ),
        }
        
        with open(tokens_path, "w") as f:
            json.dump({
                "summary": token_analysis_summary,
                "features": feature_token_analyses,
            }, f, indent=2)
        print(f"Saved token analysis to {tokens_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if reasoning_features:
        high_token_dep = sum(
            1 for a in feature_token_analyses
            if a["token_concentration"] > 0.5
        )
        print(f"\nKey findings:")
        print(f"  - {len(reasoning_features)} features show significant reasoning correlation")
        print(f"  - {high_token_dep}/{len(feature_token_analyses)} analyzed features have HIGH token dependency")
        print(f"  - This suggests {high_token_dep/len(feature_token_analyses)*100:.0f}% may rely on shallow token cues")
    
    return reasoning_features, feature_token_analyses


if __name__ == "__main__":
    main()
