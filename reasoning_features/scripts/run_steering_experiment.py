"""
Run steering experiments to test whether amplifying reasoning features improves performance.

This script evaluates model performance on benchmarks with steering of individual
reasoning features, testing the hypothesis that these features capture genuine
reasoning vs. shallow token cues.

## Steering Formula

x' = x + γ * f_max * W_dec[i]

Where:
- γ: Steering strength (typically -4 to 4)
- f_max: Maximum activation of feature i
- W_dec[i]: Decoder direction for feature i

## Hypothesis

If "reasoning features" capture actual reasoning ability:
- Positive γ should improve benchmark performance
- Negative γ should decrease performance

If "reasoning features" are spurious (token correlations):
- Positive γ will likely decrease performance
- The model may produce text that "looks like" reasoning but isn't correct

## Benchmarks

- **aime24**: AIME 2024 math competition problems (30 problems, numerical answers)
- **gpqa_diamond**: Graduate-level science questions (198 problems, A/B/C/D)
- **math500**: MATH-500 diverse math problems (500 problems, various answer formats)
  - Note: math500 requires OPENROUTER_API_KEY for LLM-based answer checking

## Output Structure

Results are saved per-feature:
    {save_dir}/feature_{index}/
        ├── result_gamma_{value}.json  # Per-gamma results
        └── feature_summary.json       # Summary for this feature

## Usage

```bash
# Run with detected features (steers each feature individually)
python run_steering_experiment.py \\
    --features-file results/layer8/reasoning_features.json \\
    --benchmark aime24 \\
    --save-dir results/steering

# Run with specific features
python run_steering_experiment.py \\
    --feature-indices 42 128 256 \\
    --benchmark gpqa_diamond \\
    --gamma-values -2 -1 0 1 2

# Quick test
python run_steering_experiment.py \\
    --features-file results/layer8/reasoning_features.json \\
    --benchmark aime24 \\
    --max-samples 5 \\
    --top-k-features 2
```
"""

import argparse
import json
import os
from pathlib import Path
import sys
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reasoning_features.steering import BenchmarkEvaluator, SteeringConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run steering experiments on benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        default="google/gemma-3-4b-it",
        help="HuggingFace model name (default: google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--sae-name",
        default="gemma-scope-2-4b-it-res-all",
        help="SAE release name (default: gemma-scope-2-4b-it-res-all)",
    )
    parser.add_argument(
        "--sae-id-format",
        default="layer_{layer}_width_16k_l0_small",
        help="SAE ID format string",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Layer index for steering (default: 8)",
    )
    
    # Feature specification
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument(
        "--features-file",
        type=Path,
        help="JSON file with reasoning features (from find_reasoning_features.py)",
    )
    feature_group.add_argument(
        "--feature-indices",
        type=int,
        nargs="+",
        help="Specific feature indices to steer",
    )
    
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=10,
        help="Number of top features to steer individually (default: 10)",
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--benchmark",
        choices=["aime24", "gpqa_diamond", "math500"],
        required=True,
        help="Benchmark to evaluate (math500 requires OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of benchmark samples (for testing)",
    )
    
    # Steering configuration
    parser.add_argument(
        "--gamma-values",
        type=float,
        nargs="+",
        default=[0.0, 2.0],
        help="Steering gamma values to test (default: 0 2)",
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-gen-toks",
        type=int,
        default=32768,
        help="Maximum tokens to generate (default: 32768 for full reasoning traces)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6, matching lm-eval)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template application",
    )
    
    # Output
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save results",
    )
    
    # Runtime
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    
    return parser.parse_args()


def load_features_from_file(file_path: Path, top_k: int) -> list[dict]:
    """Load feature info from a reasoning features JSON file.
    
    Returns:
        List of dicts with 'feature_index' and stats
    """
    with open(file_path) as f:
        data = json.load(f)
    
    features = data.get("features", [])
    if not features and "feature_indices" in data:
        # Simple format with just indices
        features = [{"feature_index": idx} for idx in data["feature_indices"]]
    
    return features[:top_k]


def get_max_activation_for_feature(
    token_analysis_path: Path,
    feature_index: int,
) -> float:
    """Get the maximum activation for a feature from token analysis.
    
    Falls back to a default value if not found.
    """
    if token_analysis_path and token_analysis_path.exists():
        with open(token_analysis_path) as f:
            data = json.load(f)
        
        for feat in data.get("features", []):
            if feat.get("feature_index") == feature_index:
                # Use the maximum from top tokens or a stored max
                top_tokens = feat.get("top_tokens", [])
                if top_tokens:
                    return max(t.get("max_activation", 1.0) for t in top_tokens)
                return feat.get("max_activation", 1.0)
    
    # Default fallback
    return 1.0


def main():
    args = parse_args()
    
    # Check for API key if using math500
    if args.benchmark == "math500":
        if not os.getenv("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY environment variable required for math500 benchmark.")
            print("The math500 benchmark uses an LLM judge to evaluate mathematical expression equivalence.")
            print("\nSet the API key with:")
            print("  export OPENROUTER_API_KEY=your_key_here")
            sys.exit(1)
    
    print("=" * 60)
    print("STEERING EXPERIMENT (Per-Feature)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"SAE: {args.sae_name}")
    print(f"Layer: {args.layer}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Gamma values: {args.gamma_values}")
    print(f"Max new tokens: {args.max_gen_toks}")
    print(f"Temperature: {args.temperature}")
    print(f"Chat template: {not args.no_chat_template}")
    print("=" * 60)
    
    # Load feature info
    if args.features_file:
        print(f"\nLoading features from {args.features_file}")
        features = load_features_from_file(args.features_file, args.top_k_features)
        feature_indices = [f["feature_index"] for f in features]
        
        # Try to find token_analysis.json in same directory for max activations
        token_analysis_path = args.features_file.parent / "token_analysis.json"
    else:
        feature_indices = args.feature_indices[:args.top_k_features]
        features = [{"feature_index": idx} for idx in feature_indices]
        token_analysis_path = None
    
    print(f"Will steer {len(feature_indices)} features individually: {feature_indices}")
    
    # Create save directory
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TransformerLens evaluator
    print("\nInitializing TransformerLens evaluator...")
    from sae_lens import SAE, HookedSAETransformer
    
    model = HookedSAETransformer.from_pretrained_no_processing(
        args.model_name,
        device="cuda",
        dtype=torch.bfloat16,
    )
    
    sae_id = args.sae_id_format.format(layer=args.layer)
    sae = SAE.from_pretrained(
        release=args.sae_name,
        sae_id=sae_id,
        device="cuda",
    )
    
    evaluator = BenchmarkEvaluator(
        model=model,
        sae=sae,
        layer_index=args.layer,
    )
    
    # Store all results for final summary
    all_results = {}
    
    # Run experiment for each feature individually
    for feat_info in features:
        feature_index = feat_info["feature_index"]
        
        print(f"\n{'='*60}")
        print(f"FEATURE {feature_index}")
        print(f"{'='*60}")
        
        # Get max activation for this feature
        max_activation = get_max_activation_for_feature(
            token_analysis_path, feature_index
        )
        print(f"Max feature activation (f_max): {max_activation:.2f}")
        
        # Create feature-specific save directory
        if args.save_dir:
            feature_save_dir = args.save_dir / f"feature_{feature_index}"
            feature_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            feature_save_dir = None
        
        # Run experiment for this feature - evaluate each gamma separately
        results = {}
        
        for gamma in args.gamma_values:
            print(f"\n{'='*60}")
            print(f"Testing gamma = {gamma}")
            print(f"{'='*60}")
            
            if gamma == 0.0:
                # Baseline
                result = evaluator.evaluate(
                    benchmark_name=args.benchmark,
                    condition="baseline",
                    max_new_tokens=args.max_gen_toks,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_samples=args.max_samples,
                    verbose=True,
                )
            else:
                # Steered
                steering_config = SteeringConfig(
                    feature_index=feature_index,
                    gamma=gamma,
                    max_feature_activation=max_activation,
                )
                result = evaluator.evaluate(
                    benchmark_name=args.benchmark,
                    condition=f"steered_gamma_{gamma}",
                    steering_config=steering_config,
                    max_new_tokens=args.max_gen_toks,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_samples=args.max_samples,
                    verbose=True,
                )
            
            results[gamma] = result
            
            # Save individual result
            if feature_save_dir:
                # Format gamma value with 2 decimal places and sign
                gamma_str = f"{gamma:.2f}" if gamma >= 0 else f"{gamma:.2f}"
                result.save(feature_save_dir / f"result_gamma_{gamma_str}.json")
        
        all_results[feature_index] = results
        
        # Print summary for this feature
        print(f"\n--- Feature {feature_index} Summary ---")
        print(f"{'Gamma':>10} {'Accuracy':>12} {'Correct':>12}")
        print("-" * 35)
        
        baseline_acc = results.get(0.0, list(results.values())[0]).accuracy
        
        for gamma, result in sorted(results.items()):
            delta = result.accuracy - baseline_acc
            delta_str = f"({delta:+.2%})" if gamma != 0.0 else "(baseline)"
            print(f"{gamma:>10.1f} {result.accuracy:>12.2%} {result.correct:>8}/{result.total:<3} {delta_str}")
        
        # Save feature summary
        if feature_save_dir:
            # Find best and worst gamma
            sorted_by_acc = sorted(results.items(), key=lambda x: x[1].accuracy)
            worst_gamma, worst_result = sorted_by_acc[0]
            best_gamma, best_result = sorted_by_acc[-1]
            
            feature_summary = {
                "feature_index": feature_index,
                "max_feature_activation": max_activation,
                "baseline_accuracy": baseline_acc,
                "best_gamma": best_gamma,
                "best_accuracy": best_result.accuracy,
                "worst_gamma": worst_gamma,
                "worst_accuracy": worst_result.accuracy,
                "results": {
                    str(gamma): {
                        "accuracy": result.accuracy,
                        "correct": result.correct,
                        "total": result.total,
                    }
                    for gamma, result in results.items()
                }
            }
            
            with open(feature_save_dir / "feature_summary.json", "w") as f:
                json.dump(feature_summary, f, indent=2)
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    # Aggregate analysis
    improvements = []
    degradations = []
    
    for feature_index, results in all_results.items():
        baseline_acc = results.get(0.0, list(results.values())[0]).accuracy
        
        for gamma, result in results.items():
            if gamma > 0:
                delta = result.accuracy - baseline_acc
                if delta > 0.02:
                    improvements.append((feature_index, gamma, delta))
                elif delta < -0.02:
                    degradations.append((feature_index, gamma, delta))
    
    print(f"\nFeatures analyzed: {len(all_results)}")
    print(f"Gamma values tested: {args.gamma_values}")
    
    if improvements:
        print(f"\n✓ Improvements (>2% accuracy gain with positive gamma):")
        for feat, gamma, delta in sorted(improvements, key=lambda x: -x[2])[:5]:
            print(f"  Feature {feat}, γ={gamma}: {delta:+.2%}")
    
    if degradations:
        print(f"\n✗ Degradations (>2% accuracy loss with positive gamma):")
        for feat, gamma, delta in sorted(degradations, key=lambda x: x[2])[:5]:
            print(f"  Feature {feat}, γ={gamma}: {delta:+.2%}")
    
    # Interpretation
    print("\n--- Interpretation ---")
    n_improved = len(set(f for f, _, _ in improvements))
    n_degraded = len(set(f for f, _, _ in degradations))
    
    if n_improved > n_degraded:
        print("✓ More features showed improvement than degradation")
        print("  → Some features may capture genuine reasoning")
    elif n_degraded > n_improved:
        print("✗ More features showed degradation than improvement")
        print("  → Features likely capture spurious correlations")
    else:
        print("○ Mixed results across features")
        print("  → Features have heterogeneous effects")
    
    # Save overall summary
    if args.save_dir:
        overall_summary_path = args.save_dir / "experiment_summary.json"
        with open(overall_summary_path, "w") as f:
            json.dump({
                "config": {
                    "model_name": args.model_name,
                    "sae_name": args.sae_name,
                    "layer": args.layer,
                    "benchmark": args.benchmark,
                    "gamma_values": args.gamma_values,
                    "feature_indices": feature_indices,
                    "max_gen_toks": args.max_gen_toks,
                    "temperature": args.temperature,
                    "chat_template": not args.no_chat_template,
                },
                "per_feature_results": {
                    str(feat_idx): {
                        str(gamma): {
                            "accuracy": result.accuracy,
                            "correct": result.correct,
                            "total": result.total,
                        }
                        for gamma, result in results.items()
                    }
                    for feat_idx, results in all_results.items()
                },
                "analysis": {
                    "n_features": len(all_results),
                    "n_improved": n_improved,
                    "n_degraded": n_degraded,
                },
            }, f, indent=2)
        print(f"\nSaved overall summary to {overall_summary_path}")
    
    return all_results


if __name__ == "__main__":
    main()
