"""
Run ANOVA experiments to disentangle token vs. behavior effects on reasoning features.

This script implements the 2×2 factorial design to determine whether SAE features
that correlate with reasoning text are actually capturing reasoning behavior,
or merely responding to token-level distributional cues.

## The 2×2 Design

For each feature, we create four conditions based on its top tokens:
- A: Reasoning text WITH feature's top tokens
- B: Reasoning text WITHOUT feature's top tokens
- C: Non-reasoning text WITH feature's top tokens
- D: Non-reasoning text WITHOUT feature's top tokens

## Key Metric: η² (Eta-Squared)

- η²_token: Variance explained by presence of top tokens
- η²_behavior: Variance explained by reasoning vs. non-reasoning

If η²_token >> η²_behavior: Feature is token-dominated (spurious)
If η²_behavior >> η²_token: Feature captures genuine reasoning

## Usage

```bash
# Run ANOVA for features detected in layer 8 using s1K dataset
python run_anova_experiment.py \\
    --token-analysis results/layer8/token_analysis.json \\
    --layer 8 \\
    --save-dir results/anova/layer8

# Run with General Inquiry CoT dataset
python run_anova_experiment.py \\
    --token-analysis results/layer8/token_analysis.json \\
    --reasoning-dataset general_inquiry_cot \\
    --layer 8 \\
    --save-dir results/anova/layer8/general_inquiry_cot

# Run with combined dataset (both s1K and General Inquiry CoT)
python run_anova_experiment.py \\
    --token-analysis results/layer8/token_analysis.json \\
    --reasoning-dataset combined \\
    --layer 8 \\
    --save-dir results/anova/layer8/combined

# Quick test with fewer samples
python run_anova_experiment.py \\
    --token-analysis results/layer8/token_analysis.json \\
    --layer 8 \\
    --n-per-condition 50 \\
    --top-k-features 10
```
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reasoning_features.datasets.anova import (
    ANOVAResult,
    compute_anova_for_feature,
    compute_anova_summary,
    load_all_top_tokens,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ANOVA experiments for token vs. behavior effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input files
    parser.add_argument(
        "--token-analysis",
        type=Path,
        required=True,
        help="Path to token_analysis.json from find_reasoning_features.py",
    )
    parser.add_argument(
        "--reasoning-features",
        type=Path,
        default=None,
        help="Path to reasoning_features.json (optional, for feature selection)",
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
        required=True,
        help="Layer index for SAE",
    )
    
    # Feature selection
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=50,
        help="Number of top reasoning features to analyze (default: 50)",
    )
    parser.add_argument(
        "--top-k-tokens",
        type=int,
        default=10,
        help="Number of top tokens per feature for classification (default: 10)",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--reasoning-dataset",
        choices=["s1k", "general_inquiry_cot", "combined"],
        default="s1k",
        help="Reasoning dataset to use (default: s1k)",
    )
    parser.add_argument(
        "--n-reasoning-chains",
        type=int,
        default=500,
        help="Number of reasoning chains to load (default: 500)",
    )
    parser.add_argument(
        "--n-nonreasoning-samples",
        type=int,
        default=1000,
        help="Number of non-reasoning samples to load (default: 1000)",
    )
    parser.add_argument(
        "--n-per-condition",
        type=int,
        default=200,
        help="Samples per ANOVA condition (default: 200)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Threshold for token presence (default: 1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization (default: 256)",
    )
    
    # Output
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Directory to save results",
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
        default=16,
        help="Batch size for activation collection (default: 16)",
    )
    
    return parser.parse_args()


def collect_activations_for_texts(
    texts: list[str],
    model,
    sae,
    tokenizer,
    layer_index: int,
    feature_indices: list[int],
    max_length: int = 256,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict[int, np.ndarray]:
    """
    Collect SAE feature activations for a list of texts.
    
    Returns dict mapping feature_index -> array of max activations per text.
    """
    n_texts = len(texts)
    
    # Initialize storage
    activations = {feat_idx: [] for feat_idx in feature_indices}
    
    # Process in batches
    for batch_start in range(0, n_texts, batch_size):
        batch_end = min(batch_start + batch_size, n_texts)
        batch_texts = texts[batch_start:batch_end]
        
        # Tokenize
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokens["input_ids"].to(device)
        
        # Get model activations
        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=[f"blocks.{layer_index}.hook_resid_post"],
            )
            
            resid = cache[f"blocks.{layer_index}.hook_resid_post"]
            
            # Get SAE activations
            sae_acts = sae.encode(resid)  # (batch, seq, n_sae_features)
            
            # Extract max activation for each feature
            for feat_idx in feature_indices:
                # Max over sequence dimension
                max_acts = sae_acts[:, :, feat_idx].max(dim=1).values
                activations[feat_idx].extend(max_acts.cpu().numpy().tolist())
    
    # Convert to numpy arrays
    return {k: np.array(v) for k, v in activations.items()}


def run_anova_for_feature(
    feature_index: int,
    feature_tokens: set[str],
    reasoning_sentences: list[str],
    nonreasoning_sentences: list[str],
    model,
    sae,
    tokenizer,
    layer_index: int,
    n_per_condition: int,
    max_length: int,
    batch_size: int,
    device: str,
    token_presence_threshold: int = 1,
) -> Optional[ANOVAResult]:
    """Run ANOVA analysis for a single feature."""
    from reasoning_features.datasets.anova import text_contains_tokens
    
    # Classify sentences by token presence
    reasoning_with_tokens = [s for s in reasoning_sentences if text_contains_tokens(s, feature_tokens, threshold=token_presence_threshold)]
    reasoning_no_tokens = [s for s in reasoning_sentences if not text_contains_tokens(s, feature_tokens, threshold=token_presence_threshold)]
    nonreasoning_with_tokens = [s for s in nonreasoning_sentences if text_contains_tokens(s, feature_tokens, threshold=token_presence_threshold)]
    nonreasoning_no_tokens = [s for s in nonreasoning_sentences if not text_contains_tokens(s, feature_tokens, threshold=token_presence_threshold)]
    
    # Check if we have enough samples
    min_samples = min(
        len(reasoning_with_tokens),
        len(reasoning_no_tokens),
        len(nonreasoning_with_tokens),
        len(nonreasoning_no_tokens),
    )
    
    if min_samples < 200:
        print(f"  Feature {feature_index}: Insufficient samples (min={min_samples})")
        return None
    
    # Limit to n_per_condition
    n = min(min_samples, n_per_condition)
    
    # Shuffle and select
    rng = np.random.RandomState(feature_index)
    rng.shuffle(reasoning_with_tokens)
    rng.shuffle(reasoning_no_tokens)
    rng.shuffle(nonreasoning_with_tokens)
    rng.shuffle(nonreasoning_no_tokens)
    
    condition_texts = {
        "has_tokens_reasoning": reasoning_with_tokens[:n],
        "no_tokens_reasoning": reasoning_no_tokens[:n],
        "has_tokens_nonreasoning": nonreasoning_with_tokens[:n],
        "no_tokens_nonreasoning": nonreasoning_no_tokens[:n],
    }
    
    # Collect activations for each condition
    all_texts = []
    text_to_condition = []
    for cond_name, texts in condition_texts.items():
        for text in texts:
            all_texts.append(text)
            text_to_condition.append(cond_name)
    
    # Get activations
    activations = collect_activations_for_texts(
        all_texts,
        model,
        sae,
        tokenizer,
        layer_index,
        [feature_index],
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    
    # Organize by condition
    feature_acts = activations[feature_index]
    activations_by_condition = {cond: [] for cond in condition_texts.keys()}
    
    for i, cond_name in enumerate(text_to_condition):
        activations_by_condition[cond_name].append(feature_acts[i])
    
    activations_by_condition = {k: np.array(v) for k, v in activations_by_condition.items()}
    
    # Compute ANOVA
    return compute_anova_for_feature(activations_by_condition, feature_index)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("ANOVA EXPERIMENT: Token vs. Behavior Effects")
    print("=" * 60)
    print(f"Token analysis: {args.token_analysis}")
    print(f"Reasoning dataset: {args.reasoning_dataset}")
    print(f"Layer: {args.layer}")
    print(f"Top-k features: {args.top_k_features}")
    print(f"Top-k tokens per feature: {args.top_k_tokens}")
    print(f"Samples per condition: {args.n_per_condition}")
    print("=" * 60)
    
    # Create output directory
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load top tokens for all features
    print("\n--- Loading Feature Tokens ---")
    all_top_tokens = load_all_top_tokens(
        args.token_analysis,
        top_k_per_feature=args.top_k_tokens,
    )
    
    # Get feature indices to analyze
    feature_indices = list(all_top_tokens.keys())[:args.top_k_features]
    print(f"Analyzing {len(feature_indices)} features")
    
    # Load datasets and split into sentences
    print("\n--- Loading and Splitting Datasets ---")
    from reasoning_features.datasets.anova import split_into_sentences
    from datasets import load_dataset
    import re
    
    # Load reasoning data based on selected dataset
    print(f"Loading reasoning data from {args.reasoning_dataset}...")
    reasoning_sentences = []
    
    if args.reasoning_dataset in ["s1k", "combined"]:
        # Load s1K-1.1 dataset
        print("  Loading s1K-1.1...")
        s1k_ds = load_dataset("simplescaling/s1K-1.1", split="train")
        for row in s1k_ds:
            # Try both Gemini and DeepSeek trajectories
            for col in ["gemini_thinking_trajectory", "deepseek_thinking_trajectory"]:
                text = row.get(col, "")
                if text:
                    sentences = split_into_sentences(text, min_length=50, max_length=500)
                    reasoning_sentences.extend(sentences)
            if args.reasoning_dataset == "s1k" and len(reasoning_sentences) >= args.n_reasoning_chains * 10:
                break
        print(f"    Got {len(reasoning_sentences)} sentences from s1K")
    
    if args.reasoning_dataset in ["general_inquiry_cot", "combined"]:
        # Load General Inquiry CoT dataset
        print("  Loading General Inquiry Thinking CoT...")
        giq_ds = load_dataset("moremilk/General_Inquiry_Thinking-Chain-Of-Thought", split="train")
        giq_count = 0
        for row in giq_ds:
            metadata = row.get("metadata", {})
            reasoning_text = metadata.get("reasoning", "") if isinstance(metadata, dict) else ""
            if reasoning_text:
                # Extract content between <think> and </think> tags if present
                think_match = re.search(r"<think>(.*?)</think>", reasoning_text, re.DOTALL)
                if think_match:
                    text = think_match.group(1).strip()
                else:
                    text = reasoning_text
                sentences = split_into_sentences(text, min_length=50, max_length=500)
                reasoning_sentences.extend(sentences)
                giq_count += len(sentences)
            if args.reasoning_dataset == "general_inquiry_cot" and len(reasoning_sentences) >= args.n_reasoning_chains * 10:
                break
        print(f"    Got {giq_count} sentences from General Inquiry CoT")
    
    print(f"  Total reasoning sentences: {len(reasoning_sentences)}")
    
    # Load non-reasoning data
    print("Loading non-reasoning data...")
    nonreasoning_ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    nonreasoning_sentences = []
    for row in nonreasoning_ds:
        text = row.get("text", "")
        if text:
            sentences = split_into_sentences(text, min_length=50, max_length=500)
            nonreasoning_sentences.extend(sentences)
        if len(nonreasoning_sentences) >= args.n_nonreasoning_samples * 3:
            break
    print(f"  Got {len(nonreasoning_sentences)} non-reasoning sentences")
    
    # Load model and SAE
    print("\n--- Loading Model and SAE ---")
    from sae_lens import SAE, HookedSAETransformer
    
    model = HookedSAETransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        dtype=torch.bfloat16,
    )
    
    sae_id = args.sae_id_format.format(layer=args.layer)
    sae = SAE.from_pretrained(
        release=args.sae_name,
        sae_id=sae_id,
        device=args.device,
    )
    
    tokenizer = model.tokenizer
    
    # Run ANOVA for each feature
    print("\n--- Running ANOVA Analysis ---")
    results = []
    condition_summaries = []
    
    for feat_idx in tqdm(feature_indices, desc="Analyzing features"):
        feature_tokens = all_top_tokens[feat_idx]
        
        # Quick check: how many sentences match?
        from reasoning_features.datasets.anova import text_contains_tokens
        n_r_with = sum(1 for s in reasoning_sentences if text_contains_tokens(s, feature_tokens, threshold=args.threshold))
        n_r_without = len(reasoning_sentences) - n_r_with
        n_nr_with = sum(1 for s in nonreasoning_sentences if text_contains_tokens(s, feature_tokens, threshold=args.threshold))
        n_nr_without = len(nonreasoning_sentences) - n_nr_with
        
        condition_summaries.append({
            "feature_index": feat_idx,
            "n_tokens": len(feature_tokens),
            "tokens_sample": list(feature_tokens)[:5],
            "n_reasoning_with_tokens": n_r_with,
            "n_reasoning_no_tokens": n_r_without,
            "n_nonreasoning_with_tokens": n_nr_with,
            "n_nonreasoning_no_tokens": n_nr_without,
        })
        
        min_samples = min(n_r_with, n_r_without, n_nr_with, n_nr_without)
        if min_samples < 100:
            print(f"  Feature {feat_idx}: Skipping (min samples = {min_samples})")
            continue
        
        result = run_anova_for_feature(
            feature_index=feat_idx,
            feature_tokens=feature_tokens,
            reasoning_sentences=reasoning_sentences,
            nonreasoning_sentences=nonreasoning_sentences,
            model=model,
            sae=sae,
            tokenizer=tokenizer,
            layer_index=args.layer,
            n_per_condition=args.n_per_condition,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            token_presence_threshold=args.threshold,
        )
        
        if result is not None:
            results.append(result)
    
    # Compute summary
    print("\n--- Computing Summary ---")
    summary = compute_anova_summary(results)
    
    print(f"\nResults:")
    print(f"  Features analyzed: {summary.get('n_features_analyzed', 0)}")
    print(f"  Token-dominated: {summary.get('n_token_dominated', 0)} ({summary.get('pct_token_dominated', 0):.1f}%)")
    print(f"  Behavior-dominated: {summary.get('n_behavior_dominated', 0)} ({summary.get('pct_behavior_dominated', 0):.1f}%)")
    print(f"  Mean η²_token: {summary.get('mean_eta_sq_token', 0):.4f}")
    print(f"  Mean η²_behavior: {summary.get('mean_eta_sq_behavior', 0):.4f}")
    print(f"  Dominant factor distribution: {summary.get('dominant_factor_distribution', {})}")
    
    # Save results
    print("\n--- Saving Results ---")
    
    # Save individual feature results
    results_path = args.save_dir / "anova_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "token_analysis": str(args.token_analysis),
                "reasoning_dataset": args.reasoning_dataset,
                "layer": args.layer,
                "top_k_features": args.top_k_features,
                "top_k_tokens": args.top_k_tokens,
                "n_per_condition": args.n_per_condition,
            },
            "summary": summary,
            "features": [r.to_dict() for r in results],
        }, f, indent=2)
    print(f"  Saved to {results_path}")
    
    # Save condition summaries
    conditions_path = args.save_dir / "condition_summaries.json"
    with open(conditions_path, "w") as f:
        json.dump(condition_summaries, f, indent=2)
    print(f"  Saved condition summaries to {conditions_path}")
    
    print("\n" + "=" * 60)
    print("ANOVA EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return results, summary


if __name__ == "__main__":
    main()
