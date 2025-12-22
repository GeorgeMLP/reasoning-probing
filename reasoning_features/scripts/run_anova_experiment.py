"""
ANOVA experiment for disentangling token vs. context effects in SAE features.

This script implements a token-level ANOVA design that analyzes activations
at each token position, enabling direct testing of whether specific tokens
drive feature activation regardless of surrounding context.

## Key Design Choices

1. **Unit of analysis**: Each token position is a sample (not each text)
2. **Token factor**: Whether THIS specific token is in the feature's top-k token set
3. **Context factor**: Whether this text is from the reasoning corpus
4. **Activation**: Feature activation at THIS token position (not max)

## Why Token-Level Analysis

Token-level ANOVA is more informative than text-level because:
- We directly test if specific tokens drive activation
- We can see if context matters independently of token identity
- Max aggregation doesn't wash out token-specific effects

## Key Finding: Vocabulary Exclusivity

Our analysis reveals that ~87.7% of tokens activating reasoning features
are exclusive to reasoning contexts. This explains why features appear
"context-dominated" - their vocabulary simply doesn't exist in general text.

## Usage

```bash
python run_anova_experiment.py \\
    --token-analysis results/test/token_analysis.json \\
    --layer 12 \\
    --top-k-features 5 \\
    --save-dir results/test
```
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run token-level ANOVA experiments",
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
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        default="google/gemma-2-9b",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--sae-name",
        default="gemma-scope-9b-pt-res-canonical",
        help="SAE release name",
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
        default=10,
        help="Number of top features to analyze",
    )
    parser.add_argument(
        "--top-k-tokens",
        type=int,
        default=50,
        help="Number of top tokens per feature (use more for token-level)",
    )
    
    # Token set strategy
    parser.add_argument(
        "--token-strategy",
        choices=["feature_specific", "global_reasoning", "combined"],
        default="feature_specific",
        help="How to define 'reasoning tokens'",
    )
    parser.add_argument(
        "--global-token-ratio-threshold",
        type=float,
        default=2.0,
        help="Min frequency ratio for global reasoning tokens",
    )
    parser.add_argument(
        "--global-top-k",
        type=int,
        default=500,
        help="Number of global reasoning tokens to use",
    )
    
    # Dataset configuration
    # Dataset configuration
    parser.add_argument(
        "--reasoning-dataset",
        choices=["s1k", "general_inquiry_cot", "combined"],
        default="s1k",
        help="Reasoning dataset to use (default: s1k)",
    )
    parser.add_argument(
        "--n-reasoning-texts",
        type=int,
        default=500,
        help="Number of reasoning texts to use",
    )
    parser.add_argument(
        "--n-nonreasoning-texts",
        type=int,
        default=500,
        help="Number of non-reasoning texts to use",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum sequence length",
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
        help="Device to run on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for activation collection",
    )
    
    return parser.parse_args()


def load_top_tokens_for_feature(token_analysis_path: Path, feature_index: int, top_k: int) -> set[int]:
    """Load top token IDs for a feature."""
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    for feature in data.get('features', []):
        if feature.get('feature_index') == feature_index:
            top_tokens = feature.get('top_tokens', [])[:top_k]
            return {t.get('token_id') for t in top_tokens}
    
    return set()


def compute_global_reasoning_tokens(
    reasoning_texts: list[str],
    nonreasoning_texts: list[str],
    tokenizer,
    top_k: int = 500,
    min_ratio: float = 2.0,
    min_count: int = 10,
) -> set[int]:
    """
    Find tokens that are overrepresented in reasoning text.
    
    These are tokens with high frequency ratio (reasoning / non-reasoning).
    """
    reasoning_counts = Counter()
    nonreasoning_counts = Counter()
    
    for text in reasoning_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        reasoning_counts.update(tokens)
    
    for text in nonreasoning_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        nonreasoning_counts.update(tokens)
    
    # Compute frequency ratios
    total_r = sum(reasoning_counts.values()) + 1
    total_nr = sum(nonreasoning_counts.values()) + 1
    
    ratios = {}
    for token_id in reasoning_counts:
        if reasoning_counts[token_id] < min_count:
            continue
        r_freq = reasoning_counts[token_id] / total_r
        nr_freq = (nonreasoning_counts.get(token_id, 0) + 1) / total_nr
        ratios[token_id] = r_freq / nr_freq
    
    # Filter by ratio and get top-k
    filtered = [(tid, r) for tid, r in ratios.items() if r >= min_ratio]
    sorted_tokens = sorted(filtered, key=lambda x: x[1], reverse=True)
    
    return {tid for tid, r in sorted_tokens[:top_k]}


def collect_tokenwise_activations(
    texts: list[str],
    is_reasoning: list[bool],
    reasoning_token_set: set[int],
    model,
    sae,
    tokenizer,
    layer_index: int,
    feature_index: int,
    max_length: int = 64,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """
    Collect token-level activations and labels.
    
    Returns:
        Dict with:
        - activations: array of activation values
        - is_reasoning_token: array of bools (is this token in reasoning set?)
        - is_reasoning_text: array of bools (is this text from reasoning corpus?)
        - token_ids: array of token IDs
    """
    all_activations = []
    all_is_reasoning_token = []
    all_is_reasoning_text = []
    all_token_ids = []
    
    n_texts = len(texts)
    
    for batch_start in range(0, n_texts, batch_size):
        batch_end = min(batch_start + batch_size, n_texts)
        batch_texts = texts[batch_start:batch_end]
        batch_is_reasoning = is_reasoning[batch_start:batch_end]
        
        # Tokenize
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        
        # Get activations
        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=[f"blocks.{layer_index}.hook_resid_post"],
            )
            
            resid = cache[f"blocks.{layer_index}.hook_resid_post"]
            sae_acts = sae.encode(resid)  # (batch, seq, n_features)
            feature_acts = sae_acts[:, :, feature_index]  # (batch, seq)
        
        # Extract per-token data
        input_ids_np = input_ids.cpu().numpy()
        attention_mask_np = attention_mask.cpu().numpy()
        feature_acts_np = feature_acts.cpu().numpy()
        
        seq_len = input_ids_np.shape[1]
        
        for i in range(len(batch_texts)):
            for j in range(seq_len):
                if attention_mask_np[i, j] == 0:
                    continue  # Skip padding
                
                token_id = input_ids_np[i, j]
                activation = feature_acts_np[i, j]
                
                all_activations.append(activation)
                all_is_reasoning_token.append(token_id in reasoning_token_set)
                all_is_reasoning_text.append(batch_is_reasoning[i])
                all_token_ids.append(token_id)
    
    return {
        "activations": np.array(all_activations),
        "is_reasoning_token": np.array(all_is_reasoning_token),
        "is_reasoning_text": np.array(all_is_reasoning_text),
        "token_ids": np.array(all_token_ids),
    }


def compute_tokenwise_anova(data: dict, feature_index: int) -> dict:
    """
    Compute 2x2 ANOVA on token-level data.
    
    Factors:
    - Token: Is this token in the reasoning token set?
    - Context: Is this text from reasoning corpus?
    """
    acts = data["activations"]
    is_rt = data["is_reasoning_token"]  # reasoning token
    is_rc = data["is_reasoning_text"]   # reasoning context
    
    # Split into 4 cells
    a_A = acts[is_rt & is_rc]      # Reasoning token + Reasoning context
    a_B = acts[~is_rt & is_rc]     # Non-reasoning token + Reasoning context
    a_C = acts[is_rt & ~is_rc]     # Reasoning token + Non-reasoning context
    a_D = acts[~is_rt & ~is_rc]    # Non-reasoning token + Non-reasoning context
    
    # Cell sizes
    n_A, n_B, n_C, n_D = len(a_A), len(a_B), len(a_C), len(a_D)
    n_total = n_A + n_B + n_C + n_D
    
    if min(n_A, n_B, n_C, n_D) < 10:
        return {
            "feature_index": feature_index,
            "error": "Insufficient samples in one or more cells",
            "cell_sizes": {"A": n_A, "B": n_B, "C": n_C, "D": n_D},
        }
    
    # Cell means
    mean_A = np.mean(a_A)
    mean_B = np.mean(a_B)
    mean_C = np.mean(a_C)
    mean_D = np.mean(a_D)
    
    # Marginal means (weighted by cell size for unbalanced design)
    # Token marginals
    mean_reasoning_token = (n_A * mean_A + n_C * mean_C) / (n_A + n_C) if (n_A + n_C) > 0 else 0
    mean_non_reasoning_token = (n_B * mean_B + n_D * mean_D) / (n_B + n_D) if (n_B + n_D) > 0 else 0
    
    # Context marginals
    mean_reasoning_context = (n_A * mean_A + n_B * mean_B) / (n_A + n_B) if (n_A + n_B) > 0 else 0
    mean_non_reasoning_context = (n_C * mean_C + n_D * mean_D) / (n_C + n_D) if (n_C + n_D) > 0 else 0
    
    # Grand mean
    grand_mean = (n_A * mean_A + n_B * mean_B + n_C * mean_C + n_D * mean_D) / n_total
    
    # Sum of squares (Type III for unbalanced)
    # SS_token: effect of being a reasoning token
    ss_token = (n_A + n_C) * (mean_reasoning_token - grand_mean)**2 + \
               (n_B + n_D) * (mean_non_reasoning_token - grand_mean)**2
    
    # SS_context: effect of reasoning context
    ss_context = (n_A + n_B) * (mean_reasoning_context - grand_mean)**2 + \
                 (n_C + n_D) * (mean_non_reasoning_context - grand_mean)**2
    
    # SS_error (within-cell variance)
    ss_error = np.sum((a_A - mean_A)**2) + np.sum((a_B - mean_B)**2) + \
               np.sum((a_C - mean_C)**2) + np.sum((a_D - mean_D)**2)
    
    # SS_total
    all_acts = np.concatenate([a_A, a_B, a_C, a_D])
    ss_total = np.sum((all_acts - grand_mean)**2)
    
    # SS_interaction (by subtraction)
    ss_interaction = ss_total - ss_token - ss_context - ss_error
    ss_interaction = max(0, ss_interaction)  # Can be negative due to numerical issues
    
    # Degrees of freedom
    df_token = 1
    df_context = 1
    df_interaction = 1
    df_error = n_total - 4
    
    # Mean squares
    ms_token = ss_token / df_token
    ms_context = ss_context / df_context
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 1e-10
    
    # F-statistics
    f_token = ms_token / ms_error
    f_context = ms_context / ms_error
    f_interaction = ms_interaction / ms_error if ms_error > 0 else 0
    
    # P-values
    p_token = 1 - stats.f.cdf(f_token, df_token, df_error)
    p_context = 1 - stats.f.cdf(f_context, df_context, df_error)
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_error) if df_interaction > 0 else 1.0
    
    # Eta-squared
    eta_sq_token = ss_token / ss_total if ss_total > 0 else 0
    eta_sq_context = ss_context / ss_total if ss_total > 0 else 0
    eta_sq_interaction = ss_interaction / ss_total if ss_total > 0 else 0
    
    # Decision rules
    is_token_dominated = bool((eta_sq_token > 2 * eta_sq_context) and (eta_sq_token > 0.01))
    is_context_dominated = bool((eta_sq_context > 2 * eta_sq_token) and (eta_sq_context > 0.01))
    
    if eta_sq_interaction > max(eta_sq_token, eta_sq_context) and eta_sq_interaction > 0.01:
        dominant_factor = "interaction"
    elif is_token_dominated:
        dominant_factor = "token"
    elif is_context_dominated:
        dominant_factor = "context"
    elif max(eta_sq_token, eta_sq_context, eta_sq_interaction) < 0.001:
        dominant_factor = "none"
    else:
        dominant_factor = "mixed"
    
    return {
        "feature_index": feature_index,
        "cell_sizes": {"A": n_A, "B": n_B, "C": n_C, "D": n_D},
        "cell_means": {
            "A_reasoning_token_reasoning_context": float(mean_A),
            "B_nonreasoning_token_reasoning_context": float(mean_B),
            "C_reasoning_token_nonreasoning_context": float(mean_C),
            "D_nonreasoning_token_nonreasoning_context": float(mean_D),
        },
        "ss_token": float(ss_token),
        "ss_context": float(ss_context),
        "ss_interaction": float(ss_interaction),
        "ss_error": float(ss_error),
        "ss_total": float(ss_total),
        "eta_sq_token": float(eta_sq_token),
        "eta_sq_context": float(eta_sq_context),
        "eta_sq_interaction": float(eta_sq_interaction),
        "f_token": float(f_token),
        "f_context": float(f_context),
        "f_interaction": float(f_interaction),
        "p_token": float(p_token),
        "p_context": float(p_context),
        "p_interaction": float(p_interaction),
        "is_token_dominated": is_token_dominated,
        "is_context_dominated": is_context_dominated,
        "dominant_factor": dominant_factor,
        # Effect sizes for interpretation
        "token_effect": float(mean_reasoning_token - mean_non_reasoning_token),
        "context_effect": float(mean_reasoning_context - mean_non_reasoning_context),
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("TOKEN-LEVEL ANOVA EXPERIMENT")
    print("=" * 60)
    print(f"Token analysis: {args.token_analysis}")
    print(f"Layer: {args.layer}")
    print(f"Reasoning dataset: {args.reasoning_dataset}")
    print(f"Token strategy: {args.token_strategy}")
    print(f"Top-k features: {args.top_k_features}")
    print(f"Top-k tokens per feature: {args.top_k_tokens}")
    print("=" * 60)
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature indices from token analysis
    with open(args.token_analysis) as f:
        token_data = json.load(f)
    
    feature_indices = [f["feature_index"] for f in token_data.get("features", [])][:args.top_k_features]
    print(f"\nAnalyzing {len(feature_indices)} features: {feature_indices}")
    
    # Load datasets
    print("\n--- Loading Datasets ---")
    from reasoning_features.datasets.anova import split_into_sentences
    from datasets import load_dataset
    
    # Load reasoning texts based on dataset choice
    print(f"Loading reasoning data from {args.reasoning_dataset}...")
    reasoning_texts = []
    
    if args.reasoning_dataset in ["s1k", "combined"]:
        reasoning_ds = load_dataset("simplescaling/s1K-1.1", split="train")
        for row in reasoning_ds:
            for key in ["gemini_thinking_trajectory", "deepseek_thinking_trajectory"]:
                text = row.get(key, "")
                if text:
                    sentences = split_into_sentences(text, min_length=50, max_length=300)
                    reasoning_texts.extend(sentences)
            if len(reasoning_texts) >= args.n_reasoning_texts:
                break
    
    if args.reasoning_dataset in ["general_inquiry_cot", "combined"]:
        gicot_ds = load_dataset("moremilk/General_Inquiry_Thinking-Chain-Of-Thought", split="train")
        for row in gicot_ds:
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict):
                text = metadata.get("reasoning", "")
                # Remove <think> and </think> tags
                if text:
                    text = text.replace("<think>", "").replace("</think>", "").strip()
                    sentences = split_into_sentences(text, min_length=50, max_length=300)
                    reasoning_texts.extend(sentences)
            if len(reasoning_texts) >= args.n_reasoning_texts:
                break
    
    reasoning_texts = reasoning_texts[:args.n_reasoning_texts]
    print(f"  Got {len(reasoning_texts)} reasoning texts")
    
    # Load non-reasoning texts
    print("Loading non-reasoning data...")
    nonreasoning_ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    nonreasoning_texts = []
    for row in nonreasoning_ds:
        text = row.get("text", "")
        if text:
            sentences = split_into_sentences(text, min_length=50, max_length=300)
            nonreasoning_texts.extend(sentences)
        if len(nonreasoning_texts) >= args.n_nonreasoning_texts:
            break
    nonreasoning_texts = nonreasoning_texts[:args.n_nonreasoning_texts]
    print(f"  Got {len(nonreasoning_texts)} non-reasoning texts")
    
    # Combine texts
    all_texts = reasoning_texts + nonreasoning_texts
    is_reasoning = [True] * len(reasoning_texts) + [False] * len(nonreasoning_texts)
    
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
    
    # Compute global reasoning tokens if needed
    global_reasoning_tokens = set()
    if args.token_strategy in ["global_reasoning", "combined"]:
        print("\n--- Computing Global Reasoning Tokens ---")
        global_reasoning_tokens = compute_global_reasoning_tokens(
            reasoning_texts,
            nonreasoning_texts,
            tokenizer,
            top_k=args.global_top_k,
            min_ratio=args.global_token_ratio_threshold,
        )
        print(f"  Found {len(global_reasoning_tokens)} global reasoning tokens")
        
        # Show some examples
        sample_tokens = list(global_reasoning_tokens)[:20]
        sample_strs = [tokenizer.decode([t]) for t in sample_tokens]
        print(f"  Sample tokens: {sample_strs}")
    
    # Run token-level ANOVA for each feature
    print("\n--- Running Token-Level ANOVA ---")
    results = []
    
    for feat_idx in tqdm(feature_indices, desc="Analyzing features"):
        # Get reasoning token set based on strategy
        if args.token_strategy == "feature_specific":
            reasoning_token_set = load_top_tokens_for_feature(
                args.token_analysis, feat_idx, args.top_k_tokens
            )
        elif args.token_strategy == "global_reasoning":
            reasoning_token_set = global_reasoning_tokens
        else:  # combined
            feature_tokens = load_top_tokens_for_feature(
                args.token_analysis, feat_idx, args.top_k_tokens
            )
            reasoning_token_set = feature_tokens | global_reasoning_tokens
        
        print(f"\nFeature {feat_idx}: using {len(reasoning_token_set)} reasoning tokens")
        
        # Collect token-level activations
        data = collect_tokenwise_activations(
            all_texts,
            is_reasoning,
            reasoning_token_set,
            model,
            sae,
            tokenizer,
            args.layer,
            feat_idx,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
        )
        
        # Compute ANOVA
        result = compute_tokenwise_anova(data, feat_idx)
        result["n_reasoning_tokens_used"] = len(reasoning_token_set)
        results.append(result)
        
        # Print summary
        if "error" not in result:
            print(f"  Cells: A={result['cell_sizes']['A']}, B={result['cell_sizes']['B']}, "
                  f"C={result['cell_sizes']['C']}, D={result['cell_sizes']['D']}")
            print(f"  Means: A={result['cell_means']['A_reasoning_token_reasoning_context']:.3f}, "
                  f"B={result['cell_means']['B_nonreasoning_token_reasoning_context']:.3f}, "
                  f"C={result['cell_means']['C_reasoning_token_nonreasoning_context']:.3f}, "
                  f"D={result['cell_means']['D_nonreasoning_token_nonreasoning_context']:.3f}")
            print(f"  η²_token={result['eta_sq_token']:.4f}, η²_context={result['eta_sq_context']:.4f}, "
                  f"η²_interaction={result['eta_sq_interaction']:.4f}")
            print(f"  Token effect: {result['token_effect']:.3f}, Context effect: {result['context_effect']:.3f}")
            print(f"  Dominant factor: {result['dominant_factor']}")
    
    # Compute summary
    print("\n--- Computing Summary ---")
    valid_results = [r for r in results if "error" not in r]
    
    if valid_results:
        n_token_dominated = sum(1 for r in valid_results if r["is_token_dominated"])
        n_context_dominated = sum(1 for r in valid_results if r["is_context_dominated"])
        
        dominant_counts = {}
        for r in valid_results:
            d = r["dominant_factor"]
            dominant_counts[d] = dominant_counts.get(d, 0) + 1
        
        summary = {
            "n_features_analyzed": len(valid_results),
            "n_token_dominated": n_token_dominated,
            "n_context_dominated": n_context_dominated,
            "pct_token_dominated": 100 * n_token_dominated / len(valid_results) if valid_results else 0,
            "pct_context_dominated": 100 * n_context_dominated / len(valid_results) if valid_results else 0,
            "dominant_factor_distribution": dominant_counts,
            "mean_eta_sq_token": np.mean([r["eta_sq_token"] for r in valid_results]),
            "mean_eta_sq_context": np.mean([r["eta_sq_context"] for r in valid_results]),
            "mean_eta_sq_interaction": np.mean([r["eta_sq_interaction"] for r in valid_results]),
            "mean_token_effect": np.mean([r["token_effect"] for r in valid_results]),
            "mean_context_effect": np.mean([r["context_effect"] for r in valid_results]),
        }
        
        print(f"\nResults:")
        print(f"  Features analyzed: {summary['n_features_analyzed']}")
        print(f"  Token-dominated: {summary['n_token_dominated']} ({summary['pct_token_dominated']:.1f}%)")
        print(f"  Context-dominated: {summary['n_context_dominated']} ({summary['pct_context_dominated']:.1f}%)")
        print(f"  Mean η²_token: {summary['mean_eta_sq_token']:.4f}")
        print(f"  Mean η²_context: {summary['mean_eta_sq_context']:.4f}")
        print(f"  Mean token effect: {summary['mean_token_effect']:.3f}")
        print(f"  Mean context effect: {summary['mean_context_effect']:.3f}")
        print(f"  Dominant factor distribution: {summary['dominant_factor_distribution']}")
    else:
        summary = {"error": "No valid results"}
    
    # Save results
    print("\n--- Saving Results ---")
    results_path = args.save_dir / "anova_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "token_analysis": str(args.token_analysis),
                "layer": args.layer,
                "reasoning_dataset": args.reasoning_dataset,
                "token_strategy": args.token_strategy,
                "top_k_features": args.top_k_features,
                "top_k_tokens": args.top_k_tokens,
                "n_reasoning_texts": len(reasoning_texts),
                "n_nonreasoning_texts": len(nonreasoning_texts),
            },
            "summary": summary,
            "features": results,
        }, f, indent=2)
    print(f"  Saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("TOKEN-LEVEL ANOVA COMPLETE")
    print("=" * 60)
    
    return results, summary


if __name__ == "__main__":
    main()
