"""
Refined ANOVA experiment using only context-agnostic tokens.

The key insight: some features' top tokens only appear in reasoning text,
confounding the token/context separation. This script uses only tokens
that appear in BOTH reasoning and non-reasoning text.

If a feature still shows high activation for these tokens in reasoning context
but low in non-reasoning context, then it's truly context-dependent, not token-driven.

If activation is high in both contexts, the feature is token-driven.

## Usage

```bash
python run_anova_refined.py \\
    --token-analysis results/test/token_analysis.json \\
    --layer 12 \\
    --top-k-features 5 \\
    --save-dir results/test/refined
```
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run refined ANOVA using context-agnostic tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--token-analysis", type=Path, required=True)
    parser.add_argument("--model-name", default="google/gemma-2-9b")
    parser.add_argument("--sae-name", default="gemma-scope-9b-pt-res-canonical")
    parser.add_argument("--sae-id-format", default="layer_{layer}/width_16k/canonical")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--top-k-features", type=int, default=10)
    parser.add_argument("--min-occurrences-both", type=int, default=5,
                        help="Min occurrences in both contexts for a token to be used")
    parser.add_argument("--n-reasoning-texts", type=int, default=1000)
    parser.add_argument("--n-nonreasoning-texts", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    
    return parser.parse_args()


def load_context_agnostic_tokens(
    token_analysis_path: Path,
    feature_index: int,
    min_occurrences_both: int = 5,
) -> tuple[set[int], dict]:
    """Load tokens that appear in both reasoning and non-reasoning contexts."""
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    for feature in data.get('features', []):
        if feature.get('feature_index') == feature_index:
            agnostic_tokens = set()
            token_stats = []
            
            for t in feature.get('top_tokens', []):
                r_count = t.get('occurrence_count_reasoning', 0)
                nr_count = t.get('occurrence_count_nonreasoning', 0)
                
                if r_count >= min_occurrences_both and nr_count >= min_occurrences_both:
                    agnostic_tokens.add(t.get('token_id'))
                    token_stats.append({
                        'token_id': t.get('token_id'),
                        'token_str': t.get('token_str'),
                        'mean_activation': t.get('mean_activation'),
                        'mean_activation_in_reasoning': t.get('mean_activation_in_reasoning'),
                        'mean_activation_in_nonreasoning': t.get('mean_activation_in_nonreasoning'),
                        'occurrence_count_reasoning': r_count,
                        'occurrence_count_nonreasoning': nr_count,
                    })
            
            return agnostic_tokens, {
                'n_agnostic_tokens': len(agnostic_tokens),
                'n_total_tokens': len(feature.get('top_tokens', [])),
                'token_stats': token_stats,
            }
    
    return set(), {}


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
    """Collect token-level activations."""
    all_activations = []
    all_is_reasoning_token = []
    all_is_reasoning_text = []
    all_token_ids = []
    
    n_texts = len(texts)
    
    for batch_start in range(0, n_texts, batch_size):
        batch_end = min(batch_start + batch_size, n_texts)
        batch_texts = texts[batch_start:batch_end]
        batch_is_reasoning = is_reasoning[batch_start:batch_end]
        
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=[f"blocks.{layer_index}.hook_resid_post"],
            )
            
            resid = cache[f"blocks.{layer_index}.hook_resid_post"]
            sae_acts = sae.encode(resid)
            feature_acts = sae_acts[:, :, feature_index]
        
        input_ids_np = input_ids.cpu().numpy()
        attention_mask_np = attention_mask.cpu().numpy()
        feature_acts_np = feature_acts.cpu().numpy()
        
        seq_len = input_ids_np.shape[1]
        
        for i in range(len(batch_texts)):
            for j in range(seq_len):
                if attention_mask_np[i, j] == 0:
                    continue
                
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


def compute_refined_anova(data: dict, feature_index: int) -> dict:
    """Compute ANOVA with focus on context-agnostic interpretation."""
    acts = data["activations"]
    is_rt = data["is_reasoning_token"]
    is_rc = data["is_reasoning_text"]
    
    # Split into 4 cells
    a_A = acts[is_rt & is_rc]      # Agnostic token + Reasoning context
    a_B = acts[~is_rt & is_rc]     # Other token + Reasoning context  
    a_C = acts[is_rt & ~is_rc]     # Agnostic token + Non-reasoning context
    a_D = acts[~is_rt & ~is_rc]    # Other token + Non-reasoning context
    
    n_A, n_B, n_C, n_D = len(a_A), len(a_B), len(a_C), len(a_D)
    n_total = n_A + n_B + n_C + n_D
    
    if min(n_A, n_B, n_C, n_D) < 10:
        return {
            "feature_index": feature_index,
            "error": "Insufficient samples",
            "cell_sizes": {"A": n_A, "B": n_B, "C": n_C, "D": n_D},
        }
    
    mean_A = np.mean(a_A)
    mean_B = np.mean(a_B)
    mean_C = np.mean(a_C)
    mean_D = np.mean(a_D)
    
    # The key comparison for token-driven features:
    # If token-driven: mean_A ≈ mean_C (same token, different context, similar activation)
    # If context-driven: mean_A >> mean_C (same token, different context, different activation)
    
    token_effect_in_reasoning = mean_A - mean_B  # Does the token matter in reasoning?
    token_effect_in_nonreasoning = mean_C - mean_D  # Does the token matter outside reasoning?
    
    context_effect_with_token = mean_A - mean_C  # Does context matter for these tokens?
    context_effect_without_token = mean_B - mean_D  # Does context matter for other tokens?
    
    # Consistency metric: if token-driven, effects should be similar across contexts
    token_consistency = 1 - abs(token_effect_in_reasoning - token_effect_in_nonreasoning) / (
        max(abs(token_effect_in_reasoning), abs(token_effect_in_nonreasoning)) + 0.01
    )
    
    # Standard ANOVA calculations
    mean_rt = (n_A * mean_A + n_C * mean_C) / (n_A + n_C) if (n_A + n_C) > 0 else 0
    mean_ot = (n_B * mean_B + n_D * mean_D) / (n_B + n_D) if (n_B + n_D) > 0 else 0
    mean_rc = (n_A * mean_A + n_B * mean_B) / (n_A + n_B) if (n_A + n_B) > 0 else 0
    mean_nrc = (n_C * mean_C + n_D * mean_D) / (n_C + n_D) if (n_C + n_D) > 0 else 0
    grand_mean = (n_A * mean_A + n_B * mean_B + n_C * mean_C + n_D * mean_D) / n_total
    
    ss_token = (n_A + n_C) * (mean_rt - grand_mean)**2 + (n_B + n_D) * (mean_ot - grand_mean)**2
    ss_context = (n_A + n_B) * (mean_rc - grand_mean)**2 + (n_C + n_D) * (mean_nrc - grand_mean)**2
    ss_error = np.sum((a_A - mean_A)**2) + np.sum((a_B - mean_B)**2) + \
               np.sum((a_C - mean_C)**2) + np.sum((a_D - mean_D)**2)
    
    all_acts = np.concatenate([a_A, a_B, a_C, a_D])
    ss_total = np.sum((all_acts - grand_mean)**2)
    ss_interaction = max(0, ss_total - ss_token - ss_context - ss_error)
    
    eta_sq_token = ss_token / ss_total if ss_total > 0 else 0
    eta_sq_context = ss_context / ss_total if ss_total > 0 else 0
    eta_sq_interaction = ss_interaction / ss_total if ss_total > 0 else 0
    
    # Decision logic
    # Token-driven: token explains more AND token effect is consistent across contexts
    # Context-driven: context explains more AND token effect varies by context
    
    is_token_dominated = bool(
        (eta_sq_token > 2 * eta_sq_context) and 
        (eta_sq_token > 0.01) and
        (token_consistency > 0.5)
    )
    
    is_context_dominated = bool(
        (eta_sq_context > 2 * eta_sq_token) and 
        (eta_sq_context > 0.01) and
        (token_consistency < 0.5)
    )
    
    # New: check if it's CONFOUNDED (token and context are correlated)
    is_confounded = bool(
        (eta_sq_context > eta_sq_token) and 
        (context_effect_with_token > context_effect_without_token * 1.5)
    )
    
    if is_token_dominated:
        dominant_factor = "token"
    elif is_confounded:
        dominant_factor = "confounded"
    elif is_context_dominated:
        dominant_factor = "context"
    else:
        dominant_factor = "mixed"
    
    return {
        "feature_index": feature_index,
        "cell_sizes": {"A": n_A, "B": n_B, "C": n_C, "D": n_D},
        "cell_means": {
            "A_agnostic_token_reasoning": float(mean_A),
            "B_other_token_reasoning": float(mean_B),
            "C_agnostic_token_nonreasoning": float(mean_C),
            "D_other_token_nonreasoning": float(mean_D),
        },
        "effects": {
            "token_effect_in_reasoning": float(token_effect_in_reasoning),
            "token_effect_in_nonreasoning": float(token_effect_in_nonreasoning),
            "context_effect_with_token": float(context_effect_with_token),
            "context_effect_without_token": float(context_effect_without_token),
            "token_consistency": float(token_consistency),
        },
        "eta_sq_token": float(eta_sq_token),
        "eta_sq_context": float(eta_sq_context),
        "eta_sq_interaction": float(eta_sq_interaction),
        "is_token_dominated": is_token_dominated,
        "is_context_dominated": is_context_dominated,
        "is_confounded": is_confounded,
        "dominant_factor": dominant_factor,
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("REFINED ANOVA: Context-Agnostic Tokens Only")
    print("=" * 60)
    print(f"Token analysis: {args.token_analysis}")
    print(f"Layer: {args.layer}")
    print(f"Min occurrences in both contexts: {args.min_occurrences_both}")
    print("=" * 60)
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature indices
    with open(args.token_analysis) as f:
        token_data = json.load(f)
    
    feature_indices = [f["feature_index"] for f in token_data.get("features", [])][:args.top_k_features]
    
    # Load datasets
    print("\n--- Loading Datasets ---")
    from reasoning_features.datasets.anova import split_into_sentences
    from datasets import load_dataset
    
    print("Loading reasoning data...")
    reasoning_ds = load_dataset("simplescaling/s1K-1.1", split="train")
    reasoning_texts = []
    for row in reasoning_ds:
        text = row.get("gemini_thinking_trajectory", "")
        if text:
            sentences = split_into_sentences(text, min_length=50, max_length=300)
            reasoning_texts.extend(sentences)
        if len(reasoning_texts) >= args.n_reasoning_texts:
            break
    reasoning_texts = reasoning_texts[:args.n_reasoning_texts]
    print(f"  Got {len(reasoning_texts)} reasoning texts")
    
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
    
    # Run analysis
    print("\n--- Running Refined ANOVA ---")
    results = []
    
    for feat_idx in tqdm(feature_indices, desc="Analyzing features"):
        # Get context-agnostic tokens
        agnostic_tokens, token_info = load_context_agnostic_tokens(
            args.token_analysis, feat_idx, args.min_occurrences_both
        )
        
        if len(agnostic_tokens) < 5:
            print(f"\nFeature {feat_idx}: Only {len(agnostic_tokens)} context-agnostic tokens, skipping")
            results.append({
                "feature_index": feat_idx,
                "error": f"Only {len(agnostic_tokens)} context-agnostic tokens",
                "token_info": token_info,
            })
            continue
        
        print(f"\nFeature {feat_idx}: {len(agnostic_tokens)} context-agnostic tokens out of {token_info['n_total_tokens']}")
        
        # Sample top tokens
        sample_tokens = token_info['token_stats'][:5]
        for t in sample_tokens:
            print(f"  '{t['token_str']}': R={t['mean_activation_in_reasoning']:.2f}, NR={t['mean_activation_in_nonreasoning']:.2f}")
        
        # Collect activations
        data = collect_tokenwise_activations(
            all_texts,
            is_reasoning,
            agnostic_tokens,
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
        result = compute_refined_anova(data, feat_idx)
        result["token_info"] = token_info
        results.append(result)
        
        if "error" not in result:
            print(f"  Cells: A={result['cell_sizes']['A']}, B={result['cell_sizes']['B']}, "
                  f"C={result['cell_sizes']['C']}, D={result['cell_sizes']['D']}")
            print(f"  Means: A={result['cell_means']['A_agnostic_token_reasoning']:.3f}, "
                  f"B={result['cell_means']['B_other_token_reasoning']:.3f}, "
                  f"C={result['cell_means']['C_agnostic_token_nonreasoning']:.3f}, "
                  f"D={result['cell_means']['D_other_token_nonreasoning']:.3f}")
            print(f"  η²_token={result['eta_sq_token']:.4f}, η²_context={result['eta_sq_context']:.4f}")
            print(f"  Token effect (R): {result['effects']['token_effect_in_reasoning']:.3f}, "
                  f"Token effect (NR): {result['effects']['token_effect_in_nonreasoning']:.3f}")
            print(f"  Token consistency: {result['effects']['token_consistency']:.3f}")
            print(f"  Dominant factor: {result['dominant_factor']}")
    
    # Summary
    print("\n--- Summary ---")
    valid = [r for r in results if "error" not in r]
    
    if valid:
        summary = {
            "n_features_analyzed": len(valid),
            "n_token_dominated": sum(1 for r in valid if r["is_token_dominated"]),
            "n_context_dominated": sum(1 for r in valid if r["is_context_dominated"]),
            "n_confounded": sum(1 for r in valid if r.get("is_confounded", False)),
            "dominant_distribution": {},
            "mean_eta_sq_token": np.mean([r["eta_sq_token"] for r in valid]),
            "mean_eta_sq_context": np.mean([r["eta_sq_context"] for r in valid]),
            "mean_token_consistency": np.mean([r["effects"]["token_consistency"] for r in valid]),
        }
        
        for r in valid:
            d = r["dominant_factor"]
            summary["dominant_distribution"][d] = summary["dominant_distribution"].get(d, 0) + 1
        
        print(f"Features analyzed: {summary['n_features_analyzed']}")
        print(f"Token-dominated: {summary['n_token_dominated']}")
        print(f"Context-dominated: {summary['n_context_dominated']}")
        print(f"Confounded: {summary['n_confounded']}")
        print(f"Distribution: {summary['dominant_distribution']}")
        print(f"Mean token consistency: {summary['mean_token_consistency']:.3f}")
    else:
        summary = {"error": "No valid results"}
    
    # Save
    print("\n--- Saving Results ---")
    with open(args.save_dir / "refined_anova_results.json", "w") as f:
        json.dump({
            "config": vars(args),
            "summary": summary,
            "features": results,
        }, f, indent=2, default=str)
    
    print(f"Saved to {args.save_dir}")
    
    print("\n" + "=" * 60)
    print("REFINED ANOVA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
