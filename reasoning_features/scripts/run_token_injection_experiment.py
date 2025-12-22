"""
Token Injection Experiment: Testing whether features are token-driven.

This experiment directly tests the causal hypothesis: if a feature is truly
a "token detector", then injecting those tokens into non-reasoning text
should activate the feature.

## Experimental Design

1. Take non-reasoning text samples
2. Inject the feature's top-k tokens into the text
3. Measure feature activation before and after injection
4. Compare to activation on actual reasoning text

## Key Metrics

- **Activation Increase**: How much does injection increase activation?
- **Transfer Ratio**: (Injected activation) / (Reasoning activation)
- **Injection Effectiveness**: Does injection achieve reasoning-level activation?

## Interpretation

If feature is TOKEN-DRIVEN:
- Injection should significantly increase activation
- Transfer ratio should be high (>0.5)

If feature captures REASONING STRUCTURE:
- Injection may not significantly increase activation
- Transfer ratio will be low (<0.2)

## Usage

```bash
python run_token_injection_experiment.py \\
    --token-analysis results/layer8/token_analysis.json \\
    --reasoning-features results/layer8/reasoning_features.json \\
    --layer 8 \\
    --top-k-features 10 \\
    --save-dir results/layer8/injection
```
"""

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_top_tokens_for_feature(token_analysis_path: str, feature_index: int, top_k: int = 30) -> list:
    """Load top-k tokens for a specific feature."""
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    for feature in data.get("features", []):
        if feature.get("feature_index") == feature_index:
            tokens = feature.get("top_tokens", [])[:top_k]
            return [t["token_str"] for t in tokens]
    
    return []


def inject_tokens_into_text(text: str, tokens: list, n_inject: int = 3, strategy: str = "prepend") -> str:
    """Inject tokens into text using various strategies."""
    selected_tokens = random.sample(tokens, min(n_inject, len(tokens)))
    
    if strategy == "prepend":
        # Add tokens at the beginning
        injection = " ".join(selected_tokens) + " "
        return injection + text
    
    elif strategy == "append":
        # Add tokens at the end
        injection = " " + " ".join(selected_tokens)
        return text + injection
    
    elif strategy == "intersperse":
        # Spread tokens throughout the text
        words = text.split()
        if len(words) < 2:
            return " ".join(selected_tokens) + " " + text
        
        # Insert tokens at random positions
        for token in selected_tokens:
            pos = random.randint(0, len(words))
            words.insert(pos, token)
        return " ".join(words)
    
    elif strategy == "replace":
        # Replace random words with tokens
        words = text.split()
        if len(words) < len(selected_tokens):
            return " ".join(selected_tokens)
        
        positions = random.sample(range(len(words)), len(selected_tokens))
        for pos, token in zip(positions, selected_tokens):
            words[pos] = token
        return " ".join(words)
    
    return text


def get_feature_activation(model, sae, tokenizer, texts: list, layer: int, 
                           feature_index: int, device: str, batch_size: int = 32) -> np.ndarray:
    """Get feature activations for a batch of texts."""
    activations = []
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, stop_at_layer=layer + 1)
            hidden = cache[hook_name]
            sae_out = sae.encode(hidden)
            
            # Get max activation per text for the target feature
            for b in range(sae_out.shape[0]):
                seq_len = int(attention_mask[b].sum().item())
                acts = sae_out[b, :seq_len, feature_index].cpu().numpy()
                activations.append(float(np.max(acts)))
        
        del cache, hidden, sae_out
        torch.cuda.empty_cache()
    
    return np.array(activations)


def run_injection_experiment(
    model,
    sae,
    tokenizer,
    feature_index: int,
    top_tokens: list,
    nonreasoning_texts: list,
    reasoning_texts: list,
    layer: int,
    device: str,
    n_inject: int = 3,
    strategies: list = ["prepend", "intersperse", "replace"],
) -> dict:
    """Run token injection experiment for a single feature."""
    
    results = {
        "feature_index": feature_index,
        "n_tokens_available": len(top_tokens),
        "n_inject": n_inject,
    }
    
    # Baseline: activation on original non-reasoning text
    baseline_acts = get_feature_activation(
        model, sae, tokenizer, nonreasoning_texts, layer, feature_index, device
    )
    results["baseline_mean"] = float(np.mean(baseline_acts))
    results["baseline_std"] = float(np.std(baseline_acts))
    results["baseline_nonzero_frac"] = float(np.mean(baseline_acts > 0.1))
    
    # Target: activation on reasoning text
    reasoning_acts = get_feature_activation(
        model, sae, tokenizer, reasoning_texts, layer, feature_index, device
    )
    results["reasoning_mean"] = float(np.mean(reasoning_acts))
    results["reasoning_std"] = float(np.std(reasoning_acts))
    results["reasoning_nonzero_frac"] = float(np.mean(reasoning_acts > 0.1))
    
    # Test each injection strategy
    strategy_results = {}
    
    for strategy in strategies:
        # Inject tokens
        injected_texts = [
            inject_tokens_into_text(text, top_tokens, n_inject, strategy)
            for text in nonreasoning_texts
        ]
        
        # Measure activation after injection
        injected_acts = get_feature_activation(
            model, sae, tokenizer, injected_texts, layer, feature_index, device
        )
        
        # Compute metrics
        activation_increase = np.mean(injected_acts) - np.mean(baseline_acts)
        
        # Transfer ratio: how much of reasoning-level activation does injection achieve?
        reasoning_gap = results["reasoning_mean"] - results["baseline_mean"]
        if reasoning_gap > 0.1:
            transfer_ratio = activation_increase / reasoning_gap
        else:
            transfer_ratio = 0.0 if activation_increase < 0.1 else 1.0
        
        # Statistical test: is injection activation significantly higher than baseline?
        t_stat, p_value = stats.ttest_ind(injected_acts, baseline_acts)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(injected_acts) + np.var(baseline_acts)) / 2)
        cohens_d = activation_increase / pooled_std if pooled_std > 0 else 0
        
        strategy_results[strategy] = {
            "injected_mean": float(np.mean(injected_acts)),
            "injected_std": float(np.std(injected_acts)),
            "injected_nonzero_frac": float(np.mean(injected_acts > 0.1)),
            "activation_increase": float(activation_increase),
            "transfer_ratio": float(transfer_ratio),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": bool(p_value < 0.01 and cohens_d > 0.3),
        }
    
    results["strategies"] = strategy_results
    
    # Overall assessment
    best_strategy = max(strategy_results.keys(), 
                       key=lambda s: strategy_results[s]["transfer_ratio"])
    best_transfer = strategy_results[best_strategy]["transfer_ratio"]
    best_significant = strategy_results[best_strategy]["significant"]
    
    # Classification
    if best_transfer > 0.5 and best_significant:
        classification = "token_driven"
        interpretation = "Feature activates when tokens are injected (shallow)"
    elif best_transfer > 0.2 and best_significant:
        classification = "partially_token_driven"
        interpretation = "Feature partially responds to token injection"
    elif best_significant:
        classification = "weakly_token_driven"
        interpretation = "Tokens have small but significant effect"
    else:
        classification = "context_dependent"
        interpretation = "Feature does NOT activate with token injection alone"
    
    results["classification"] = classification
    results["interpretation"] = interpretation
    results["best_strategy"] = best_strategy
    results["best_transfer_ratio"] = best_transfer
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Token Injection Experiment")
    parser.add_argument("--token-analysis", type=str, required=True,
                        help="Path to token_analysis.json")
    parser.add_argument("--reasoning-features", type=str, required=True,
                        help="Path to reasoning_features.json")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--top-k-features", type=int, default=10)
    parser.add_argument("--top-k-tokens", type=int, default=30)
    parser.add_argument("--n-inject", type=int, default=3,
                        help="Number of tokens to inject per text")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples per condition")
    parser.add_argument("--model-name", type=str, default="google/gemma-2-9b")
    parser.add_argument("--sae-name", type=str, default="gemma-scope-9b-pt-res-canonical")
    parser.add_argument("--sae-id-format", type=str, default="layer_{layer}/width_16k/canonical")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("TOKEN INJECTION EXPERIMENT")
    print(f"{'='*60}\n")
    
    # Load feature indices
    with open(args.reasoning_features) as f:
        feat_data = json.load(f)
    feature_indices = feat_data["feature_indices"][:args.top_k_features]
    print(f"Testing features: {feature_indices}")
    
    # Load model and SAE
    print("\nLoading model and SAE...")
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
    
    # Load datasets
    print("\nLoading datasets...")
    from datasets import load_dataset
    
    # Reasoning text
    s1k = load_dataset("simplescaling/s1K-1.1", split="train")
    reasoning_texts = []
    for row in s1k:
        for key in ["deepseek_thinking_trajectory", "gemini_thinking_trajectory"]:
            if row.get(key):
                reasoning_texts.append(row[key][:512])
                if len(reasoning_texts) >= args.n_samples:
                    break
        if len(reasoning_texts) >= args.n_samples:
            break
    
    # Non-reasoning text
    pile = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    nonreasoning_texts = []
    for row in pile:
        text = row.get("text", "")
        if text and len(text) > 50:
            nonreasoning_texts.append(text[:512])
            if len(nonreasoning_texts) >= args.n_samples:
                break
    
    print(f"Loaded {len(reasoning_texts)} reasoning texts")
    print(f"Loaded {len(nonreasoning_texts)} non-reasoning texts")
    
    # Run experiment for each feature
    print(f"\n{'='*60}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for feat_idx in tqdm(feature_indices, desc="Features"):
        # Load top tokens for this feature
        top_tokens = load_top_tokens_for_feature(
            args.token_analysis, feat_idx, args.top_k_tokens
        )
        
        if len(top_tokens) < 3:
            print(f"  Feature {feat_idx}: Not enough tokens, skipping")
            continue
        
        print(f"\n  Feature {feat_idx}: Testing with {len(top_tokens)} tokens")
        print(f"    Top tokens: {top_tokens[:5]}...")
        
        result = run_injection_experiment(
            model, sae, tokenizer,
            feat_idx, top_tokens,
            nonreasoning_texts, reasoning_texts,
            args.layer, args.device,
            n_inject=args.n_inject,
        )
        
        all_results.append(result)
        
        # Print summary
        print(f"    Baseline activation: {result['baseline_mean']:.3f}")
        print(f"    Reasoning activation: {result['reasoning_mean']:.3f}")
        print(f"    Best transfer ratio: {result['best_transfer_ratio']:.3f}")
        print(f"    Classification: {result['classification']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    classifications = [r["classification"] for r in all_results]
    for cls in ["token_driven", "partially_token_driven", "weakly_token_driven", "context_dependent"]:
        count = classifications.count(cls)
        pct = 100 * count / len(classifications) if classifications else 0
        print(f"  {cls}: {count} ({pct:.1f}%)")
    
    avg_transfer = np.mean([r["best_transfer_ratio"] for r in all_results])
    print(f"\n  Average best transfer ratio: {avg_transfer:.3f}")
    
    # Save results
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    output = {
        "config": {
            "layer": args.layer,
            "top_k_features": args.top_k_features,
            "top_k_tokens": args.top_k_tokens,
            "n_inject": args.n_inject,
            "n_samples": args.n_samples,
        },
        "summary": {
            "n_features": len(all_results),
            "classification_counts": {
                cls: classifications.count(cls) 
                for cls in ["token_driven", "partially_token_driven", 
                            "weakly_token_driven", "context_dependent"]
            },
            "avg_transfer_ratio": float(avg_transfer),
            "avg_baseline_activation": float(np.mean([r["baseline_mean"] for r in all_results])),
            "avg_reasoning_activation": float(np.mean([r["reasoning_mean"] for r in all_results])),
        },
        "features": all_results,
    }
    
    output_path = save_path / "injection_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
