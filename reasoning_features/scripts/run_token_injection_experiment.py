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

- **Cohen's d (Effect Size)**: Standardized mean difference between injected and baseline activations
- **Statistical Significance**: t-test p-value with appropriate alpha levels

## Classification (Based on Cohen's d, 1988)

The classification uses well-established effect size conventions from Cohen (1988),
providing statistically principled and reviewable thresholds:

- **token_driven** (Large effect, d >= 0.8, p < 0.01):
  79% of injected samples exceed baseline median.
  Feature strongly activates on tokens alone.

- **partially_token_driven** (Medium effect, 0.5 <= d < 0.8, p < 0.01):
  69% of injected samples exceed baseline median.
  Feature moderately activates on tokens.

- **weakly_token_driven** (Small effect, 0.2 <= d < 0.5, p < 0.05):
  58% of injected samples exceed baseline median.
  Feature weakly activates on tokens.

- **context_dependent** (Negligible effect, d < 0.2 or p >= 0.05):
  Token injection does not meaningfully increase activation.
  Feature may capture deeper reasoning patterns or context.

## Interpretation

If feature is TOKEN-DRIVEN (large effect size):
- Injection significantly increases activation
- Feature likely detects surface patterns, not reasoning

If feature captures REASONING STRUCTURE (small/negligible effect):
- Injection does not significantly increase activation
- Feature may detect genuine reasoning patterns

## Usage

```bash
python run_token_injection_experiment.py \\
    --token-analysis results/layer8/token_analysis.json \\
    --reasoning-features results/layer8/reasoning_features.json \\
    --layer 8 \\
    --top-k-features 10 \\
    --save-dir results/layer8/injection

# Custom effect size thresholds (defaults are Cohen's conventions)
python run_token_injection_experiment.py \\
    --token-analysis results/layer8/token_analysis.json \\
    --reasoning-features results/layer8/reasoning_features.json \\
    --layer 8 \\
    --d-large 0.8 --d-medium 0.5 --d-small 0.2 \\
    --save-dir results/layer8/injection
```
"""

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Optional, Literal

import numpy as np
import torch
from einops import reduce
from jaxtyping import Float, Int
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_token_contexts(
    reasoning_texts: list[str],
    target_tokens: list[str],
    context_window: int = 2,
) -> dict[str, dict[str, list[str]]]:
    """Extract common contexts (preceding/following tokens) for target tokens.
    
    Args:
        reasoning_texts: List of reasoning text samples
        target_tokens: List of tokens to find contexts for
        context_window: Number of tokens before/after to consider
        
    Returns:
        Dict mapping token -> {"before": [common_preceding_tokens], "after": [common_following_tokens]}
    """
    from collections import Counter
    
    token_contexts = {token: {"before": Counter(), "after": Counter()} for token in target_tokens}
    
    for text in reasoning_texts:
        words = text.split()
        for i, word in enumerate(words):
            # Normalize word (strip punctuation for matching)
            normalized = word.strip('.,!?;:').lower()
            
            for target in target_tokens:
                if target.strip().lower() in normalized:
                    # Extract preceding tokens
                    if i > 0:
                        prev_word = words[i-1].strip('.,!?;:').lower()
                        token_contexts[target]["before"][prev_word] += 1
                    
                    # Extract following tokens
                    if i < len(words) - 1:
                        next_word = words[i+1].strip('.,!?;:').lower()
                        token_contexts[target]["after"][next_word] += 1
    
    # Convert Counters to sorted lists (most common first)
    result = {}
    for token in target_tokens:
        result[token] = {
            "before": [tok for tok, _ in token_contexts[token]["before"].most_common(10)],
            "after": [tok for tok, _ in token_contexts[token]["after"].most_common(10)],
        }
    
    return result


def extract_active_trigram_sequences(
    model,
    sae,
    tokenizer,
    reasoning_texts: list[str],
    feature_index: int,
    layer: int,
    device: str,
    activation_threshold: float = 0.1,
    max_sequences: int = 50,
    max_length: int = 128,
) -> list[list[str]]:
    """Extract consecutive 3-token sequences where all tokens activate the feature.
    
    This finds natural trigrams from reasoning texts where all three consecutive
    tokens have feature activation above the threshold (as percentage of max activation).
    
    Args:
        model: The transformer model
        sae: The sparse autoencoder
        tokenizer: The tokenizer
        reasoning_texts: List of reasoning text samples
        feature_index: Index of the feature to check
        layer: Layer index
        device: Device to run on
        activation_threshold: Threshold as percentage of max activation (0.0 to 1.0)
        max_sequences: Maximum number of sequences to return
        max_length: Maximum sequence length for tokenization
        
    Returns:
        List of [token1, token2, token3] sequences as strings
    """
    sequences = []
    
    # Process texts in small batches to find active trigrams
    for text in reasoning_texts:
        if len(sequences) >= max_sequences:
            break
            
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)
        
        token_ids = inputs["input_ids"][0].tolist()
        if len(token_ids) < 4:  # Need at least 3 tokens + BOS
            continue
        
        # Get activations
        with torch.no_grad():
            _, cache = model.run_with_cache(
                inputs["input_ids"],
                names_filter=[f"blocks.{layer}.hook_resid_post"],
            )
            hidden = cache[f"blocks.{layer}.hook_resid_post"]
            sae_acts = sae.encode(hidden)
            feature_acts = sae_acts[0, :, feature_index].cpu().numpy()
        
        # Compute threshold based on max activation in this text
        max_act = feature_acts.max()
        if max_act <= 0:
            continue
        threshold = max_act * activation_threshold
        
        # Find consecutive trigrams where all tokens are active
        for i in range(1, len(token_ids) - 2):  # Skip BOS, leave room for trigram
            if (feature_acts[i] >= threshold and 
                feature_acts[i+1] >= threshold and 
                feature_acts[i+2] >= threshold):
                # Decode each token
                t1 = tokenizer.decode([token_ids[i]])
                t2 = tokenizer.decode([token_ids[i+1]])
                t3 = tokenizer.decode([token_ids[i+2]])
                
                # Skip if any token is empty or just whitespace
                if t1.strip() and t2.strip() and t3.strip():
                    sequences.append([t1, t2, t3])
                    
                    if len(sequences) >= max_sequences:
                        break
    
    return sequences[:max_sequences]


def load_top_tokens_for_feature(
    token_analysis_path: str,
    feature_index: int,
    top_k: int = 30,
) -> list[str]:
    """Load top-k tokens (as strings) for a specific feature.
    
    Args:
        token_analysis_path: Path to token_analysis.json
        feature_index: Index of the feature
        top_k: Number of top tokens to return
        
    Returns:
        List of token strings
    """
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    for feature in data.get("features", []):
        if feature.get("feature_index") == feature_index:
            tokens = feature.get("top_tokens", [])[:top_k]
            return [t["token_str"] for t in tokens]
    
    return []


def load_top_ngrams_for_feature(
    token_analysis_path: str,
    feature_index: int,
    n: int = 2,
    top_k: int = 10,
) -> list[str]:
    """Load top-k n-grams (as concatenated strings) for a specific feature.
    
    Args:
        token_analysis_path: Path to token_analysis.json
        feature_index: Index of the feature
        n: N-gram size (2 for bigram, 3 for trigram)
        top_k: Number of top n-grams to return
        
    Returns:
        List of n-gram strings (concatenated tokens)
    """
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    key = "top_bigrams" if n == 2 else "top_trigrams"
    
    for feature in data.get("features", []):
        if feature.get("feature_index") == feature_index:
            ngrams = feature.get(key, [])[:top_k]
            return [ng["ngram_str"] for ng in ngrams]
    
    return []


def inject_tokens_into_text(
    text: str,
    tokens: list[str],
    n_inject: int = 3,
    strategy: Literal[
        "prepend", "append", "intersperse", "replace",
        "inject_bigram", "inject_trigram",
        "bigram_before", "bigram_after", "trigram",
        "comma_list", "active_trigram"
    ] = "prepend",
    token_contexts: Optional[dict[str, dict[str, list[str]]]] = None,
    ngrams: Optional[list[str]] = None,
) -> str:
    """Inject tokens into text using various strategies.
    
    Args:
        text: The original text
        tokens: List of tokens to inject
        n_inject: Number of tokens to inject
        strategy: Injection strategy (see below)
        token_contexts: Optional dict mapping token -> {"before": [...], "after": [...]}
                       Required for contextual strategies
        ngrams: Optional list of n-gram strings for inject_bigram/inject_trigram strategies
        
    Simple strategies (use n_inject):
        - prepend: Add tokens at the beginning
        - append: Add tokens at the end
        - intersperse: Spread tokens throughout
        - replace: Replace random words
        
    N-gram injection strategies (use n_inject_bigram/n_inject_trigram):
        - inject_bigram: Inject top bigrams from analysis
        - inject_trigram: Inject top trigrams from analysis
        
    Contextual strategies:
        - bigram_before: Inject [context_word, target_token] pairs
        - bigram_after: Inject [target_token, context_word] pairs
        - trigram: Inject [before, target_token, after] triplets
        - comma_list: Inject as comma-separated list
        
    Returns:
        Modified text with tokens injected
    """
    selected_tokens = random.sample(tokens, min(n_inject, len(tokens)))
    words = text.split()
    
    # Simple strategies
    if strategy == "prepend":
        injection = " ".join(selected_tokens) + " "
        return injection + text
    
    elif strategy == "append":
        injection = " " + " ".join(selected_tokens)
        return text + injection
    
    elif strategy == "intersperse":
        if len(words) < 2:
            return " ".join(selected_tokens) + " " + text
        for token in selected_tokens:
            pos = random.randint(0, len(words))
            words.insert(pos, token)
        return " ".join(words)
    
    elif strategy == "replace":
        if len(words) < len(selected_tokens):
            return " ".join(selected_tokens)
        positions = random.sample(range(len(words)), len(selected_tokens))
        for pos, token in zip(positions, selected_tokens):
            words[pos] = token
        return " ".join(words)
    
    # N-gram injection strategies (use pre-computed ngrams from analysis)
    elif strategy == "inject_bigram":
        if not ngrams:
            # Fallback to simple prepend
            return " ".join(selected_tokens) + " " + text
        selected_ngrams = random.sample(ngrams, min(n_inject, len(ngrams)))
        if len(words) < 2:
            return " ".join(selected_ngrams) + " " + text
        for ngram in selected_ngrams:
            pos = random.randint(0, len(words))
            words.insert(pos, ngram)
        return " ".join(words)
    
    elif strategy == "inject_trigram":
        if not ngrams:
            return " ".join(selected_tokens) + " " + text
        selected_ngrams = random.sample(ngrams, min(n_inject, len(ngrams)))
        if len(words) < 2:
            return " ".join(selected_ngrams) + " " + text
        for ngram in selected_ngrams:
            pos = random.randint(0, len(words))
            words.insert(pos, ngram)
        return " ".join(words)
    
    # Contextual strategies (require token_contexts)
    elif strategy == "bigram_before":
        if not token_contexts:
            # Fallback to prepend if no context available
            return " ".join(selected_tokens) + " " + text
        
        # Inject [context_word, target_token] bigrams
        bigrams = []
        for token in selected_tokens:
            context_words = token_contexts.get(token, {}).get("before", [])
            if context_words:
                context = random.choice(context_words[:3])  # Pick from top 3
                bigrams.append(f"{context} {token}")
            else:
                bigrams.append(token)
        
        if len(words) < 2:
            return " ".join(bigrams) + " " + text
        # Intersperse bigrams
        for bigram in bigrams:
            pos = random.randint(0, len(words))
            words.insert(pos, bigram)
        return " ".join(words)
    
    elif strategy == "bigram_after":
        if not token_contexts:
            return text + " " + " ".join(selected_tokens)
        
        # Inject [target_token, context_word] bigrams
        bigrams = []
        for token in selected_tokens:
            context_words = token_contexts.get(token, {}).get("after", [])
            if context_words:
                context = random.choice(context_words[:3])
                bigrams.append(f"{token} {context}")
            else:
                bigrams.append(token)
        
        if len(words) < 2:
            return " ".join(bigrams) + " " + text
        for bigram in bigrams:
            pos = random.randint(0, len(words))
            words.insert(pos, bigram)
        return " ".join(words)
    
    elif strategy == "trigram":
        if not token_contexts:
            return " ".join(selected_tokens) + " " + text
        
        # Inject [before, target_token, after] trigrams
        trigrams = []
        for token in selected_tokens:
            contexts = token_contexts.get(token, {})
            before_words = contexts.get("before", [])
            after_words = contexts.get("after", [])
            
            if before_words and after_words:
                before = random.choice(before_words[:3])
                after = random.choice(after_words[:3])
                trigrams.append(f"{before} {token} {after}")
            elif before_words:
                before = random.choice(before_words[:3])
                trigrams.append(f"{before} {token}")
            elif after_words:
                after = random.choice(after_words[:3])
                trigrams.append(f"{token} {after}")
            else:
                trigrams.append(token)
        
        if len(words) < 2:
            return " ".join(trigrams) + " " + text
        for trigram in trigrams:
            pos = random.randint(0, len(words))
            words.insert(pos, trigram)
        return " ".join(words)
    
    elif strategy == "comma_list":
        # Inject as a comma-separated list
        list_str = ", ".join(selected_tokens)
        if len(words) < 2:
            return list_str + " " + text
        # Insert list at random position
        pos = random.randint(0, len(words))
        words.insert(pos, list_str)
        return " ".join(words)
    
    elif strategy == "active_trigram":
        # Inject consecutive token sequences from reasoning texts that all activate the feature
        # token_contexts should contain "active_sequences" key with list of [tok1, tok2, tok3] sequences
        if not token_contexts or "active_sequences" not in token_contexts:
            # Fallback: just use three consecutive selected tokens
            if len(selected_tokens) >= 3:
                trigram_str = " ".join(selected_tokens[:3])
            else:
                trigram_str = " ".join(selected_tokens)
            if len(words) < 2:
                return trigram_str + " " + text
            pos = random.randint(0, len(words))
            words.insert(pos, trigram_str)
            return " ".join(words)
        
        # Use pre-extracted active sequences
        sequences = token_contexts["active_sequences"]
        if not sequences:
            # Fallback
            trigram_str = " ".join(selected_tokens[:min(3, len(selected_tokens))])
            if len(words) < 2:
                return trigram_str + " " + text
            pos = random.randint(0, len(words))
            words.insert(pos, trigram_str)
            return " ".join(words)
        
        # Select n_inject sequences
        selected_seqs = random.sample(sequences, min(n_inject, len(sequences)))
        
        for seq in selected_seqs:
            seq_str = " ".join(seq)
            if len(words) < 2:
                words = [seq_str] + words
            else:
                pos = random.randint(0, len(words))
                words.insert(pos, seq_str)
        return " ".join(words)
    
    return text


def get_feature_activation(
    model,
    sae,
    tokenizer,
    texts: list[str],
    layer: int,
    feature_index: int,
    device: str,
    batch_size: int = 16,
    max_length: int = 128,
) -> Float[np.ndarray, "n_texts"]:
    """Get max feature activations for a batch of texts.
    
    Args:
        model: The transformer model
        sae: The sparse autoencoder
        tokenizer: The tokenizer
        texts: List of text strings to process
        layer: Layer index for activation extraction
        feature_index: Index of the feature to extract
        device: Device to run on
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        Array of shape (n_texts,) with max activation per text
    """
    activations: list[float] = []
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids: Int[torch.Tensor, "batch seq"] = tokens["input_ids"].to(device)
        attention_mask: Int[torch.Tensor, "batch seq"] = tokens["attention_mask"].to(device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, stop_at_layer=layer + 1)
            hidden: Float[torch.Tensor, "batch seq d_model"] = cache[hook_name]
            sae_out: Float[torch.Tensor, "batch seq n_features"] = sae.encode(hidden)
            
            # Get max activation per text for the target feature
            for b in range(sae_out.shape[0]):
                seq_len = int(attention_mask[b].sum().item())
                acts: Float[np.ndarray, "seq"] = sae_out[b, :seq_len, feature_index].cpu().numpy()
                activations.append(float(np.max(acts)))
        
        del cache, hidden, sae_out
        torch.cuda.empty_cache()
    
    return np.array(activations)


def run_injection_experiment(
    model,
    sae,
    tokenizer,
    feature_index: int,
    top_tokens: list[str],
    nonreasoning_texts: list[str],
    reasoning_texts: list[str],
    layer: int,
    device: str,
    n_inject: int = 3,
    n_inject_bigram: int = 2,
    n_inject_trigram: int = 1,
    strategies: Optional[list[str]] = None,
    token_contexts: Optional[dict[str, dict[str, list[str]]]] = None,
    top_bigrams: Optional[list[str]] = None,
    top_trigrams: Optional[list[str]] = None,
    batch_size: int = 16,
    max_length: int = 128,
    # Effect size thresholds for classification (Cohen's d conventions)
    d_large: float = 0.8,       # Large effect threshold
    d_medium: float = 0.5,      # Medium effect threshold
    d_small: float = 0.2,       # Small effect threshold
    alpha: float = 0.01,        # Significance level for large/medium effects
    alpha_weak: float = 0.05,   # Significance level for weak effects
) -> dict:
    """Run token injection experiment for a single feature.
    
    This experiment tests whether a feature is token-driven by injecting
    the feature's top tokens into non-reasoning text and measuring the
    resulting activation increase.
    
    Args:
        model: The transformer model
        sae: The sparse autoencoder
        tokenizer: The tokenizer
        feature_index: Index of the feature to test
        top_tokens: List of top token strings for this feature
        nonreasoning_texts: Non-reasoning text samples (baseline)
        reasoning_texts: Reasoning text samples (target)
        layer: Layer index
        device: Device to run on
        n_inject: Number of tokens to inject for simple strategies
        n_inject_bigram: Number of bigram sequences to inject
        n_inject_trigram: Number of trigram sequences to inject
        strategies: Injection strategies to test
        token_contexts: Optional context information for contextual strategies
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        Dictionary containing experiment results and classification
    """
    if strategies is None:
        strategies = ["prepend", "intersperse", "replace"]
    
    results = {
        "feature_index": feature_index,
        "n_tokens_available": len(top_tokens),
        "n_inject": n_inject,
    }
    
    # Baseline: activation on original non-reasoning text
    baseline_acts = get_feature_activation(
        model, sae, tokenizer, nonreasoning_texts, layer, feature_index, device,
        batch_size=batch_size, max_length=max_length
    )
    baseline_mean = float(reduce(baseline_acts, 'samples -> ', 'mean'))
    baseline_std = float(np.std(baseline_acts))
    # Use relative threshold for nonzero: activation > 0.5 std above mean
    nonzero_threshold = max(baseline_std * 0.5, 0.01)
    
    results["baseline_mean"] = baseline_mean
    results["baseline_std"] = baseline_std
    results["baseline_nonzero_frac"] = float(reduce((baseline_acts > nonzero_threshold).astype(float), 'samples -> ', 'mean'))
    
    # Target: activation on reasoning text
    reasoning_acts = get_feature_activation(
        model, sae, tokenizer, reasoning_texts, layer, feature_index, device,
        batch_size=batch_size, max_length=max_length
    )
    results["reasoning_mean"] = float(reduce(reasoning_acts, 'samples -> ', 'mean'))
    results["reasoning_std"] = float(np.std(reasoning_acts))
    results["reasoning_nonzero_frac"] = float(reduce((reasoning_acts > nonzero_threshold).astype(float), 'samples -> ', 'mean'))
    
    # Test each injection strategy
    strategy_results = {}
    
    # Define strategy categories
    simple_strategies = {"prepend", "append", "intersperse", "replace"}
    ngram_bigram_strategies = {"inject_bigram", "bigram_before", "bigram_after"}
    ngram_trigram_strategies = {"inject_trigram", "trigram", "active_trigram"}
    # comma_list uses n_inject since a list needs multiple items
    
    def get_n_inject_for_strategy(strat: str) -> int:
        if strat in simple_strategies or strat == "comma_list":
            return n_inject
        elif strat in ngram_bigram_strategies:
            return n_inject_bigram
        else:  # trigram strategies
            return n_inject_trigram
    
    for strategy in strategies:
        # Use appropriate n_inject based on strategy type
        n_to_inject = get_n_inject_for_strategy(strategy)
        
        # Select appropriate ngrams for this strategy
        ngrams = None
        if strategy == "inject_bigram":
            ngrams = top_bigrams
        elif strategy == "inject_trigram":
            ngrams = top_trigrams
        
        # Inject tokens
        injected_texts = [
            inject_tokens_into_text(text, top_tokens, n_to_inject, strategy, token_contexts, ngrams)
            for text in nonreasoning_texts
        ]
        
        # Measure activation after injection
        injected_acts = get_feature_activation(
            model, sae, tokenizer, injected_texts, layer, feature_index, device,
            batch_size=batch_size, max_length=max_length
        )
        
        # Compute metrics using einops
        injected_mean = float(reduce(injected_acts, 'samples -> ', 'mean'))
        baseline_mean_local = float(reduce(baseline_acts, 'samples -> ', 'mean'))
        activation_increase = injected_mean - baseline_mean_local
        baseline_std = results["baseline_std"]
        
        # Statistical test: is injection activation significantly higher than baseline?
        t_stat, p_value = stats.ttest_ind(injected_acts, baseline_acts)
        
        # Effect size (Cohen's d) - primary metric for classification
        # Using pooled standard deviation (assumes equal variances)
        pooled_std = np.sqrt((np.var(injected_acts) + np.var(baseline_acts)) / 2)
        cohens_d = activation_increase / pooled_std if pooled_std > 0 else 0
        
        # Determine significance based on effect size thresholds
        is_large_effect = cohens_d >= d_large and p_value < alpha
        is_medium_effect = cohens_d >= d_medium and p_value < alpha
        is_small_effect = cohens_d >= d_small and p_value < alpha_weak
        
        # Nonzero fraction using relative threshold (1 std above 0)
        nonzero_threshold = max(baseline_std * 0.5, 0.01)
        
        strategy_results[strategy] = {
            "injected_mean": injected_mean,
            "injected_std": float(np.std(injected_acts)),
            "injected_nonzero_frac": float(reduce((injected_acts > nonzero_threshold).astype(float), 'samples -> ', 'mean')),
            "activation_increase": float(activation_increase),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": bool(is_small_effect),
            "is_large_effect": bool(is_large_effect),
            "is_medium_effect": bool(is_medium_effect),
            "is_small_effect": bool(is_small_effect),
        }
    
    results["strategies"] = strategy_results
    
    # Overall assessment - find best strategy by Cohen's d (effect size)
    best_strategy = max(strategy_results.keys(), 
                        key=lambda s: strategy_results[s]["cohens_d"])
    
    best_d = strategy_results[best_strategy]["cohens_d"]
    best_p = strategy_results[best_strategy]["p_value"]
    
    # Classification based on Cohen's d effect size conventions (Cohen, 1988)
    # This provides statistically principled thresholds:
    # - Large effect (d ≥ 0.8): 79% of treatment exceeds control median
    # - Medium effect (d ≥ 0.5): 69% of treatment exceeds control median
    # - Small effect (d ≥ 0.2): 58% of treatment exceeds control median
    
    if best_d >= d_large and best_p < alpha:
        classification = "token_driven"
        interpretation = f"Large effect size (d={best_d:.2f}): tokens strongly activate feature"
    elif best_d >= d_medium and best_p < alpha:
        classification = "partially_token_driven"
        interpretation = f"Medium effect size (d={best_d:.2f}): tokens moderately activate feature"
    elif best_d >= d_small and best_p < alpha_weak:
        classification = "weakly_token_driven"
        interpretation = f"Small effect size (d={best_d:.2f}): tokens weakly activate feature"
    else:
        classification = "context_dependent"
        interpretation = f"Negligible effect (d={best_d:.2f}): tokens alone do not activate feature"
    
    results["classification"] = classification
    results["interpretation"] = interpretation
    results["best_strategy"] = best_strategy
    results["best_cohens_d"] = best_d
    results["best_p_value"] = best_p
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Token Injection Experiment")
    parser.add_argument("--token-analysis", type=str, required=True,
                        help="Path to token_analysis.json")
    parser.add_argument("--reasoning-features", type=str, required=True,
                        help="Path to reasoning_features.json")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--top-k-features", type=int, default=10)
    parser.add_argument("--top-k-tokens", type=int, default=10)
    parser.add_argument("--n-inject", type=int, default=3,
                        help="Number of tokens to inject for simple strategies "
                             "(prepend, append, intersperse, replace, comma_list)")
    parser.add_argument("--n-inject-bigram", type=int, default=2,
                        help="Number of bigram sequences to inject for bigram strategies "
                             "(bigram_before, bigram_after)")
    parser.add_argument("--n-inject-trigram", type=int, default=1,
                        help="Number of trigram sequences to inject for trigram strategies "
                             "(trigram, active_trigram)")
    parser.add_argument("--active-trigram-threshold", type=float, default=0.1,
                        help="Threshold for 'active' tokens as percentage of max activation (0.0 to 1.0). "
                             "Tokens with activation >= max_activation * threshold are considered active.")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["prepend", "intersperse", "replace", 
                                 "inject_bigram", "inject_trigram",
                                 "bigram_before", "trigram", "comma_list"],
                        help="Injection strategies to test. Options: prepend, append, intersperse, "
                             "replace, inject_bigram, inject_trigram, bigram_before, bigram_after, "
                             "trigram, comma_list, active_trigram")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples per condition")
    parser.add_argument("--reasoning-dataset", type=str, default="s1k",
                        choices=["s1k", "general_inquiry_cot", "combined"],
                        help="Reasoning dataset to use")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--sae-name", type=str, default="gemma-scope-2-4b-it-res-all")
    parser.add_argument("--sae-id-format", type=str, default="layer_{layer}_width_16k_l0_small")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for processing texts")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum sequence length for tokenization")
    
    # Effect size thresholds (Cohen's d conventions)
    parser.add_argument("--d-large", type=float, default=0.8,
                        help="Cohen's d threshold for large effect / token_driven (default: 0.8)")
    parser.add_argument("--d-medium", type=float, default=0.5,
                        help="Cohen's d threshold for medium effect / partially_token_driven (default: 0.5)")
    parser.add_argument("--d-small", type=float, default=0.2,
                        help="Cohen's d threshold for small effect / weakly_token_driven (default: 0.2)")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Significance level for large/medium effects (default: 0.01)")
    parser.add_argument("--alpha-weak", type=float, default=0.05,
                        help="Significance level for weak effects (default: 0.05)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("TOKEN INJECTION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Classification thresholds (Cohen's d, 1988 conventions):")
    print(f"  token_driven:           d >= {args.d_large}, p < {args.alpha}")
    print(f"  partially_token_driven: d >= {args.d_medium}, p < {args.alpha}")
    print(f"  weakly_token_driven:    d >= {args.d_small}, p < {args.alpha_weak}")
    print(f"  context_dependent:      d <  {args.d_small} or p >= {args.alpha_weak}")
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
    if isinstance(sae, tuple):
        sae = sae[0]
    tokenizer = model.tokenizer
    
    # Load datasets
    print(f"\nLoading datasets (reasoning: {args.reasoning_dataset})...")
    from datasets import load_dataset
    
    # Reasoning text based on dataset choice
    reasoning_texts: list[str] = []
    
    if args.reasoning_dataset in ["s1k", "combined"]:
        s1k = load_dataset("simplescaling/s1K-1.1", split="train")
        for row in s1k:
            for key in ["deepseek_thinking_trajectory", "gemini_thinking_trajectory"]:
                if row.get(key):
                    reasoning_texts.append(row[key][:512])
                    if len(reasoning_texts) >= args.n_samples:
                        break
            if len(reasoning_texts) >= args.n_samples:
                break
    
    if args.reasoning_dataset in ["general_inquiry_cot", "combined"]:
        if len(reasoning_texts) < args.n_samples:
            gicot = load_dataset("moremilk/General_Inquiry_Thinking-Chain-Of-Thought", split="train")
            for row in gicot:
                metadata = row.get("metadata", {})
                if isinstance(metadata, dict):
                    text = metadata.get("reasoning", "")
                    if text:
                        # Remove <think> tags
                        text = text.replace("<think>", "").replace("</think>", "").strip()
                        reasoning_texts.append(text[:512])
                        if len(reasoning_texts) >= args.n_samples:
                            break
    
    # Non-reasoning text
    pile = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    nonreasoning_texts: list[str] = []
    for row in pile:
        text = row.get("text", "")
        if text and len(text) > 50:
            nonreasoning_texts.append(text[:512])
            if len(nonreasoning_texts) >= args.n_samples:
                break
    
    print(f"Loaded {len(reasoning_texts)} reasoning texts")
    print(f"Loaded {len(nonreasoning_texts)} non-reasoning texts")
    
    # Check if contextual strategies or ngram injection are requested
    contextual_strategies = {"bigram_before", "bigram_after", "trigram", "active_trigram"}
    ngram_injection_strategies = {"inject_bigram", "inject_trigram"}
    use_contexts = bool(set(args.strategies) & contextual_strategies)
    use_ngram_injection = bool(set(args.strategies) & ngram_injection_strategies)
    
    # Extract token contexts if needed
    print(f"\nStrategies to test: {', '.join(args.strategies)}")
    if use_contexts:
        print("Extracting token contexts from reasoning texts...")
        # We'll extract contexts for all features at once
        all_feature_tokens = []
        for feat_idx in feature_indices:
            tokens = load_top_tokens_for_feature(args.token_analysis, feat_idx, args.top_k_tokens)
            all_feature_tokens.extend(tokens)
        all_feature_tokens = list(set(all_feature_tokens))  # Deduplicate
        
        global_token_contexts = extract_token_contexts(
            reasoning_texts[:min(1000, len(reasoning_texts))],  # Use up to 1000 texts for context extraction
            all_feature_tokens,
            context_window=2
        )
        print(f"  Extracted contexts for {len(global_token_contexts)} unique tokens")
    else:
        global_token_contexts = None
    
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
        
        # Load top bigrams and trigrams if needed
        feature_bigrams = None
        feature_trigrams = None
        if use_ngram_injection:
            if "inject_bigram" in args.strategies:
                feature_bigrams = load_top_ngrams_for_feature(
                    args.token_analysis, feat_idx, n=2, top_k=20
                )
            if "inject_trigram" in args.strategies:
                feature_trigrams = load_top_ngrams_for_feature(
                    args.token_analysis, feat_idx, n=3, top_k=10
                )
        
        # Get contexts for this feature's tokens
        if use_contexts and global_token_contexts:
            feature_token_contexts = {
                token: global_token_contexts.get(token, {"before": [], "after": []})
                for token in top_tokens
            }
            # Add active trigram sequences if that strategy is requested
            if "active_trigram" in args.strategies:
                print(f"    Extracting active trigram sequences (threshold={args.active_trigram_threshold})...")
                active_seqs = extract_active_trigram_sequences(
                    model, sae, tokenizer,
                    reasoning_texts[:50],  # Limit texts for efficiency
                    feat_idx, args.layer, args.device,
                    activation_threshold=args.active_trigram_threshold,
                    max_sequences=50,
                    max_length=args.max_length,
                )
                feature_token_contexts["active_sequences"] = active_seqs
                if active_seqs:
                    print(f"    Found {len(active_seqs)} active trigram sequences")
                    print(f"    Examples: {active_seqs[:3]}")
                else:
                    print(f"    No active trigram sequences found, will fallback")
        else:
            feature_token_contexts = None
        
        print(f"\n  Feature {feat_idx}: Testing with {len(top_tokens)} tokens")
        print(f"    Top tokens: {top_tokens[:5]}...")
        if feature_bigrams:
            print(f"    Top bigrams: {feature_bigrams[:3]}...")
        if feature_trigrams:
            print(f"    Top trigrams: {feature_trigrams[:2]}...")
        
        result = run_injection_experiment(
            model, sae, tokenizer,
            feat_idx, top_tokens,
            nonreasoning_texts, reasoning_texts,
            args.layer, args.device,
            n_inject=args.n_inject,
            n_inject_bigram=args.n_inject_bigram,
            n_inject_trigram=args.n_inject_trigram,
            strategies=args.strategies,
            token_contexts=feature_token_contexts,
            top_bigrams=feature_bigrams,
            top_trigrams=feature_trigrams,
            batch_size=args.batch_size,
            max_length=args.max_length,
            # Effect size thresholds
            d_large=args.d_large,
            d_medium=args.d_medium,
            d_small=args.d_small,
            alpha=args.alpha,
            alpha_weak=args.alpha_weak,
        )
        
        all_results.append(result)
        
        # Print summary
        print(f"    Baseline activation: {result['baseline_mean']:.3f}")
        print(f"    Reasoning activation: {result['reasoning_mean']:.3f}")
        print(f"    Best Cohen's d: {result['best_cohens_d']:.3f} (p={result['best_p_value']:.2e})")
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
    
    cohens_d_values = np.array([r["best_cohens_d"] for r in all_results])
    avg_cohens_d = float(reduce(cohens_d_values, 'features -> ', 'mean'))
    print(f"\n  Average best Cohen's d: {avg_cohens_d:.3f}")
    
    # Save results
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    output = {
        "config": {
            "layer": args.layer,
            "reasoning_dataset": args.reasoning_dataset,
            "top_k_features": args.top_k_features,
            "top_k_tokens": args.top_k_tokens,
            "n_inject": args.n_inject,
            "n_inject_bigram": args.n_inject_bigram,
            "n_inject_trigram": args.n_inject_trigram,
            "active_trigram_threshold": args.active_trigram_threshold,
            "strategies": args.strategies,
            "n_samples": args.n_samples,
            # Effect size thresholds (Cohen, 1988)
            "d_large": args.d_large,
            "d_medium": args.d_medium,
            "d_small": args.d_small,
            "alpha": args.alpha,
            "alpha_weak": args.alpha_weak,
        },
        "summary": {
            "n_features": len(all_results),
            "classification_counts": {
                cls: classifications.count(cls) 
                for cls in ["token_driven", "partially_token_driven", 
                            "weakly_token_driven", "context_dependent"]
            },
            "avg_cohens_d": float(avg_cohens_d),
            "avg_baseline_activation": float(reduce(np.array([r["baseline_mean"] for r in all_results]), 'features -> ', 'mean')),
            "avg_reasoning_activation": float(reduce(np.array([r["reasoning_mean"] for r in all_results]), 'features -> ', 'mean')),
        },
        "features": all_results,
    }
    
    output_path = save_path / "injection_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
