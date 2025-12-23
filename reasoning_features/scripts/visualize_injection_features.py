"""
Visualize token-level feature activations on baseline, injected, and reasoning texts.

This script creates interactive HTML visualizations showing how features activate on
individual tokens across different text types (non-reasoning, injected with various
strategies, and reasoning texts).

## Usage

```bash
python visualize_injection_features.py \
    --injection-results results/test/gemma-2-9b/s1k/layer12/injection_results.json \
    --token-analysis results/test/gemma-2-9b/s1k/layer12/token_analysis.json \
    --layer 12 \
    --n-features 5 \
    --n-examples 3 \
    --output-dir visualizations/token_level/layer12
```
"""

import argparse
import json
from pathlib import Path
import sys
import random
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from jaxtyping import Float, Int

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize token-level feature activations from injection experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--injection-results", type=Path, required=True)
    parser.add_argument("--token-analysis", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--n-examples", type=int, default=10,
                        help="Number of example texts per condition")
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help="Maximum sequence length to display")
    parser.add_argument("--model-name", default="google/gemma-2-9b")
    parser.add_argument("--sae-name", default="gemma-scope-9b-pt-res-canonical")
    parser.add_argument("--sae-id-format", default="layer_{layer}/width_16k/canonical")
    parser.add_argument("--reasoning-dataset", default="s1k",
                        choices=["s1k", "general_inquiry_cot"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    
    return parser.parse_args()


def load_top_tokens_for_feature(token_analysis_path: Path, feature_index: int, top_k: int = 10) -> list[str]:
    """Load top-k token strings for a feature."""
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    for feature in data.get("features", []):
        if feature.get("feature_index") == feature_index:
            tokens = feature.get("top_tokens", [])[:top_k]
            return [t["token_str"] for t in tokens]
    return []


def load_top_tokens_with_activations(token_analysis_path: Path, feature_index: int, top_k: int = 20) -> list[dict]:
    """Load top-k tokens with full data (token_str and mean_activation) for a feature."""
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    for feature in data.get("features", []):
        if feature.get("feature_index") == feature_index:
            return feature.get("top_tokens", [])[:top_k]
    return []


def get_token_sequences_from_reasoning_texts(
    reasoning_texts: list[str],
    target_tokens: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Extract bigrams, trigrams, and contextual patterns from reasoning texts.
    
    For each target token, extracts:
    - before: Tokens that appear immediately before the target
    - after: Tokens that appear immediately after the target
    
    Args:
        reasoning_texts: List of reasoning texts to analyze
        target_tokens: Tokens to find contexts for
        
    Returns:
        Dict mapping token -> {"before": [words], "after": [words]}
    """
    from collections import Counter
    
    # Normalize target tokens for matching
    target_set = {t.strip().lower() for t in target_tokens}
    
    # Count contexts
    token_contexts = {token: {"before": Counter(), "after": Counter()} for token in target_tokens}
    
    for text in reasoning_texts:
        words = text.split()
        for i, word in enumerate(words):
            # Normalize word for matching
            normalized = word.strip('.,!?;:').lower()
            
            # Check if any target token matches
            for target in target_tokens:
                if normalized == target.strip().lower():
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


def inject_tokens_into_text(
    text: str,
    tokens: list[str],
    n_inject: int,
    strategy: str,
    seed: Optional[int] = None,
    token_contexts: Optional[dict[str, dict[str, list[str]]]] = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Inject tokens into text using specified strategy.
    
    This function mirrors the logic in run_token_injection_experiment.py exactly,
    but also tracks character positions of injected content.
    
    Args:
        text: Original text
        tokens: List of tokens to inject (already have space prefix from token_analysis)
        n_inject: Number of tokens/phrases to inject
        strategy: Injection strategy
        seed: Random seed for reproducibility
        token_contexts: Optional context information for contextual strategies
    
    Returns:
        tuple: (injected_text, list of (start_char, end_char) tuples for injected regions)
    """
    if seed is not None:
        random.seed(seed)
    
    # Use unique markers to track injected content
    START_MARKER = "⟪INJECT⟫"
    END_MARKER = "⟪/INJECT⟫"
    
    selected_tokens = random.sample(tokens, min(n_inject, len(tokens)))
    words = text.split()
    
    # Simple strategies - wrap injected content with markers
    if strategy == "prepend":
        injection = " ".join(selected_tokens) + " "
        marked_injection = START_MARKER + injection + END_MARKER
        result = marked_injection + text
    
    elif strategy == "intersperse":
        if len(words) < 2:
            injection = " ".join(selected_tokens) + " "
            marked_injection = START_MARKER + injection + END_MARKER
            result = marked_injection + text
        else:
            for token in selected_tokens:
                pos = random.randint(0, len(words))
                # Mark each injected token
                words.insert(pos, START_MARKER + token + END_MARKER)
            result = " ".join(words)
    
    elif strategy == "replace":
        if len(words) < len(selected_tokens):
            result = " ".join([START_MARKER + t + END_MARKER for t in selected_tokens])
        else:
            positions = random.sample(range(len(words)), len(selected_tokens))
            for pos, token in zip(positions, selected_tokens):
                # Mark replaced word
                words[pos] = START_MARKER + token + END_MARKER
            result = " ".join(words)
    
    elif strategy == "append":
        injection = " " + " ".join(selected_tokens)
        marked_injection = START_MARKER + injection + END_MARKER
        result = text + marked_injection
    
    # Contextual strategies
    elif strategy == "bigram_before":
        if not token_contexts:
            injection = " ".join(selected_tokens) + " "
            marked_injection = START_MARKER + injection + END_MARKER
            result = marked_injection + text
        else:
            bigrams = []
            for token in selected_tokens:
                context_words = token_contexts.get(token, {}).get("before", [])
                if context_words:
                    context = random.choice(context_words[:3])
                    # Mark the entire bigram
                    bigrams.append(START_MARKER + f"{context} {token}" + END_MARKER)
                else:
                    bigrams.append(START_MARKER + token + END_MARKER)
            
            if len(words) < 2:
                result = " ".join(bigrams) + " " + text
            else:
                for bigram in bigrams:
                    pos = random.randint(0, len(words))
                    words.insert(pos, bigram)
                result = " ".join(words)
    
    elif strategy == "bigram_after":
        if not token_contexts:
            injection = " " + " ".join(selected_tokens)
            marked_injection = START_MARKER + injection + END_MARKER
            result = text + marked_injection
        else:
            bigrams = []
            for token in selected_tokens:
                context_words = token_contexts.get(token, {}).get("after", [])
                if context_words:
                    context = random.choice(context_words[:3])
                    bigrams.append(START_MARKER + f"{token} {context}" + END_MARKER)
                else:
                    bigrams.append(START_MARKER + token + END_MARKER)
            
            if len(words) < 2:
                result = " ".join(bigrams) + " " + text
            else:
                for bigram in bigrams:
                    pos = random.randint(0, len(words))
                    words.insert(pos, bigram)
                result = " ".join(words)
    
    elif strategy == "trigram":
        if not token_contexts:
            injection = " ".join(selected_tokens) + " "
            marked_injection = START_MARKER + injection + END_MARKER
            result = marked_injection + text
        else:
            trigrams = []
            for token in selected_tokens:
                contexts = token_contexts.get(token, {})
                before_words = contexts.get("before", [])
                after_words = contexts.get("after", [])
                
                if before_words and after_words:
                    before = random.choice(before_words[:3])
                    after = random.choice(after_words[:3])
                    trigrams.append(START_MARKER + f"{before} {token} {after}" + END_MARKER)
                elif before_words:
                    before = random.choice(before_words[:3])
                    trigrams.append(START_MARKER + f"{before} {token}" + END_MARKER)
                elif after_words:
                    after = random.choice(after_words[:3])
                    trigrams.append(START_MARKER + f"{token} {after}" + END_MARKER)
                else:
                    trigrams.append(START_MARKER + token + END_MARKER)
            
            if len(words) < 2:
                result = " ".join(trigrams) + " " + text
            else:
                for trigram in trigrams:
                    pos = random.randint(0, len(words))
                    words.insert(pos, trigram)
                result = " ".join(words)
    
    elif strategy == "comma_list":
        injection = ", ".join(selected_tokens)
        marked_injection = START_MARKER + injection + END_MARKER
        result = text + " " + marked_injection
    
    else:
        return text, []
    
    # Find all marker pairs first
    marker_pairs = []
    search_pos = 0
    while True:
        start_idx = result.find(START_MARKER, search_pos)
        if start_idx == -1:
            break
        end_idx = result.find(END_MARKER, start_idx)
        if end_idx == -1:
            break
        
        # Content is between the markers (exclusive of markers themselves)
        content_start_marked = start_idx + len(START_MARKER)
        content_end_marked = end_idx
        marker_pairs.append((start_idx, content_start_marked, content_end_marked, end_idx + len(END_MARKER)))
        search_pos = end_idx + len(END_MARKER)
    
    # Calculate positions in clean text (after all markers removed)
    # Each marker pair removes len(START_MARKER) + len(END_MARKER) characters
    marker_len = len(START_MARKER) + len(END_MARKER)
    regions = []
    
    for i, (marker_start, content_start_marked, content_end_marked, marker_end) in enumerate(marker_pairs):
        # How many marker characters are before this content?
        # All previous marker pairs contribute their full length
        offset = i * marker_len
        # Plus the START_MARKER of current pair
        offset += len(START_MARKER)
        
        # Positions in clean text
        content_start_clean = content_start_marked - offset
        content_end_clean = content_end_marked - offset
        regions.append((content_start_clean, content_end_clean))
    
    # Remove all markers
    clean_result = result.replace(START_MARKER, "").replace(END_MARKER, "")
    
    return clean_result, regions


def get_token_activations(
    text: str,
    model,
    sae,
    tokenizer,
    layer: int,
    feature_index: int,
    device: str,
    injected_regions: Optional[list[tuple[int, int]]] = None,
) -> tuple[list[str], Float[np.ndarray, "seq"], list[bool]]:
    """Get token-level activations for a single text.
    
    Args:
        injected_regions: List of (start_char, end_char) tuples marking injected text regions
    
    Returns:
        tuple: (token_strings, activations, is_injected_flags)
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
    input_ids: Int[torch.Tensor, "1 seq"] = tokens["input_ids"].to(device)
    offset_mapping = tokens.get("offset_mapping", [[]] * len(input_ids))[0]
    
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids, stop_at_layer=layer + 1)
        hidden: Float[torch.Tensor, "1 seq d_model"] = cache[hook_name]
        sae_out: Float[torch.Tensor, "1 seq n_features"] = sae.encode(hidden)
        activations = sae_out[0, :, feature_index].cpu().numpy()
    
    # Get token strings
    token_ids = input_ids[0].cpu().tolist()
    token_strs = []
    kept_indices = []
    is_injected = []
    
    for i, tid in enumerate(token_ids):
        token_str = tokenizer.decode([tid])
        
        if tid in tokenizer.all_special_ids:
            continue
        
        if not token_str:
            continue
        
        token_str = token_str.replace('\n', ' ').replace('\r', ' ')
        
        # Determine if this token overlaps with any injected region
        token_is_injected = False
        if injected_regions and i < len(offset_mapping):
            token_start, token_end = offset_mapping[i]
            for region_start, region_end in injected_regions:
                # Mark if overlaps AND not whitespace-only
                if not (token_end <= region_start or token_start >= region_end):
                    is_whitespace = token_str.strip() == ''
                    if not is_whitespace:
                        token_is_injected = True
                        break
        
        token_strs.append(token_str)
        kept_indices.append(i)
        is_injected.append(token_is_injected)
    
    filtered_activations = activations[kept_indices] if kept_indices else activations[:0]
    
    return token_strs, filtered_activations, is_injected


def generate_html_for_feature(
    feature_data: dict,
    examples: dict,
    output_path: Path,
    top_tokens: Optional[list[dict]] = None,
):
    """Generate HTML visualization for a single feature.
    
    Args:
        examples: Dict with keys like 'baseline', 'reasoning', 'prepend', etc.
                  Each value is a list of tuples: (tokens, activations, injected_tokens_set)
        top_tokens: List of dicts with 'token_str' and 'mean_activation' keys
    """
    
    feat_idx = feature_data["feature_index"]
    transfer_ratio = feature_data["best_transfer_ratio"]
    classification = feature_data["classification"]
    best_strategy = feature_data["best_strategy"]
    
    # Classification colors
    class_colors = {
        "token_driven": "#e74c3c",
        "partially_token_driven": "#f39c12",
        "weakly_token_driven": "#3498db",
        "context_dependent": "#2ecc71",
    }
    class_color = class_colors.get(classification, "#95a5a6")
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Feature {feat_idx} - Token Activation Visualization</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 15px 0;
            font-size: 32px;
        }}
        .metadata {{
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}
        .metadata-item {{
            display: flex;
            flex-direction: column;
        }}
        .metadata-label {{
            font-size: 12px;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metadata-value {{
            font-size: 20px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .classification-badge {{
            display: inline-block;
            padding: 8px 16px;
            background-color: {class_color};
            color: white;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section-title {{
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .example-container {{
            margin-bottom: 30px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .example-header {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #555;
            font-size: 14px;
        }}
        .text-container {{
            line-height: 2.2;
            font-size: 15px;
            font-family: 'Courier New', monospace;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
        }}
        .token {{
            display: inline;
            padding: 2px 1px;
            margin: 0;
            border-radius: 3px;
            transition: all 0.2s;
            position: relative;
        }}
        .token:hover {{
            transform: scale(1.1);
            z-index: 10;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        .token:hover::after {{
            content: attr(data-activation);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            white-space: nowrap;
            z-index: 100;
        }}
        .token-injected {{
            border: 2px solid #ff00ff;
            border-radius: 5px;
            padding: 3px 4px;
            font-weight: bold;
            box-shadow: 0 0 5px rgba(255, 0, 255, 0.3);
        }}
        .token-injected::before {{
            content: "💉";
            position: absolute;
            top: -12px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 10px;
        }}
        .legend {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 15px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
            font-size: 13px;
        }}
        .legend-gradient {{
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, 
                rgba(255, 255, 255, 0.9), 
                rgba(255, 200, 0, 0.9), 
                rgba(255, 100, 0, 0.9), 
                rgba(255, 0, 0, 0.9));
            border-radius: 3px;
            border: 1px solid #ccc;
        }}
        .strategy-label {{
            display: inline-block;
            padding: 4px 10px;
            background-color: #667eea;
            color: white;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-card {{
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Feature {feat_idx}: Token-Level Activations</h1>
        <div class="metadata">
            <div class="metadata-item">
                <span class="metadata-label">Classification</span>
                <span class="metadata-value">
                    <span class="classification-badge">{classification.replace('_', ' ').title()}</span>
                </span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Transfer Ratio</span>
                <span class="metadata-value">{transfer_ratio:.3f}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Best Strategy</span>
                <span class="metadata-value">{best_strategy.title()}</span>
            </div>
        </div>
    </div>
    
    <div class="section">
        <div class="section-title">📊 Activation Statistics</div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Baseline Mean</div>
                <div class="stat-value">{feature_data['baseline_mean']:.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Reasoning Mean</div>
                <div class="stat-value">{feature_data['reasoning_mean']:.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Reasoning Gap</div>
                <div class="stat-value">{feature_data['reasoning_mean'] - feature_data['baseline_mean']:.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active on Baseline</div>
                <div class="stat-value">{feature_data['baseline_nonzero_frac']*100:.1f}%</div>
            </div>
        </div>
    </div>
"""
    
    # Add top tokens section if provided
    if top_tokens:
        html += """
    <div class="section">
        <div class="section-title">🔝 Top Activating Tokens</div>
        <div style="font-size: 14px; color: #666; margin-bottom: 15px;">
            These tokens produce the strongest activation for this feature in the reasoning dataset.
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">
"""
        for i, tok_data in enumerate(top_tokens[:20]):  # Show top 20
            token_str = tok_data['token_str'].replace('<', '&lt;').replace('>', '&gt;')
            mean_act = tok_data['mean_activation']
            html += f"""
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #667eea;">
                <div style="font-family: 'Courier New', monospace; font-weight: bold; margin-bottom: 5px;">
                    "{token_str}"
                </div>
                <div style="font-size: 12px; color: #666;">
                    Activation: {mean_act:.2f}
                </div>
            </div>
"""
        html += """
        </div>
    </div>
"""
    
    # Add examples for each condition
    conditions = [
        ("baseline", "🔵 Baseline (Non-Reasoning)", "#3498db"),
        ("reasoning", "🟢 Reasoning Text", "#2ecc71"),
        ("prepend", "🔴 Injected: Prepend", "#e74c3c"),
        ("append", "🟤 Injected: Append", "#8b4513"),
        ("intersperse", "🟠 Injected: Intersperse", "#f39c12"),
        ("replace", "🟣 Injected: Replace", "#9b59b6"),
        ("bigram_before", "🔶 Injected: Bigram Before", "#ff8c00"),
        ("bigram_after", "🔷 Injected: Bigram After", "#1e90ff"),
        ("trigram", "⭐ Injected: Trigram", "#ffd700"),
        ("comma_list", "📝 Injected: Comma List", "#20b2aa"),
    ]
    
    for condition_key, condition_name, color in conditions:
        if condition_key not in examples:
            continue
            
        html += f"""
    <div class="section">
        <div class="section-title" style="border-bottom-color: {color};">{condition_name}</div>
"""
        
        for i, example_data in enumerate(examples[condition_key]):
            # Unpack example data (tokens, activations, is_injected_flags)
            if len(example_data) == 3:
                tokens, activations, is_injected_flags = example_data
            else:
                tokens, activations = example_data
                is_injected_flags = [False] * len(tokens)
            
            # Get activation range for normalization
            max_act = max(max(activations), 0.001) if len(activations) > 0 else 0.001
            
            html += f"""
        <div class="example-container">
            <div class="example-header">Example {i+1}</div>
            <div class="text-container">
"""
            
            for token, act, is_injected in zip(tokens, activations, is_injected_flags):
                # Normalize activation for color
                norm_act = min(act / max_act, 1.0)
                
                # Color based on activation strength
                if norm_act < 0.01:
                    bg_color = f"rgba(255, 255, 255, 0.9)"
                elif norm_act < 0.3:
                    bg_color = f"rgba(255, 200, 0, {0.3 + norm_act * 0.3})"
                elif norm_act < 0.7:
                    bg_color = f"rgba(255, 100, 0, {0.4 + norm_act * 0.4})"
                else:
                    bg_color = f"rgba(255, 0, 0, {0.5 + norm_act * 0.4})"
                
                # Use the is_injected flag directly
                injected_class = " token-injected" if is_injected else ""
                
                # Escape HTML (keep original spacing for display)
                token_escaped = token.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
                
                html += f'<span class="token{injected_class}" style="background-color: {bg_color};" data-activation="Act: {act:.3f}">{token_escaped}</span>'
            
            html += """
            </div>
        </div>
"""
        
        # Add legend
        legend_extra = ""
        if condition_key not in ["baseline", "reasoning"]:
            legend_extra = ' • <span style="border: 2px solid #ff00ff; padding: 2px 6px; border-radius: 3px; font-weight: bold;">💉 Purple border</span> = Injected token/phrase'
        
        html += f"""
        <div class="legend">
            <span><strong>Color Key:</strong></span>
            <div class="legend-gradient"></div>
            <span>Low → High Activation{legend_extra}</span>
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"  Saved visualization for feature {feat_idx} to: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("TOKEN-LEVEL ACTIVATION VISUALIZATION")
    print("=" * 60)
    
    # Load injection results
    print(f"\nLoading injection results from {args.injection_results}")
    with open(args.injection_results) as f:
        inj_data = json.load(f)
    
    features = inj_data.get("features", [])
    if not features:
        print("No features found")
        return
    
    # Sort by transfer ratio and take top N
    features_sorted = sorted(features, key=lambda f: f.get("best_transfer_ratio", 0), reverse=True)
    top_features = features_sorted[:args.n_features]
    
    print(f"\nVisualizing top {len(top_features)} features:")
    for f in top_features:
        print(f"  Feature {f['feature_index']}: "
              f"transfer={f['best_transfer_ratio']:.3f}, "
              f"class={f['classification']}")
    
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
    
    # Load example texts
    print("\n--- Loading Example Texts ---")
    
    # Non-reasoning texts
    pile = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    nonreasoning_texts = []
    for row in pile:
        text = row.get("text", "")
        if text and len(text) > 50:
            nonreasoning_texts.append(text[:300])
            if len(nonreasoning_texts) >= args.n_examples * 5:
                break
    
    # Reasoning texts (load more for context extraction)
    reasoning_texts = []
    if args.reasoning_dataset == "s1k":
        ds = load_dataset("simplescaling/s1K-1.1", split="train")
        for row in ds:
            for key in ["gemini_thinking_trajectory", "deepseek_thinking_trajectory"]:
                if row.get(key):
                    reasoning_texts.append(row[key])
                    if len(reasoning_texts) >= 200:  # More texts for context
                        break
            if len(reasoning_texts) >= 200:
                break
    else:
        ds = load_dataset("moremilk/General_Inquiry_Thinking-Chain-Of-Thought", split="train")
        for row in ds:
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict):
                text = metadata.get("reasoning", "")
                if text:
                    text = text.replace("<think>", "").replace("</think>", "").strip()
                    reasoning_texts.append(text)
                    if len(reasoning_texts) >= 200:
                        break
    
    print(f"Loaded {len(nonreasoning_texts)} non-reasoning and {len(reasoning_texts)} reasoning texts")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations for each feature
    print("\n--- Generating Visualizations ---")
    
    for feature_data in top_features:
        feat_idx = feature_data["feature_index"]
        print(f"\nProcessing feature {feat_idx}...")
        
        # Load top tokens for injection (strings only)
        top_tokens = load_top_tokens_for_feature(args.token_analysis, feat_idx, top_k=10)
        if len(top_tokens) < 3:
            print(f"  Not enough tokens for feature {feat_idx}, skipping")
            continue
        
        # Load top tokens with full data for display
        top_tokens_with_data = load_top_tokens_with_activations(args.token_analysis, feat_idx, top_k=20)
        
        # Extract token contexts for contextual strategies
        print(f"  Extracting token contexts from reasoning texts...")
        token_contexts = get_token_sequences_from_reasoning_texts(reasoning_texts, top_tokens)
        
        # Sample texts for this feature (shorter excerpts for visualization)
        sample_nonreasoning = random.sample(nonreasoning_texts, min(args.n_examples, len(nonreasoning_texts)))
        sample_reasoning = [text[:300] for text in random.sample(reasoning_texts, min(args.n_examples, len(reasoning_texts)))]
        
        examples = {
            "baseline": [],
            "reasoning": [],
            "prepend": [],
            "intersperse": [],
            "replace": [],
            "append": [],
            "bigram_before": [],
            "bigram_after": [],
            "trigram": [],
            "comma_list": [],
        }
        
        # Get baseline activations (no injected tokens)
        for text in sample_nonreasoning:
            tokens, acts, is_injected = get_token_activations(
                text, model, sae, tokenizer, args.layer, feat_idx, args.device
            )
            tokens = tokens[:args.max_seq_len]
            acts = acts[:args.max_seq_len]
            is_injected = is_injected[:args.max_seq_len]
            examples["baseline"].append((tokens, acts, is_injected))
        
        # Get reasoning activations (no injected tokens)
        for text in sample_reasoning:
            tokens, acts, is_injected = get_token_activations(
                text, model, sae, tokenizer, args.layer, feat_idx, args.device
            )
            tokens = tokens[:args.max_seq_len]
            acts = acts[:args.max_seq_len]
            is_injected = is_injected[:args.max_seq_len]
            examples["reasoning"].append((tokens, acts, is_injected))
        
        # Get injected activations for each strategy
        all_strategies = [
            "prepend", "append", "intersperse", "replace",
            "bigram_before", "bigram_after", "trigram", "comma_list"
        ]
        
        for strategy in all_strategies:
            for idx, text in enumerate(sample_nonreasoning):
                # Inject tokens and get regions
                injected_text, injected_regions = inject_tokens_into_text(
                    text, top_tokens, n_inject=3, strategy=strategy,
                    seed=feat_idx * 1000 + idx, token_contexts=token_contexts
                )
                
                # Get activations with region-based marking
                tokens, acts, is_injected = get_token_activations(
                    injected_text, model, sae, tokenizer, args.layer, feat_idx, args.device,
                    injected_regions=injected_regions
                )
                
                tokens = tokens[:args.max_seq_len]
                acts = acts[:args.max_seq_len]
                is_injected = is_injected[:args.max_seq_len]
                examples[strategy].append((tokens, acts, is_injected))
        
        # Generate HTML
        output_path = args.output_dir / f"feature_{feat_idx}.html"
        generate_html_for_feature(feature_data, examples, output_path, top_tokens=top_tokens_with_data)
    
    # Generate index page
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Feature Activation Visualizations</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{ margin: 0; }}
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        .feature-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .feature-title {{
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .feature-meta {{
            color: #666;
            font-size: 14px;
            margin-bottom: 15px;
        }}
        .view-button {{
            display: inline-block;
            padding: 10px 20px;
            background-color: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }}
        .view-button:hover {{
            background-color: #5568d3;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Token-Level Feature Activation Visualizations</h1>
        <p>Layer {args.layer} • {len(top_features)} features • {args.reasoning_dataset.upper()} dataset</p>
    </div>
    
    <div class="features-grid">
"""
    
    for feature_data in top_features:
        feat_idx = feature_data["feature_index"]
        transfer_ratio = feature_data["best_transfer_ratio"]
        classification = feature_data["classification"].replace("_", " ").title()
        
        index_html += f"""
        <div class="feature-card">
            <div class="feature-title">Feature {feat_idx}</div>
            <div class="feature-meta">
                Transfer Ratio: {transfer_ratio:.3f}<br>
                Classification: {classification}
            </div>
            <a href="feature_{feat_idx}.html" class="view-button">View Activations →</a>
        </div>
"""
    
    index_html += """
    </div>
</body>
</html>
"""
    
    index_path = args.output_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nAll visualizations saved to: {args.output_dir}")
    print(f"Open index page: {index_path}")


if __name__ == "__main__":
    main()
