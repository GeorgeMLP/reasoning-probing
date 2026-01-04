"""
Generate LaTeX appendix section with all LLM-generated feature interpretations.

This script creates a complete LaTeX file documenting all analyzed features,
including token-level activation visualizations, LLM interpretations, and
generated counterexamples.

Usage:
    python generate_interpretation_latex.py \
        --results-dir results/cohens_d \
        --output docs/feature_interpretations.tex \
        --n-tokens-visualize 20 \
        --n-examples 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    replacements = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '%': r'\%',
        '_': r'\_',
        '#': r'\#',
        '&': r'\&',
        '^': r'\^{}',
        '~': r'\textasciitilde{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    
    result = text
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    
    return result


def token_to_latex(token: str) -> str:
    """Convert a token string to LaTeX-safe format."""
    # Replace special characters
    token = token.replace('\n', ' ')
    token = token.replace('\t', '  ')
    token = token.replace('\\', r'\textbackslash{}')
    token = token.replace('{', r'\{')
    token = token.replace('}', r'\}')
    token = token.replace('^', r'\^{}')
    token = token.replace(',', r'{,}')
    token = token.replace('/', r'{/}')
    token = token.replace('%', r'\%')
    token = token.replace('$', r'\$')
    token = token.replace('_', r'\_')
    token = token.replace('#', r'\#')
    token = token.replace('&', r'\&')
    token = token.replace('(', r'{(}')
    token = token.replace(')', r'{)}')
    token = token.replace("'", r'\textquotesingle{}')
    token = token.replace('`', r'\textasciigrave{}')
    token = token.replace('~', r'\textasciitilde{}')
    token = token.replace('<', r'\textless{}')
    token = token.replace('>', r'\textgreater{}')
    
    # Handle spaces
    if token.startswith(' '):
        token = r'\ ' + token[1:]
    if token == ' ':
        token = r'\ '
    
    return token


def activations_to_highlight_pairs(tokens: List[str], activations: List[float]) -> str:
    """Convert tokens and activations to LaTeX highlightpairs format.
    
    Activations are normalized to [0, 1] range and tokens are LaTeX-escaped.
    """
    # Normalize activations to [0, 1]
    max_act = max(activations) if activations else 1.0
    if max_act > 0:
        normalized = [min(act / max_act, 1.0) for act in activations]
    else:
        normalized = [0.0] * len(activations)
    
    # Build the pairs string
    pairs = []
    for i, (tok, act) in enumerate(zip(tokens, normalized)):
        tok_latex = token_to_latex(tok)
        # Remove leading space from first token
        if i == 0 and tok_latex.startswith(r'\ '):
            tok_latex = tok_latex[2:]
        
        pairs.append(f"{tok_latex}/{act:.2f}")
    
    return ", ".join(pairs)


def get_top_activating_examples(
    model,
    sae,
    tokenizer,
    feature_index: int,
    reasoning_texts: List[str],
    layer: int,
    device: str,
    n_examples: int = 3,
    n_tokens_visualize: int = 20,
) -> List[Tuple[str, List[str], List[float]]]:
    """Get top N examples with token-level activations.
    
    Returns list of (full_text, tokens_in_window, activations_in_window).
    """
    examples = []
    
    for text in reasoning_texts[:500]:
        # Get activations
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                inputs["input_ids"],
                names_filter=[f"blocks.{layer}.hook_resid_post"],
            )
            hidden = cache[f"blocks.{layer}.hook_resid_post"]
            sae_acts = sae.encode(hidden)
            acts = sae_acts[0, :, feature_index].cpu().numpy()
        
        # Decode tokens
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        
        # Find window with highest mean activation
        if len(acts) > n_tokens_visualize:
            window_means = []
            for i in range(len(acts) - n_tokens_visualize + 1):
                window_mean = np.mean(acts[i:i + n_tokens_visualize])
                window_means.append(window_mean)
            
            best_start = int(np.argmax(window_means))
            best_end = best_start + n_tokens_visualize
            
            window_tokens = tokens[best_start:best_end]
            window_acts = acts[best_start:best_end].tolist()
            max_act = float(acts.max())
        else:
            window_tokens = tokens
            window_acts = acts.tolist()
            max_act = float(acts.max())
        
        examples.append((text, window_tokens, window_acts, max_act))
    
    # Sort by max activation and return top N
    examples.sort(key=lambda x: x[3], reverse=True)
    return [(text, tokens, acts) for text, tokens, acts, _ in examples[:n_examples]]


def generate_latex_document(
    results_dir: Path,
    output_path: Path,
    n_tokens_visualize: int = 20,
    n_examples: int = 3,
):
    """Generate complete LaTeX document with all feature interpretations."""
    
    # Load model and SAE for activation computation
    print("Loading models (this may take a while)...")
    from sae_lens import SAE, HookedSAETransformer
    from datasets import load_dataset
    
    # We'll need to load models on demand
    loaded_models = {}
    
    # Configurations to process
    configs = [
        # ("gemma-3-12b-it", "gemma-scope-2-12b-it-res-all", [17, 22, 27]),
        # ("gemma-3-4b-it", "gemma-scope-2-4b-it-res-all", [17, 22, 27]),
        # ("deepseek-r1-distill-llama-8b", "gemma-scope-2-8b-it-res-all", [19]),
        ("gemma-3-4b-it", "gemma-scope-2-4b-it-res-all", [27]),
    ]
    
    # datasets = ["s1k", "general_inquiry_cot"]
    datasets = ["s1k"]
    
    # Start building LaTeX content
    latex_lines = []
    
    # Header
    latex_lines.append(r"% Feature Interpretations Appendix")
    latex_lines.append(r"% Auto-generated from experimental results")
    latex_lines.append(r"")
    latex_lines.append(r"\section{LLM-guided feature interpretation results}\label{app: analysis}")
    latex_lines.append(r"")
    latex_lines.append(r"This section provides complete documentation of all LLM-generated feature interpretations, including high-activation examples with token-level visualization, refined interpretations, and generated counterexamples. For each feature, we show three high-activation examples from the reasoning corpus with tokens colored by activation strength (darker blue indicates higher activation), followed by the LLM's interpretation and classification, and examples of false positives (non-reasoning text that activates the feature) and false negatives (reasoning text that does not activate the feature).")
    latex_lines.append(r"")
    
    total_features = 0
    
    for model_name, sae_name, layers in configs:
        for layer in layers:
            for dataset in datasets:
                # Check if interpretation results exist
                interp_path = results_dir / model_name / dataset / f"layer{layer}" / "feature_interpretations.json"
                
                if not interp_path.exists():
                    print(f"  Skipping {model_name} layer {layer} {dataset} (no results)")
                    continue
                
                print(f"\nProcessing {model_name} layer {layer} {dataset}...")
                
                # Load interpretation results
                with open(interp_path) as f:
                    interp_data = json.load(f)
                
                features = interp_data.get("features", [])
                if not features:
                    continue
                
                # Create subsection
                model_display = model_name.replace("-", " ").replace("_", " ").title()
                dataset_display = "s1K-1.1" if dataset == "s1k" else "General Inquiry CoT"
                
                latex_lines.append(r"\subsection{" + f"{model_display}, Layer {layer}, {dataset_display}" + r"}")
                latex_lines.append(r"")
                
                # Load model and SAE if not already loaded
                model_key = f"{model_name}_layer{layer}"
                if model_key not in loaded_models:
                    print(f"  Loading model {model_name}...")
                    
                    model = HookedSAETransformer.from_pretrained_no_processing(
                        f"google/{model_name}" if "gemma" in model_name else model_name,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    
                    sae_id = f"layer_{layer}_width_16k_l0_small"
                    sae = SAE.from_pretrained(
                        release=sae_name,
                        sae_id=sae_id,
                        device="cuda",
                    )
                    if isinstance(sae, tuple):
                        sae = sae[0]
                    
                    tokenizer = model.tokenizer
                    loaded_models[model_key] = (model, sae, tokenizer)
                    print(f"  Model loaded!")
                else:
                    model, sae, tokenizer = loaded_models[model_key]
                
                # Load reasoning texts
                print(f"  Loading reasoning texts...")
                reasoning_texts = []
                if dataset == "s1k":
                    ds = load_dataset("simplescaling/s1K-1.1", split="train")
                    for row in ds:
                        for key in ["deepseek_thinking_trajectory", "gemini_thinking_trajectory"]:
                            if row.get(key):
                                reasoning_texts.append(row[key][:1000])
                                if len(reasoning_texts) >= 200:
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
                                reasoning_texts.append(text[:1000])
                                if len(reasoning_texts) >= 200:
                                    break
                
                print(f"  Loaded {len(reasoning_texts)} reasoning texts")
                
                # Process each feature
                for feat in features:
                    feat_idx = feat['feature_index']
                    print(f"    Processing feature {feat_idx}...")
                    
                    # Get top activating examples
                    try:
                        top_examples = get_top_activating_examples(
                            model, sae, tokenizer, feat_idx, reasoning_texts,
                            layer, "cuda", n_examples, n_tokens_visualize
                        )
                    except Exception as e:
                        print(f"      Error getting activations: {e}")
                        top_examples = []
                    
                    # Start feature section
                    latex_lines.append(r"\noindent\textbf{Feature " + str(feat_idx) + r"}")
                    latex_lines.append(r"")
                    
                    # High-activation examples
                    if top_examples:
                        latex_lines.append(r"\textit{High-Activation Examples:}")
                        latex_lines.append(r"")
                        
                        for i, (full_text, tokens, acts) in enumerate(top_examples, 1):
                            pairs = activations_to_highlight_pairs(tokens, acts)
                            latex_lines.append(f"\\textit{{Example {i}:}} \\highlightpairs{{{pairs}}}")
                            latex_lines.append(r"")
                    
                    # LLM Interpretation
                    latex_lines.append(r"\textit{Interpretation:} " + escape_latex(feat['refined_interpretation']))
                    latex_lines.append(r"")
                    
                    # Classification
                    is_genuine = feat['is_genuine_reasoning_feature']
                    confidence = feat['confidence']
                    latex_lines.append(r"\textit{Classification:} " + 
                                     ("Genuine Reasoning Feature" if is_genuine else "Confound") +
                                     f" (Confidence: {confidence})")
                    latex_lines.append(r"")
                    
                    # False positives
                    fps = [ex for ex in feat['false_positive_examples'] if ex.get('is_valid_counterexample')]
                    if fps:
                        latex_lines.append(r"\textit{False Positives (Non-reasoning text that activates):}")
                        latex_lines.append(r"\begin{enumerate}[left=0pt, itemsep=0pt]")
                        for i, fp in enumerate(fps[:3], 1):
                            text_escaped = escape_latex(fp['text'][:200])
                            if len(fp['text']) > 200:
                                text_escaped += "..."
                            latex_lines.append(f"\t\\item {text_escaped}")
                        latex_lines.append(r"\end{enumerate}")
                        latex_lines.append(r"")
                    
                    # False negatives
                    fns = [ex for ex in feat['false_negative_examples'] if ex.get('is_valid_counterexample')]
                    if fns:
                        latex_lines.append(r"\textit{False Negatives (Reasoning text that does not activate):}")
                        latex_lines.append(r"\begin{enumerate}[left=0pt, itemsep=0pt]")
                        for i, fn in enumerate(fns[:3], 1):
                            text_escaped = escape_latex(fn['text'][:200])
                            if len(fn['text']) > 200:
                                text_escaped += "..."
                            latex_lines.append(f"\t\\item {text_escaped}")
                        latex_lines.append(r"\end{enumerate}")
                        latex_lines.append(r"")
                    
                    latex_lines.append(r"\vspace{0.3cm}")
                    latex_lines.append(r"")
                    
                    total_features += 1
                
                # Clear CUDA cache between configurations
                torch.cuda.empty_cache()
    
    # Write to file
    print(f"\nWriting LaTeX file to {output_path}...")
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"\nGeneration complete!")
    print(f"  Total features documented: {total_features}")
    print(f"  Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX appendix with feature interpretations")
    parser.add_argument("--results-dir", type=Path, default=Path("results/cohens_d"),
                       help="Directory containing experimental results")
    parser.add_argument("--output", type=Path, default=Path("docs/feature_interpretations.tex"),
                       help="Output LaTeX file path")
    parser.add_argument("--n-tokens-visualize", type=int, default=20,
                       help="Number of tokens to visualize per example")
    parser.add_argument("--n-examples", type=int, default=3,
                       help="Number of high-activation examples per feature")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FEATURE INTERPRETATION LATEX GENERATOR")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output file: {args.output}")
    print(f"Tokens to visualize: {args.n_tokens_visualize}")
    print(f"Examples per feature: {args.n_examples}")
    print("=" * 60)
    
    generate_latex_document(
        args.results_dir,
        args.output,
        args.n_tokens_visualize,
        args.n_examples,
    )


if __name__ == "__main__":
    main()
