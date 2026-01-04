"""
Generate LaTeX appendix section with LLM-generated feature interpretations (without activation visualization).

This simplified version doesn't require loading models and SAEs, and generates
interpretations without token-level activation visualization. Use this for quick
testing or if activation visualization is not needed.

Usage:
    python generate_interpretation_latex_simple.py \
        --results-dir results/cohens_d \
        --output docs/feature_interpretations_simple.tex
"""

import argparse
import json
from pathlib import Path
from typing import List


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    # Remove markdown bold (just remove the asterisks)
    text = text.replace('**', '')
    
    # Escape LaTeX special characters
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('$', r'\$'),
        ('%', r'\%'),
        ('_', r'\_'),
        ('#', r'\#'),
        ('&', r'\&'),
        ('^', r'\^{}'),
        ('~', r'\textasciitilde{}'),
        ('<', r'\textless{}'),
        ('>',r'\textgreater{}'),
    ]
    
    result = text
    for char, replacement in replacements:
        result = result.replace(char, replacement)
    
    return result


def generate_latex_document(results_dir: Path, output_path: Path):
    """Generate complete LaTeX document with all feature interpretations."""
    
    # Configurations to process
    configs = [
        ("gemma-3-12b-it", [17, 22, 27]),
        ("gemma-3-4b-it", [17, 22, 27]),
        ("deepseek-r1-distill-llama-8b", [19]),
    ]
    
    datasets = ["s1k", "general_inquiry_cot"]
    
    # Start building LaTeX content
    latex_lines = []
    
    # Header
    latex_lines.append(r"% Feature Interpretations Appendix")
    latex_lines.append(r"% Auto-generated from experimental results")
    latex_lines.append(r"")
    latex_lines.append(r"\section{LLM-guided feature interpretation results}\label{app: analysis}")
    latex_lines.append(r"")
    latex_lines.append(r"This section provides complete documentation of all LLM-generated feature interpretations. For each analyzed context-dependent feature, we present the refined interpretation from the LLM-guided analysis protocol, the binary classification (genuine reasoning feature or confound), confidence level, and examples of false positives (non-reasoning text that activates the feature) and false negatives (reasoning text that does not activate the feature). All features analyzed across all configurations were classified as confounds with high confidence in the majority of cases.")
    latex_lines.append(r"")
    
    total_features = 0
    
    for model_name, layers in configs:
        for layer in layers:
            for dataset in datasets:
                # Check if interpretation results exist
                interp_path = results_dir / model_name / dataset / f"layer{layer}" / "feature_interpretations.json"
                
                if not interp_path.exists():
                    print(f"  Skipping {model_name} layer {layer} {dataset} (no results)")
                    continue
                
                print(f"Processing {model_name} layer {layer} {dataset}...")
                
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
                
                # Process each feature
                for feat in features:
                    feat_idx = feat['feature_index']
                    
                    # Feature header
                    latex_lines.append(r"\noindent\textbf{Feature " + str(feat_idx) + r"}")
                    latex_lines.append(r"")
                    
                    # Interpretation
                    interp_text = escape_latex(feat['refined_interpretation'])
                    latex_lines.append(r"\textit{Interpretation:} " + interp_text)
                    latex_lines.append(r"")
                    
                    # Classification
                    is_genuine = feat['is_genuine_reasoning_feature']
                    confidence = feat['confidence']
                    classification_text = "Genuine Reasoning Feature" if is_genuine else "Confound"
                    latex_lines.append(r"\textit{Classification:} " + classification_text + 
                                     f" (Confidence: {confidence})")
                    latex_lines.append(r"")
                    
                    # Summary
                    summary_text = escape_latex(feat['summary'])
                    latex_lines.append(r"\textit{Summary:} " + summary_text)
                    latex_lines.append(r"")
                    
                    # Activates on
                    if feat['activates_on']:
                        activates_list = feat['activates_on'][:3]
                        activates_text = "; ".join(escape_latex(item) for item in activates_list)
                        latex_lines.append(r"\textit{Activates on:} " + activates_text)
                        latex_lines.append(r"")
                    
                    # False positives
                    fps = [ex for ex in feat['false_positive_examples'] 
                          if ex.get('is_valid_counterexample')]
                    if fps:
                        latex_lines.append(r"\textit{False Positives (Non-reasoning text that activates):}")
                        for i, fp in enumerate(fps[:3], 1):
                            text = fp['text'][:250]
                            if len(fp['text']) > 250:
                                text += "..."
                            text_escaped = escape_latex(text)
                            act_val = fp.get('max_activation', 0)
                            latex_lines.append(f"{i}. {text_escaped} [Activation: {act_val:.1f}]")
                        latex_lines.append(r"")
                    
                    # False negatives
                    fns = [ex for ex in feat['false_negative_examples'] 
                          if ex.get('is_valid_counterexample')]
                    if fns:
                        latex_lines.append(r"\textit{False Negatives (Reasoning text that does not activate):}")
                        for i, fn in enumerate(fns[:3], 1):
                            text = fn['text'][:250]
                            if len(fn['text']) > 250:
                                text += "..."
                            text_escaped = escape_latex(text)
                            act_val = fn.get('max_activation', 0)
                            latex_lines.append(f"{i}. {text_escaped} [Activation: {act_val:.1f}]")
                        latex_lines.append(r"")
                    
                    latex_lines.append(r"\vspace{0.5cm}")
                    latex_lines.append(r"")
                    
                    total_features += 1
    
    # Write to file
    print(f"\nWriting LaTeX file to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"\nGeneration complete!")
    print(f"  Total features documented: {total_features}")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX appendix with feature interpretations")
    parser.add_argument("--results-dir", type=Path, default=Path("results/cohens_d"),
                       help="Directory containing experimental results")
    parser.add_argument("--output", type=Path, default=Path("docs/feature_interpretations_simple.tex"),
                       help="Output LaTeX file path")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FEATURE INTERPRETATION LATEX GENERATOR (SIMPLE)")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output file: {args.output}")
    print("=" * 60)
    
    generate_latex_document(args.results_dir, args.output)
