"""
Simple Script to Recompute Missing Gemma-3-12B Features

This is a simplified script that recomputes LLM interpretations for the 3 missing
Gemma-3-12B features. It prints results to console for manual addition to papers.

Usage:
    python recompute_gemma_missing.py
"""

import json
import os
import sys
from pathlib import Path
import time

import torch
import numpy as np
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# The missing features we need to recompute
EXPERIMENTS = [
    {
        "model": "google/gemma-3-12b-it",
        "model_short": "gemma-3-12b-it",
        "sae_name": "gemma-scope-3-12b-it-res-all",
        "dataset": "s1k",
        "layer": 22,
        "feature_index": 1010,
        "exp_dir": "results/cohens_d/gemma-3-12b-it/s1k/layer22",
    },
    {
        "model": "google/gemma-3-12b-it",
        "model_short": "gemma-3-12b-it",
        "sae_name": "gemma-scope-3-12b-it-res-all",
        "dataset": "general_inquiry_cot",
        "layer": 22,
        "feature_index": 1141,
        "exp_dir": "results/cohens_d/gemma-3-12b-it/general_inquiry_cot/layer22",
    },
    {
        "model": "google/gemma-3-12b-it",
        "model_short": "gemma-3-12b-it",
        "sae_name": "gemma-scope-3-12b-it-res-all",
        "dataset": "general_inquiry_cot",
        "layer": 27,
        "feature_index": 4875,
        "exp_dir": "results/cohens_d/gemma-3-12b-it/general_inquiry_cot/layer27",
    },
]


def load_token_data(token_analysis_path: Path, feature_index: int) -> list[str]:
    """Load top tokens for a feature."""
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    for feat in data.get("features", []):
        if feat["feature_index"] == feature_index:
            return [t["token_str"] for t in feat.get("top_tokens", [])]
    return []


def main():
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)
    
    print(f"{'='*80}")
    print("RECOMPUTING MISSING GEMMA-3-12B FEATURES")
    print(f"{'='*80}\n")
    
    from sae_lens import SAE, HookedSAETransformer
    from datasets import load_dataset
    from reasoning_features.scripts.analyze_feature_interpretation import (
        LLMClient,
        FeatureAnalyzer,
    )
    
    # Process each experiment
    for exp_idx, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*80}")
        print(f"Feature {exp_idx}/{len(EXPERIMENTS)}: {exp['model_short']}/{exp['dataset']}/layer{exp['layer']}/feature{exp['feature_index']}")
        print(f"{'='*80}\n")
        
        # Load model and SAE (reuse if same as previous)
        if exp_idx == 1 or EXPERIMENTS[exp_idx-2]['model'] != exp['model']:
            print(f"Loading model: {exp['model']}")
            model = HookedSAETransformer.from_pretrained_no_processing(
                exp['model'],
                device="cuda",
                dtype=torch.bfloat16,
            )
            tokenizer = model.tokenizer
            print("Model loaded!")
        
        # Load SAE for this layer
        sae_id = f"layer_{exp['layer']}_width_16k_l0_small"
        print(f"Loading SAE: {exp['sae_name']}/{sae_id}")
        sae = SAE.from_pretrained(
            release=exp['sae_name'],
            sae_id=sae_id,
            device="cuda",
        )
        if isinstance(sae, tuple):
            sae = sae[0]
        print("SAE loaded!")
        
        # Load reasoning texts
        print(f"Loading reasoning dataset: {exp['dataset']}")
        reasoning_texts = []
        
        if exp["dataset"] == "s1k":
            ds = load_dataset("simplescaling/s1K-1.1", split="train")
            for row in ds:
                for key in ["gemini_thinking_trajectory", "deepseek_thinking_trajectory"]:
                    if row.get(key):
                        reasoning_texts.append(row[key][:1000])
                        if len(reasoning_texts) >= 200:
                            break
                if len(reasoning_texts) >= 200:
                    break
        else:  # general_inquiry_cot
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
        
        print(f"Loaded {len(reasoning_texts)} reasoning texts")
        
        # Load top tokens
        token_analysis_path = Path(exp["exp_dir"]) / "token_analysis.json"
        top_tokens = load_token_data(token_analysis_path, exp['feature_index'])
        print(f"Top tokens: {top_tokens[:10]}")
        
        # Initialize LLM client and analyzer
        llm_client = LLMClient(api_key, "google/gemini-3-pro-preview")
        analyzer = FeatureAnalyzer(model, sae, tokenizer, llm_client, exp['layer'], "cuda")
        
        # Override collect_activation_examples to use adaptive threshold
        original_method = analyzer.collect_activation_examples
        
        def adaptive_collect_activation_examples(
            feature_index: int,
            reasoning_texts: list[str],
            n_examples: int = 10,
        ) -> list[dict]:
            """Collect examples with adaptive threshold."""
            all_activations = []
            text_data = []
            
            # First pass: collect all activations
            print("  Collecting activations from reasoning texts...")
            for text in reasoning_texts[:500]:
                max_act, mean_act, top_tokens_act = analyzer.get_activation(text, feature_index)
                all_activations.append(max_act)
                text_data.append({
                    "text": text[:500],
                    "max_activation": max_act,
                    "mean_activation": mean_act,
                    "top_tokens": top_tokens_act[:10],
                })
            
            # Use 50th percentile or 1.0, whichever is higher, as threshold
            threshold = max(np.percentile(all_activations, 50), 1.0)
            
            print(f"  Adaptive threshold: {threshold:.2f} (max: {max(all_activations):.2f}, " +
                  f"mean: {np.mean(all_activations):.2f}, median: {np.median(all_activations):.2f})")
            
            # Filter examples above threshold
            examples = [ex for ex in text_data if ex["max_activation"] > threshold]
            print(f"  Found {len(examples)} examples above threshold")
            
            # Sort by activation and return top N
            examples.sort(key=lambda x: x["max_activation"], reverse=True)
            return examples[:n_examples]
        
        # Temporarily replace the method
        analyzer.collect_activation_examples = adaptive_collect_activation_examples
        
        # Run analysis
        try:
            print("\nRunning LLM-guided feature analysis...")
            interpretation = analyzer.analyze_feature(
                exp['feature_index'],
                reasoning_texts,
                top_tokens,
                max_iterations=10,
                min_false_positives=3,
                min_false_negatives=3,
                threshold_ratio=0.5,
            )
            
            # Print result in a nice format
            print(f"\n{'='*80}")
            print(f"RESULT FOR FEATURE {exp['feature_index']}")
            print(f"{'='*80}\n")
            
            print(f"Model: {exp['model_short']}")
            print(f"Dataset: {exp['dataset']}")
            print(f"Layer: {exp['layer']}")
            print(f"Feature Index: {exp['feature_index']}")
            print()
            print(f"Is Genuine Reasoning Feature: {interpretation.is_genuine_reasoning_feature}")
            print(f"Confidence: {interpretation.confidence}")
            print(f"Iterations Used: {interpretation.iterations_used}")
            print()
            print(f"Summary:")
            print(f"  {interpretation.summary}")
            print()
            print(f"Refined Interpretation:")
            for line in interpretation.refined_interpretation.split('\n'):
                print(f"  {line}")
            print()
            print(f"Activates On ({len(interpretation.activates_on)} items):")
            for item in interpretation.activates_on:
                print(f"  - {item}")
            print()
            print(f"Does Not Activate On ({len(interpretation.does_not_activate_on)} items):")
            for item in interpretation.does_not_activate_on:
                print(f"  - {item}")
            print()
            print(f"False Positives: {len(interpretation.false_positive_examples)}")
            print(f"False Negatives: {len(interpretation.false_negative_examples)}")
            print()
            
            # Also save as JSON
            output_file = Path(f"feature_{exp['feature_index']}_recomputed.json")
            with open(output_file, 'w') as f:
                json.dump(asdict(interpretation), f, indent=2)
            print(f"Full results saved to: {output_file}")
            
            # Rate limiting
            if exp_idx < len(EXPERIMENTS):
                print("\nWaiting 5 seconds before next feature...")
                time.sleep(5)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # Restore original method
            analyzer.collect_activation_examples = original_method
        
        # Clean up SAE
        del sae
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print("ALL FEATURES RECOMPUTED")
    print(f"{'='*80}\n")
    print("You can now manually add these results to your paper tables.")


if __name__ == "__main__":
    main()
