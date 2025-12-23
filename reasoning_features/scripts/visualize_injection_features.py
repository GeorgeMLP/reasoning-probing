"""
Visualize reasoning features with token injection experiment results using sae_dashboard.

This script creates interactive visualizations for features tested in the token injection
experiment, showing their activations on baseline, reasoning, and injected texts.

## Usage

```bash
python visualize_injection_features.py \
    --injection-results results/test/gemma-2-9b/s1k/layer12/injection_results.json \
    --layer 12 \
    --n-features 5 \
    --output-dir visualizations/layer12
```
"""

import argparse
import json
from pathlib import Path
import sys

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize features from token injection experiment using sae_dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--injection-results",
        type=Path,
        required=True,
        help="Path to injection_results.json",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=10,
        help="Number of features to visualize (default: 10)",
    )
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
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=256,
        help="Number of prompts to use for visualization (default: 256)",
    )
    parser.add_argument(
        "--minibatch-size-tokens",
        type=int,
        default=64,
        help="Minibatch size for token processing (default: 64)",
    )
    parser.add_argument(
        "--minibatch-size-features",
        type=int,
        default=32,
        help="Minibatch size for feature processing (default: 32)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Data type for computations",
    )
    
    return parser.parse_args()


def load_prompts_from_pile(n_prompts: int = 256, max_length: int = 128) -> list[str]:
    """Load random prompts from the Pile dataset."""
    print(f"Loading {n_prompts} prompts from Pile dataset...")
    pile = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    
    prompts = []
    for row in pile:
        text = row.get("text", "")
        if text and len(text) > 50:
            prompts.append(text[:max_length * 4])  # Approximate char count
            if len(prompts) >= n_prompts:
                break
    
    return prompts


def main():
    args = parse_args()
    
    print("=" * 60)
    print("FEATURE VISUALIZATION WITH SAE DASHBOARD")
    print("=" * 60)
    
    # Load injection results
    print(f"\nLoading injection results from {args.injection_results}")
    with open(args.injection_results) as f:
        inj_data = json.load(f)
    
    # Get top features by transfer ratio
    features = inj_data.get("features", [])
    if not features:
        print("No features found in injection results")
        return
    
    # Sort by best transfer ratio
    features_sorted = sorted(features, key=lambda f: f.get("best_transfer_ratio", 0), reverse=True)
    top_features = features_sorted[:args.n_features]
    feature_indices = [f["feature_index"] for f in top_features]
    
    print(f"\nTop {len(feature_indices)} features by transfer ratio:")
    for f in top_features:
        print(f"  Feature {f['feature_index']}: "
              f"transfer={f['best_transfer_ratio']:.3f}, "
              f"classification={f['classification']}")
    
    # Load model and SAE
    print("\n--- Loading Model and SAE ---")
    from sae_lens import SAE, HookedSAETransformer
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    
    model = HookedSAETransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        dtype=dtype,
    )
    
    sae_id = args.sae_id_format.format(layer=args.layer)
    sae, _, _ = SAE.from_pretrained(
        release=args.sae_name,
        sae_id=sae_id,
        device=args.device,
    )
    sae.fold_W_dec_norm()
    
    print(f"Model: {args.model_name}")
    print(f"SAE: {args.sae_name} / {sae_id}")
    print(f"Hook point: {sae.cfg.hook_name}")
    
    # Load prompts
    prompts = load_prompts_from_pile(n_prompts=args.n_prompts)
    
    # Tokenize prompts
    print(f"\nTokenizing {len(prompts)} prompts...")
    tokens = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )["input_ids"]
    
    print(f"Token dataset shape: {tokens.shape}")
    
    # Configure visualization
    print("\n--- Configuring SAE Dashboard ---")
    from sae_dashboard.sae_vis_data import SaeVisConfig
    from sae_dashboard.sae_vis_runner import SaeVisRunner
    
    config = SaeVisConfig(
        hook_point=sae.cfg.hook_name,
        features=feature_indices,
        minibatch_size_features=args.minibatch_size_features,
        minibatch_size_tokens=args.minibatch_size_tokens,
        device=args.device,
        dtype=args.dtype,
    )
    
    # Generate visualization data
    print("\n--- Generating Visualization Data ---")
    print(f"Processing {len(feature_indices)} features...")
    
    runner = SaeVisRunner(config)
    vis_data = runner.run(encoder=sae, model=model, tokens=tokens)
    
    # Save visualizations
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Saving Visualizations ---")
    from sae_dashboard.data_writing_fns import save_feature_centric_vis
    
    # Save combined visualization
    combined_path = args.output_dir / f"layer{args.layer}_features.html"
    save_feature_centric_vis(
        sae_vis_data=vis_data,
        filename=str(combined_path)
    )
    print(f"Saved combined visualization to: {combined_path}")
    
    # Save individual feature visualizations
    for i, feat_idx in enumerate(feature_indices):
        # Create single-feature visualization
        single_config = SaeVisConfig(
            hook_point=sae.cfg.hook_name,
            features=[feat_idx],
            minibatch_size_features=1,
            minibatch_size_tokens=args.minibatch_size_tokens,
            device=args.device,
            dtype=args.dtype,
        )
        
        single_runner = SaeVisRunner(single_config)
        single_vis_data = single_runner.run(encoder=sae, model=model, tokens=tokens)
        
        feat_path = args.output_dir / f"layer{args.layer}_feature{feat_idx}.html"
        save_feature_centric_vis(
            sae_vis_data=single_vis_data,
            filename=str(feat_path)
        )
        print(f"  Saved feature {feat_idx} to: {feat_path}")
    
    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "sae_name": args.sae_name,
        "sae_id": sae_id,
        "layer": args.layer,
        "features": [
            {
                "index": f["feature_index"],
                "transfer_ratio": f["best_transfer_ratio"],
                "classification": f["classification"],
                "baseline_mean": f["baseline_mean"],
                "reasoning_mean": f["reasoning_mean"],
            }
            for f in top_features
        ],
        "n_prompts": len(prompts),
    }
    
    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nView visualizations at: {args.output_dir}")
    print(f"Main dashboard: {combined_path}")


if __name__ == "__main__":
    main()
