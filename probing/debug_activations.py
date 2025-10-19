"""
Debug script to examine the collected activations and find trivial differences.
"""

import torch
from pathlib import Path
import numpy as np

def analyze_activations(activation_path='data/probing/experiments/quick_test/activations/activations.pt'):
    """Analyze the collected activations to find trivial differences."""
    print("=" * 80)
    print("ACTIVATION ANALYSIS")
    print("=" * 80)
    print(f"Loading from: {activation_path}\n")
    
    # Load activations
    data = torch.load(activation_path)
    normal = data['normal']
    reasoning = data['reasoning']
    
    print(f"\n--- Dataset Sizes ---")
    print(f"Normal samples: {len(normal['original'])}")
    print(f"Reasoning samples: {len(reasoning['original'])}")
    
    print(f"\n--- Activation Statistics ---")
    for name, acts in [('Normal', normal), ('Reasoning', reasoning)]:
        print(f"\n{name}:")
        for act_type in ['original', 'reconstructed', 'residue']:
            act = acts[act_type]
            print(f"  {act_type}:")
            print(f"    Shape: {act.shape}")
            print(f"    Mean: {act.mean():.4f}")
            print(f"    Std: {act.std():.4f}")
            print(f"    Min: {act.min():.4f}")
            print(f"    Max: {act.max():.4f}")
            print(f"    Norm (per sample): mean={act.norm(dim=1).mean():.4f}, std={act.norm(dim=1).std():.4f}")
    
    print(f"\n--- Distribution Comparison ---")
    # Check if distributions are completely separable
    for act_type in ['original', 'reconstructed', 'residue']:
        normal_norms = normal[act_type].norm(dim=1)
        reasoning_norms = reasoning[act_type].norm(dim=1)
        
        # Check if ranges overlap
        normal_min, normal_max = normal_norms.min(), normal_norms.max()
        reasoning_min, reasoning_max = reasoning_norms.min(), reasoning_norms.max()
        
        overlap = not (normal_max < reasoning_min or reasoning_max < normal_min)
        
        print(f"\n{act_type} norms:")
        print(f"  Normal range: [{normal_min:.4f}, {normal_max:.4f}]")
        print(f"  Reasoning range: [{reasoning_min:.4f}, {reasoning_max:.4f}]")
        print(f"  Ranges overlap: {overlap}")
        
        if not overlap:
            print(f"  ⚠️  NO OVERLAP! Trivially separable by norm alone!")
    
    print(f"\n--- Checking for Zero/Constant Patterns ---")
    for name, acts in [('Normal', normal), ('Reasoning', reasoning)]:
        for act_type in ['original', 'reconstructed', 'residue']:
            act = acts[act_type]
            # Check if any dimensions are constant across all samples
            variance_per_dim = act.var(dim=0)
            zero_var_dims = (variance_per_dim < 1e-6).sum().item()
            print(f"{name} {act_type}: {zero_var_dims} / {act.shape[1]} dimensions have zero variance")
    
    print(f"\n--- Dimensionwise Mean Differences ---")
    for act_type in ['original', 'reconstructed', 'residue']:
        normal_mean = normal[act_type].mean(dim=0)
        reasoning_mean = reasoning[act_type].mean(dim=0)
        diff = (normal_mean - reasoning_mean).abs()
        
        print(f"\n{act_type}:")
        print(f"  Max abs difference: {diff.max():.4f}")
        print(f"  Mean abs difference: {diff.mean():.4f}")
        print(f"  Dimensions with |diff| > 0.1: {(diff > 0.1).sum().item()}")
        
        # Check if many dimensions have large systematic differences
        large_diff_ratio = (diff > 0.1).sum().item() / len(diff)
        if large_diff_ratio > 0.5:
            print(f"  ⚠️  {large_diff_ratio*100:.1f}% of dimensions have large systematic differences!")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nIf activations have:")
    print("  - Non-overlapping norms → Trivially separable")
    print("  - Large systematic mean differences → Easy to classify")  
    print("  - Constant dimensions → Potential issues with data")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/probing/experiments/quick_test/activations/activations.pt'
    analyze_activations(path)

