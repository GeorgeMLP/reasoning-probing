"""
Visualize Strategy Redundancy Analysis

Create visualizations showing which strategies are essential for each experiment.

Usage:
    python visualize_strategy_redundancy.py \\
        --analysis-file results/cohens_d/minimal_strategies_analysis.json \\
        --output-dir results/cohens_d/strategy_analysis
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_strategy_heatmap(per_experiment: dict, output_path: Path):
    """
    Create a heatmap showing which strategies are in the minimal set for each experiment.
    """
    # Extract experiment names and strategies
    experiments = sorted(per_experiment.keys())
    
    # Get all strategies across all experiments
    all_strategies = set()
    for exp_data in per_experiment.values():
        all_strategies.update(exp_data["all_strategies"])
    all_strategies = sorted(all_strategies)
    
    # Create matrix: 1 if strategy is in minimal set, 0 otherwise
    matrix = np.zeros((len(experiments), len(all_strategies)))
    
    for i, exp_name in enumerate(experiments):
        exp_data = per_experiment[exp_name]
        minimal_set = set(exp_data["minimal_set"])
        
        for j, strategy in enumerate(all_strategies):
            if strategy in minimal_set:
                matrix[i, j] = 1
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(matrix.T, cmap="YlGnBu", aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(experiments)))
    ax.set_yticks(np.arange(len(all_strategies)))
    ax.set_xticklabels([exp.replace("/", "\n") for exp in experiments], fontsize=8)
    ax.set_yticklabels(all_strategies, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("In Minimal Set", fontsize=12)
    
    # Add grid
    ax.set_xticks(np.arange(len(experiments)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(all_strategies)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Strategy", fontsize=12)
    ax.set_title("Strategy Inclusion in Minimal Sets Across Experiments", fontsize=14, fontweight='bold')
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def plot_strategy_usage(summary: dict, output_path: Path):
    """
    Create bar plot showing how often each strategy achieves best Cohen's d.
    """
    strategy_counts = summary["global_strategy_counts"]
    
    # Sort by count
    sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
    strategies, counts = zip(*sorted_strategies)
    
    # Color by category
    always_needed = set(summary["always_needed"])
    sometimes_needed = set(summary["sometimes_needed"])
    always_redundant = set(summary["always_redundant"])
    
    colors = []
    for strategy in strategies:
        if strategy in always_needed:
            colors.append('#2ecc71')  # Green
        elif strategy in sometimes_needed:
            colors.append('#f39c12')  # Orange
        else:  # always_redundant
            colors.append('#e74c3c')  # Red
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(strategies)), counts, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_ylabel("Times Achieving Best Cohen's d", fontsize=12)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_title("Strategy Performance Across All Experiments", fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label=f'Always Needed ({len(always_needed)})'),
        Patch(facecolor='#f39c12', label=f'Sometimes Needed ({len(sometimes_needed)})'),
        Patch(facecolor='#e74c3c', label=f'Always Redundant ({len(always_redundant)})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved usage plot to {output_path}")


def plot_minimal_set_sizes(per_experiment: dict, output_path: Path):
    """
    Create bar plot showing the size of minimal sets for each experiment.
    """
    experiments = sorted(per_experiment.keys())
    
    minimal_sizes = []
    total_sizes = []
    
    for exp_name in experiments:
        exp_data = per_experiment[exp_name]
        minimal_sizes.append(len(exp_data["minimal_set"]))
        total_sizes.append(len(exp_data["all_strategies"]))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(experiments))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, total_sizes, width, label='All Strategies', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, minimal_sizes, width, label='Minimal Set', color='#2ecc71', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([exp.replace("/", "\n") for exp in experiments], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Number of Strategies", fontsize=12)
    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_title("Minimal Set Size vs All Strategies", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Add reduction percentage on top of minimal bars
    for i, (total, minimal) in enumerate(zip(total_sizes, minimal_sizes)):
        reduction = (total - minimal) / total * 100
        ax.text(i + width/2, minimal + 0.1, f'-{reduction:.0f}%', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved minimal set size plot to {output_path}")


def create_summary_table(summary: dict, per_experiment: dict, output_path: Path):
    """
    Create a text summary table.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("STRATEGY REDUNDANCY ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append(f"Total experiments analyzed: {summary['n_experiments']}")
    lines.append("")
    
    lines.append("STRATEGY CATEGORIES:")
    lines.append("-" * 80)
    lines.append("")
    
    lines.append("Always Needed (in ALL minimal sets):")
    for strategy in sorted(summary['always_needed']):
        count = summary['global_strategy_counts'].get(strategy, 0)
        lines.append(f"  - {strategy:20s} (best for {count} features)")
    lines.append("")
    
    lines.append("Sometimes Needed (in SOME minimal sets):")
    for strategy in sorted(summary['sometimes_needed']):
        count = summary['global_strategy_counts'].get(strategy, 0)
        # Count in how many experiments it's needed
        needed_in = sum(1 for exp_data in per_experiment.values() if strategy in exp_data['minimal_set'])
        lines.append(f"  - {strategy:20s} (best for {count} features, needed in {needed_in}/{summary['n_experiments']} experiments)")
    lines.append("")
    
    lines.append("Always Redundant (NEVER in minimal sets):")
    for strategy in sorted(summary['always_redundant']):
        count = summary['global_strategy_counts'].get(strategy, 0)
        lines.append(f"  - {strategy:20s} (best for {count} features, but always redundant)")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("RECOMMENDATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Use these {len(summary['recommended_minimal_set'])} strategies in your paper:")
    for strategy in sorted(summary['recommended_minimal_set']):
        lines.append(f"  - {strategy}")
    lines.append("")
    lines.append("This ensures all features maintain their classification across all experiments.")
    lines.append("")
    
    # Per-experiment breakdown
    lines.append("=" * 80)
    lines.append("PER-EXPERIMENT BREAKDOWN")
    lines.append("=" * 80)
    lines.append("")
    
    for exp_name in sorted(per_experiment.keys()):
        exp_data = per_experiment[exp_name]
        lines.append(f"{exp_name}:")
        lines.append(f"  Features: {exp_data['n_features']}")
        lines.append(f"  All strategies: {len(exp_data['all_strategies'])}")
        lines.append(f"  Minimal set: {len(exp_data['minimal_set'])} ({', '.join(sorted(exp_data['minimal_set']))})")
        lines.append(f"  Redundant: {len(exp_data['redundant_strategies'])} ({', '.join(sorted(exp_data['redundant_strategies']))})")
        lines.append("")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved summary table to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize strategy redundancy analysis"
    )
    parser.add_argument(
        "--analysis-file",
        type=Path,
        required=True,
        help="Path to minimal_strategies_analysis.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cohens_d/strategy_analysis"),
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    if not args.analysis_file.exists():
        print(f"Error: Analysis file {args.analysis_file} does not exist")
        return
    
    # Load analysis
    with open(args.analysis_file) as f:
        analysis = json.load(f)
    
    summary = analysis["summary"]
    per_experiment = analysis["per_experiment"]
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    plot_strategy_heatmap(
        per_experiment,
        args.output_dir / "strategy_heatmap.png"
    )
    
    plot_strategy_usage(
        summary,
        args.output_dir / "strategy_usage.png"
    )
    
    plot_minimal_set_sizes(
        per_experiment,
        args.output_dir / "minimal_set_sizes.png"
    )
    
    create_summary_table(
        summary,
        per_experiment,
        args.output_dir / "summary.txt"
    )
    
    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

