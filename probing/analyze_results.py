"""
Utility script to analyze and visualize results from probing experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_experiment_results(exp_dir: Path) -> dict:
    """Load results from an experiment directory."""
    results_path = exp_dir / 'results.json'
    config_path = exp_dir / 'config.json'
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found at {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return {'results': results, 'config': config, 'path': exp_dir}


def plot_comparison(experiments: list[dict], save_path: Path = None):
    """
    Plot comparison of multiple experiments.
    
    Args:
        experiments: List of experiment dicts from load_experiment_results()
        save_path: Path to save the plot
    """
    n_exp = len(experiments)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    activation_types = ['original', 'reconstructed', 'residue']
    
    # Prepare data
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        # For each experiment
        x = np.arange(len(activation_types))
        width = 0.8 / n_exp
        
        for exp_idx, exp in enumerate(experiments):
            results = exp['results']
            config = exp['config']
            exp_name = exp['path'].name
            
            values = [results[act_type]['test_metrics'][metric] for act_type in activation_types]
            offset = (exp_idx - n_exp/2 + 0.5) * width
            
            ax.bar(x + offset, values, width, label=exp_name, alpha=0.8)
        
        ax.set_xlabel('Activation Type')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([a.capitalize() for a in activation_types])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_single_experiment(exp: dict, save_path: Path = None):
    """
    Plot detailed results for a single experiment.
    
    Args:
        exp: Experiment dict from load_experiment_results()
        save_path: Path to save the plot
    """
    results = exp['results']
    config = exp['config']
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    activation_types = ['original', 'reconstructed', 'residue']
    
    # Plot 1: Test metrics comparison
    ax1 = fig.add_subplot(gs[0, :])
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    x = np.arange(len(activation_types))
    width = 0.2
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[act]['test_metrics'][metric] for act in activation_types]
        ax1.bar(x + i*width, values, width, label=name, alpha=0.8)
    
    ax1.set_xlabel('Activation Type')
    ax1.set_ylabel('Score')
    ax1.set_title('Test Metrics Comparison')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([a.capitalize() for a in activation_types])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # Plot 2-4: Training curves for each activation type
    for i, act_type in enumerate(activation_types):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        history = results[act_type]['train_history']
        epochs = range(len(history['train_loss']))
        
        ax.plot(epochs, history['train_loss'], label='Train Loss', alpha=0.7)
        ax.plot(epochs, history['val_loss'], label='Val Loss', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(epochs, history['val_acc'], label='Val Acc', color='green', alpha=0.7, linestyle='--')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color='blue')
        ax2.set_ylabel('Accuracy', color='green')
        ax.set_title(f'{act_type.capitalize()} Training')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Plot 5-7: F1 curves for each activation type
    for i, act_type in enumerate(activation_types):
        row = 2
        col = i
        ax = fig.add_subplot(gs[row, col])
        
        history = results[act_type]['train_history']
        epochs = range(len(history['train_f1']))
        
        ax.plot(epochs, history['train_f1'], label='Train F1', alpha=0.7)
        ax.plot(epochs, history['val_f1'], label='Val F1', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{act_type.capitalize()} F1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Add config info as text
    config_text = f"Config: {config.get('model_name', 'N/A')}, Layer {config.get('layer_index', 'N/A')}, {config.get('probe_type', 'N/A')} probe"
    fig.suptitle(f"Probing Experiment Results\n{config_text}", fontsize=14, y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved detailed plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(exp: dict):
    """Print a text summary of experiment results."""
    results = exp['results']
    config = exp['config']
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT SUMMARY: {exp['path'].name}")
    print("=" * 80)
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-" * 80)
    print(f"{'Activation':<15} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    
    for act_type in ['original', 'reconstructed', 'residue']:
        metrics = results[act_type]['test_metrics']
        print(f"{act_type.capitalize():<15} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f}")
    
    # Determine best
    best_act = max(['original', 'reconstructed', 'residue'], 
                   key=lambda x: results[x]['test_metrics']['f1'])
    best_f1 = results[best_act]['test_metrics']['f1']
    
    print("\n" + "=" * 80)
    print(f"Best: {best_act.upper()} (F1 = {best_f1:.4f})")
    print("=" * 80)
    
    # Interpretation
    print("\nInterpretation:")
    if best_act == 'residue':
        print("  → Residue activations perform best")
        print("  → Reasoning information is NOT fully captured by the SAE")
        print("  → Higher-level features remain in reconstruction error")
    elif best_act == 'reconstructed':
        print("  → Reconstructed activations perform best")
        print("  → SAE successfully captures reasoning information")
        print("  → Linear features encode reasoning patterns")
    else:
        print("  → Original activations perform best (expected)")
        
        # Compare recon vs residue
        recon_f1 = results['reconstructed']['test_metrics']['f1']
        resid_f1 = results['residue']['test_metrics']['f1']
        
        if recon_f1 > resid_f1:
            print(f"  → Reconstructed ({recon_f1:.4f}) > Residue ({resid_f1:.4f})")
            print("  → Most information is in SAE features")
        else:
            print(f"  → Residue ({resid_f1:.4f}) > Reconstructed ({recon_f1:.4f})")
            print("  → More information is in reconstruction error")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze probing experiment results')
    
    parser.add_argument('experiment_dirs', nargs='+', type=str,
                       help='Directories containing experiment results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--save_plots', type=str, default=None,
                       help='Directory to save plots')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison plot (for multiple experiments)')
    
    args = parser.parse_args()
    
    # Load experiments
    experiments = []
    for exp_dir in args.experiment_dirs:
        exp_path = Path(exp_dir)
        if not exp_path.exists():
            print(f"Warning: {exp_dir} does not exist, skipping")
            continue
        
        try:
            exp = load_experiment_results(exp_path)
            experiments.append(exp)
            print(f"Loaded: {exp_dir}")
        except Exception as e:
            print(f"Error loading {exp_dir}: {e}")
    
    if not experiments:
        print("No valid experiments found")
        return
    
    # Print summaries
    for exp in experiments:
        print_summary(exp)
    
    # Generate plots
    if args.plot:
        save_dir = Path(args.save_plots) if args.save_plots else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Individual plots
        for exp in experiments:
            save_path = save_dir / f"{exp['path'].name}_detailed.png" if save_dir else None
            plot_single_experiment(exp, save_path)
        
        # Comparison plot
        if args.compare and len(experiments) > 1:
            save_path = save_dir / "comparison.png" if save_dir else None
            plot_comparison(experiments, save_path)


if __name__ == '__main__':
    main()
