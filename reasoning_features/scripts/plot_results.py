"""
Plotting script for reasoning features analysis results.

Generates various plots from the JSON results files:
1. Layer-level statistics (number of features, mean scores, etc.)
2. Distributions of statistics for features on each layer
3. Token dependency metrics across layers
4. Scatter plots of reasoning stats vs token stats

Usage:
    python plot_results.py --results-dir results/initial-setting/gemma-2-2b/s1k
    python plot_results.py --results-dir results/initial-setting --all-experiments
    python plot_results.py --results-dir results/initial-setting/gemma-2-2b/s1k --only-layer-stats
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Suppress expected warnings
warnings.filterwarnings('ignore', category=np.RankWarning)
warnings.filterwarnings('ignore', message='invalid value encountered')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def load_experiment_data(experiment_dir: Path) -> dict:
    """Load all data for an experiment (all layers).
    
    Expects directory structure: results/setting/model/dataset/layerX/
    where experiment_dir is the 'dataset' directory containing layer subdirs.
    """
    data = {
        'layers': [],
        'reasoning_features': {},
        'token_analysis': {},
        'feature_stats': {},
        'steering_results': {},
        'injection_results': {},
        'interpretation_results': {},
    }
    
    # Find all layer directories
    layer_dirs = sorted([d for d in experiment_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('layer')],
                        key=lambda d: int(d.name.replace('layer', '')))
    
    for layer_dir in layer_dirs:
        layer_idx = int(layer_dir.name.replace('layer', ''))
        data['layers'].append(layer_idx)
        
        # Load reasoning features
        rf_path = layer_dir / 'reasoning_features.json'
        if rf_path.exists():
            with open(rf_path) as f:
                data['reasoning_features'][layer_idx] = json.load(f)
        
        # Load token analysis
        ta_path = layer_dir / 'token_analysis.json'
        if ta_path.exists():
            with open(ta_path) as f:
                data['token_analysis'][layer_idx] = json.load(f)
        
        # Load feature stats (can be large, load selectively)
        fs_path = layer_dir / 'feature_stats.json'
        if fs_path.exists():
            with open(fs_path) as f:
                data['feature_stats'][layer_idx] = json.load(f)
        
        # Load steering results (per-feature structure)
        for benchmark in ['aime24', 'gpqa_diamond', 'math500']:
            benchmark_dir = layer_dir / benchmark
            if benchmark_dir.exists():
                summary_path = benchmark_dir / 'experiment_summary.json'
                if summary_path.exists():
                    with open(summary_path) as f:
                        if layer_idx not in data['steering_results']:
                            data['steering_results'][layer_idx] = {}
                        data['steering_results'][layer_idx][benchmark] = json.load(f)
        
        # Load injection results
        injection_path = layer_dir / 'injection_results.json'
        if injection_path.exists():
            with open(injection_path) as f:
                data['injection_results'][layer_idx] = json.load(f)
        
        # Load feature interpretation results
        interp_path = layer_dir / 'feature_interpretations.json'
        if interp_path.exists():
            with open(interp_path) as f:
                data['interpretation_results'][layer_idx] = json.load(f)
    
    return data


def compute_score_components(feature: dict) -> dict:
    """Compute the components of the reasoning score."""
    mean_r = feature.get('mean_reasoning', 0)
    mean_nr = feature.get('mean_nonreasoning', 0)
    
    # Direction
    direction = 1 if mean_r > mean_nr else -1
    
    # AUC contribution
    roc_auc = feature.get('roc_auc', 0.5)
    auc_contrib = abs(roc_auc - 0.5) * 2
    
    # Effect size contribution
    cohens_d = abs(feature.get('cohens_d', 0))
    effect_contrib = min(cohens_d, 3.0) / 3.0
    
    # P-value contribution
    p_value = feature.get('mannwhitney_p', 1.0)
    p_contrib = min(-np.log10(p_value + 1e-300), 50) / 50
    
    # Frequency contribution
    freq_r = feature.get('freq_active_reasoning', 0)
    freq_nr = feature.get('freq_active_nonreasoning', 0)
    freq_ratio = (freq_r + 0.01) / (freq_nr + 0.01)
    freq_contrib = min(np.log2(freq_ratio + 1) / 5, 1.0) if freq_ratio > 1 else 0
    
    return {
        'direction': direction,
        'auc_contrib': auc_contrib,
        'effect_contrib': effect_contrib,
        'p_contrib': p_contrib,
        'freq_contrib': freq_contrib,
        'reasoning_score': feature.get('reasoning_score', 0),
    }


# =============================================================================
# Layer-Level Statistics Plots
# =============================================================================

def plot_layer_feature_counts(data: dict, output_dir: Path):
    """Plot number of reasoning features per layer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = data['layers']
    counts = []
    percentages = []
    
    for layer in layers:
        rf = data['reasoning_features'].get(layer, {})
        summary = rf.get('summary', {})
        counts.append(summary.get('reasoning_features_count', 0))
        percentages.append(summary.get('percentage_reasoning', 0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    ax1.bar(range(len(layers)), counts, color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([f'L{l}' for l in layers])
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Reasoning Features')
    ax1.set_title('Reasoning Features Count by Layer')
    for i, v in enumerate(counts):
        ax1.text(i, v + max(counts)*0.02, str(v), ha='center', fontsize=9)
    
    # Percentage plot
    ax2.bar(range(len(layers)), percentages, color='coral', alpha=0.8)
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([f'L{l}' for l in layers])
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Percentage of Features (%)')
    ax2.set_title('Reasoning Features Percentage by Layer')
    for i, v in enumerate(percentages):
        ax2.text(i, v + max(percentages)*0.02, f'{v:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_feature_counts.png', bbox_inches='tight')
    plt.close()


def plot_layer_mean_statistics(data: dict, output_dir: Path):
    """Plot mean statistics of reasoning features per layer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = data['layers']
    metrics = {
        'mean_auc': [],
        'mean_cohens_d': [],
        'mean_reasoning_score': [],
    }
    
    for layer in layers:
        rf = data['reasoning_features'].get(layer, {})
        summary = rf.get('summary', {})
        features = rf.get('features', [])
        
        metrics['mean_auc'].append(summary.get('mean_auc_reasoning_features', 0))
        metrics['mean_cohens_d'].append(summary.get('mean_cohens_d_reasoning_features', 0))
        
        if features:
            scores = [f.get('reasoning_score', 0) for f in features]
            metrics['mean_reasoning_score'].append(np.mean(scores))
        else:
            metrics['mean_reasoning_score'].append(0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ['Mean ROC-AUC', "Mean Cohen's d", 'Mean Reasoning Score']
    colors = ['steelblue', 'seagreen', 'coral']
    
    for ax, (name, values), title, color in zip(axes, metrics.items(), titles, colors):
        ax.bar(range(len(layers)), values, color=color, alpha=0.8)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Layer')
        
        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.02, f'{v:.3f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_mean_statistics.png', bbox_inches='tight')
    plt.close()


def plot_layer_score_components(data: dict, output_dir: Path):
    """Plot average score components across layers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = data['layers']
    component_names = ['auc_contrib', 'effect_contrib', 'p_contrib', 'freq_contrib']
    
    layer_components = {name: [] for name in component_names}
    
    for layer in layers:
        rf = data['reasoning_features'].get(layer, {})
        features = rf.get('features', [])
        
        if features:
            for name in component_names:
                values = [compute_score_components(f)[name] for f in features]
                layer_components[name].append(np.mean(values))
        else:
            for name in component_names:
                layer_components[name].append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(layers))
    width = 0.2
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    labels = ['AUC Contribution', 'Effect Size Contribution', 
              'P-value Contribution', 'Frequency Contribution']
    
    for i, (name, values) in enumerate(layer_components.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=labels[i], color=colors[i], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Contribution')
    ax.set_title('Reasoning Score Components by Layer')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_score_components.png', bbox_inches='tight')
    plt.close()


# =============================================================================
# Distribution Plots
# =============================================================================

def plot_feature_stat_distributions(data: dict, output_dir: Path):
    """Plot distributions of feature statistics for each layer."""
    dist_dir = output_dir / 'distributions'
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    layers = data['layers']
    
    # Metrics to plot
    metrics = ['roc_auc', 'cohens_d', 'reasoning_score', 'mannwhitney_p']
    metric_labels = ['ROC-AUC', "Cohen's d", 'Reasoning Score', 'Mann-Whitney p-value']
    
    for metric, label in zip(metrics, metric_labels):
        fig, axes = plt.subplots(2, (len(layers) + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()
        
        for ax_idx, layer in enumerate(layers):
            ax = axes[ax_idx]
            
            # Get feature stats
            rf = data['reasoning_features'].get(layer, {})
            features = rf.get('features', [])
            
            if features:
                values = [f.get(metric, 0) for f in features]
                
                if metric == 'mannwhitney_p':
                    # Log transform p-values
                    values = [-np.log10(v + 1e-300) for v in values]
                    label_plot = '-log10(p-value)'
                else:
                    label_plot = label
                
                ax.hist(values, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
                ax.set_xlabel(label_plot)
                ax.set_ylabel('Count')
                ax.set_title(f'Layer {layer}')
                
                # Add mean line
                mean_val = np.mean(values)
                ax.axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.3f}')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Layer {layer}')
        
        # Hide unused axes
        for ax_idx in range(len(layers), len(axes)):
            axes[ax_idx].set_visible(False)
        
        plt.suptitle(f'Distribution of {label} for Reasoning Features', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(dist_dir / f'distribution_{metric}.png', bbox_inches='tight')
        plt.close()


def plot_all_features_distributions(data: dict, output_dir: Path):
    """Plot distributions comparing all features vs reasoning features."""
    dist_dir = output_dir / 'distributions'
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    layers = data['layers']
    
    for layer in layers:
        if layer not in data['feature_stats'] or layer not in data['reasoning_features']:
            continue
        
        all_stats = data['feature_stats'][layer]
        rf = data['reasoning_features'].get(layer, {})
        reasoning_indices = set(rf.get('feature_indices', []))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = [
            ('roc_auc', 'ROC-AUC'),
            ('cohens_d', "Cohen's d"),
            ('reasoning_score', 'Reasoning Score'),
            ('freq_active_reasoning', 'Activation Frequency (Reasoning)')
        ]
        
        for ax, (metric, label) in zip(axes.flatten(), metrics):
            all_values = [f.get(metric, 0) for f in all_stats]
            reasoning_values = [f.get(metric, 0) for f in all_stats 
                               if f['feature_index'] in reasoning_indices]
            
            # Plot histograms
            ax.hist(all_values, bins=50, alpha=0.5, label='All Features', 
                   color='gray', density=True)
            if reasoning_values:
                ax.hist(reasoning_values, bins=30, alpha=0.7, 
                       label='Reasoning Features', color='coral', density=True)
            
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.legend()
            ax.set_title(f'{label} Distribution')
        
        plt.suptitle(f'Layer {layer}: All Features vs Reasoning Features', fontsize=14)
        plt.tight_layout()
        plt.savefig(dist_dir / f'layer{layer}_all_vs_reasoning.png', bbox_inches='tight')
        plt.close()


# =============================================================================
# Token Dependency Plots
# =============================================================================

def plot_token_dependency_by_layer(data: dict, output_dir: Path):
    """Plot token dependency metrics across layers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = data['layers']
    
    metrics = {
        'high_token_dependency_percentage': [],
        'mean_token_concentration': [],
        'mean_normalized_entropy': [],
    }
    
    for layer in layers:
        ta = data['token_analysis'].get(layer, {})
        summary = ta.get('summary', {})
        
        metrics['high_token_dependency_percentage'].append(
            summary.get('high_token_dependency_percentage', 0)
        )
        metrics['mean_token_concentration'].append(
            summary.get('mean_token_concentration', 0)
        )
        metrics['mean_normalized_entropy'].append(
            summary.get('mean_normalized_entropy', 0)
        )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = [
        'High Token Dependency (%)',
        'Mean Token Concentration',
        'Mean Normalized Entropy'
    ]
    colors = ['#C44E52', '#DD8452', '#55A868']
    
    for ax, (name, values), title, color in zip(axes, metrics.items(), titles, colors):
        ax.bar(range(len(layers)), values, color=color, alpha=0.8)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Layer')
        
        for i, v in enumerate(values):
            if v > 0:
                ax.text(i, v + max(values)*0.02, f'{v:.2f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'token_dependency_by_layer.png', bbox_inches='tight')
    plt.close()


def plot_token_concentration_distribution(data: dict, output_dir: Path):
    """Plot distribution of token concentration for each layer."""
    dist_dir = output_dir / 'distributions'
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    layers = data['layers']
    
    fig, axes = plt.subplots(2, (len(layers) + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx]
        
        ta = data['token_analysis'].get(layer, {})
        features = ta.get('features', [])
        
        if features:
            concentrations = [f.get('token_concentration', 0) for f in features]
            entropies = [f.get('normalized_entropy', 0) for f in features]
            
            ax.hist(concentrations, bins=20, alpha=0.7, color='coral', 
                   label='Token Concentration')
            ax.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
            ax.set_xlabel('Token Concentration')
            ax.set_ylabel('Count')
            ax.set_title(f'Layer {layer}')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Layer {layer}')
    
    for ax_idx in range(len(layers), len(axes)):
        axes[ax_idx].set_visible(False)
    
    plt.suptitle('Token Concentration Distribution by Layer', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(dist_dir / 'token_concentration_distribution.png', bbox_inches='tight')
    plt.close()


# =============================================================================
# Scatter Plots: Reasoning Stats vs Token Stats
# =============================================================================

def plot_reasoning_vs_token_scatter(data: dict, output_dir: Path):
    """Plot scatter plots of reasoning stats vs token dependency stats."""
    scatter_dir = output_dir / 'scatter_plots'
    scatter_dir.mkdir(parents=True, exist_ok=True)
    
    layers = data['layers']
    
    # Collect data across all layers
    all_data = []
    
    for layer in layers:
        ta = data['token_analysis'].get(layer, {})
        features = ta.get('features', [])
        
        for f in features:
            feature_stats = f.get('feature_stats', {})
            if feature_stats:
                all_data.append({
                    'layer': layer,
                    'token_concentration': f.get('token_concentration', 0),
                    'normalized_entropy': f.get('normalized_entropy', 0),
                    'roc_auc': feature_stats.get('roc_auc', 0.5),
                    'cohens_d': feature_stats.get('cohens_d', 0),
                    'reasoning_score': feature_stats.get('reasoning_score', 0),
                    **compute_score_components(feature_stats),
                })
    
    if not all_data:
        print("No data for scatter plots")
        return
    
    # Create scatter plots for different combinations
    x_metrics = [
        ('reasoning_score', 'Reasoning Score'),
        ('roc_auc', 'ROC-AUC'),
        ('cohens_d', "Cohen's d"),
        ('auc_contrib', 'AUC Contribution'),
        ('effect_contrib', 'Effect Size Contribution'),
        ('p_contrib', 'P-value Contribution'),
        ('freq_contrib', 'Frequency Contribution'),
    ]
    
    y_metrics = [
        ('token_concentration', 'Token Concentration'),
        ('normalized_entropy', 'Normalized Entropy'),
    ]
    
    for x_metric, x_label in x_metrics:
        for y_metric, y_label in y_metrics:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get unique layers for coloring
            unique_layers = sorted(set(d['layer'] for d in all_data))
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
            layer_colors = dict(zip(unique_layers, colors))
            
            for layer in unique_layers:
                layer_data = [d for d in all_data if d['layer'] == layer]
                x_vals = [d[x_metric] for d in layer_data]
                y_vals = [d[y_metric] for d in layer_data]
                
                ax.scatter(x_vals, y_vals, c=[layer_colors[layer]], 
                          label=f'Layer {layer}', alpha=0.6, s=50)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'{y_label} vs {x_label}')
            ax.legend(loc='best', fontsize=9)
            
            # Add trend line
            x_all = [d[x_metric] for d in all_data]
            y_all = [d[y_metric] for d in all_data]
            
            if len(x_all) > 2:
                z = np.polyfit(x_all, y_all, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(x_all), max(x_all), 100)
                ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label='Trend')
                
                # Compute correlation
                corr = np.corrcoef(x_all, y_all)[0, 1]
                ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(scatter_dir / f'{x_metric}_vs_{y_metric}.png', bbox_inches='tight')
            plt.close()


def plot_per_layer_scatter(data: dict, output_dir: Path):
    """Plot scatter plots for each layer separately."""
    scatter_dir = output_dir / 'scatter_plots' / 'per_layer'
    scatter_dir.mkdir(parents=True, exist_ok=True)
    
    layers = data['layers']
    
    for layer in layers:
        ta = data['token_analysis'].get(layer, {})
        features = ta.get('features', [])
        
        if not features:
            continue
        
        # Collect data for this layer
        layer_data = []
        for f in features:
            feature_stats = f.get('feature_stats', {})
            if feature_stats:
                layer_data.append({
                    'token_concentration': f.get('token_concentration', 0),
                    'normalized_entropy': f.get('normalized_entropy', 0),
                    'reasoning_score': feature_stats.get('reasoning_score', 0),
                    'roc_auc': feature_stats.get('roc_auc', 0.5),
                    'cohens_d': feature_stats.get('cohens_d', 0),
                })
        
        if not layer_data:
            continue
        
        # Create 2x2 scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        pairs = [
            ('reasoning_score', 'token_concentration', 'Reasoning Score', 'Token Concentration'),
            ('roc_auc', 'token_concentration', 'ROC-AUC', 'Token Concentration'),
            ('reasoning_score', 'normalized_entropy', 'Reasoning Score', 'Normalized Entropy'),
            ('cohens_d', 'token_concentration', "Cohen's d", 'Token Concentration'),
        ]
        
        for ax, (x_key, y_key, x_label, y_label) in zip(axes.flatten(), pairs):
            x_vals = [d[x_key] for d in layer_data]
            y_vals = [d[y_key] for d in layer_data]
            
            ax.scatter(x_vals, y_vals, alpha=0.6, s=60, c='steelblue')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            if len(x_vals) > 2:
                corr = np.corrcoef(x_vals, y_vals)[0, 1]
                ax.set_title(f'{y_label} vs {x_label}\n(r = {corr:.3f})')
            else:
                ax.set_title(f'{y_label} vs {x_label}')
        
        plt.suptitle(f'Layer {layer}: Reasoning vs Token Statistics', fontsize=14)
        plt.tight_layout()
        plt.savefig(scatter_dir / f'layer{layer}_scatter.png', bbox_inches='tight')
        plt.close()


# =============================================================================
# Steering Results Plots
# =============================================================================

def plot_steering_results(data: dict, output_dir: Path):
    """Plot steering experiment results across layers (per-feature, gamma-based).
    
    New steering formula: x' = x + γ * f_max * W_dec[i]
    Results are per-feature with gamma values instead of multipliers.
    """
    steering_dir = output_dir / 'steering'
    steering_dir.mkdir(parents=True, exist_ok=True)
    
    if not data['steering_results']:
        print("No steering results to plot")
        return
    
    for benchmark in ['aime24', 'gpqa_diamond', 'math500']:
        layers_with_data = []
        all_gamma_values = set()
        all_feature_indices = set()
        results_by_layer = {}
        
        for layer, benchmarks in data['steering_results'].items():
            if benchmark in benchmarks:
                layers_with_data.append(layer)
                steering_data = benchmarks[benchmark]
                
                # New format: per_feature_results
                per_feature = steering_data.get('per_feature_results', {})
                if per_feature:
                    results_by_layer[layer] = per_feature
                    for feat_idx, gamma_results in per_feature.items():
                        all_feature_indices.add(feat_idx)
                        all_gamma_values.update(gamma_results.keys())
        
        if not layers_with_data:
            continue
        
        gamma_values = sorted([float(g) for g in all_gamma_values])
        feature_indices = sorted(all_feature_indices, key=lambda x: int(x))
        
        if not gamma_values or not feature_indices:
            print(f"  No per-feature steering results for {benchmark}")
            continue
        
        # Plot 1: Average accuracy vs gamma across all features
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(layers_with_data))
        width = 0.15
        
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(gamma_values)))
        
        for i, gamma in enumerate(gamma_values):
            avg_accuracies = []
            for layer in layers_with_data:
                per_feature = results_by_layer.get(layer, {})
                layer_accs = []
                for feat_idx in feature_indices:
                    if feat_idx in per_feature:
                        acc = per_feature[feat_idx].get(str(gamma), {}).get('accuracy', 0)
                        layer_accs.append(acc)
                avg_acc = np.mean(layer_accs) if layer_accs else 0
                avg_accuracies.append(avg_acc)
            
            offset = (i - len(gamma_values)/2 + 0.5) * width
            ax.bar(x + offset, avg_accuracies, width, label=f'γ={gamma}', 
                  color=colors[i], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'Layer {l}' for l in layers_with_data])
        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Accuracy')
        ax.set_title(f'{benchmark.upper()}: Steering Results by Layer and Gamma (Avg across features)')
        ax.legend(loc='best')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(steering_dir / f'{benchmark}_steering_by_gamma.png', bbox_inches='tight')
        plt.close()
        
        # Plot 2: Per-feature delta from baseline for each layer
        for layer in layers_with_data:
            per_feature = results_by_layer.get(layer, {})
            if not per_feature:
                continue
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            feat_list = sorted(per_feature.keys(), key=lambda x: int(x))
            x = np.arange(len(feat_list))
            width = 0.15
            
            # Compute baseline (gamma=0) for each feature
            baseline_accs = {}
            for feat_idx in feat_list:
                baseline_accs[feat_idx] = per_feature[feat_idx].get('0.0', per_feature[feat_idx].get('0', {})).get('accuracy', 0)
            
            # Plot delta for each non-zero gamma
            nonzero_gammas = [g for g in gamma_values if g != 0.0]
            colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(nonzero_gammas)))
            
            for i, gamma in enumerate(nonzero_gammas):
                deltas = []
                for feat_idx in feat_list:
                    acc = per_feature[feat_idx].get(str(gamma), {}).get('accuracy', 0)
                    baseline = baseline_accs[feat_idx]
                    deltas.append(acc - baseline)
                
                offset = (i - len(nonzero_gammas)/2 + 0.5) * width
                ax.bar(x + offset, deltas, width, label=f'γ={gamma}', 
                      color=colors[i], alpha=0.8)
            
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f'F{f}' for f in feat_list], rotation=45, fontsize=8)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Accuracy Delta (vs γ=0)')
            ax.set_title(f'{benchmark.upper()} Layer {layer}: Per-Feature Steering Effect')
            ax.legend(loc='best', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(steering_dir / f'{benchmark}_layer{layer}_per_feature.png', bbox_inches='tight')
            plt.close()
        
        # Plot 3: Summary - number of features improved/degraded by positive gamma
        if len(layers_with_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            improved_counts = []
            degraded_counts = []
            neutral_counts = []
            
            for layer in layers_with_data:
                per_feature = results_by_layer.get(layer, {})
                n_improved = 0
                n_degraded = 0
                n_neutral = 0
                
                for feat_idx, gamma_results in per_feature.items():
                    baseline = gamma_results.get('0.0', gamma_results.get('0', {})).get('accuracy', 0)
                    
                    # Check positive gamma values
                    for gamma in [g for g in gamma_values if g > 0]:
                        acc = gamma_results.get(str(gamma), {}).get('accuracy', 0)
                        delta = acc - baseline
                        if delta > 0.02:
                            n_improved += 1
                            break
                        elif delta < -0.02:
                            n_degraded += 1
                            break
                    else:
                        n_neutral += 1
                
                improved_counts.append(n_improved)
                degraded_counts.append(n_degraded)
                neutral_counts.append(n_neutral)
            
            x = np.arange(len(layers_with_data))
            width = 0.25
            
            ax.bar(x - width, improved_counts, width, label='Improved (>2%)', color='#55A868', alpha=0.8)
            ax.bar(x, neutral_counts, width, label='Neutral', color='#8C8C8C', alpha=0.8)
            ax.bar(x + width, degraded_counts, width, label='Degraded (<-2%)', color='#C44E52', alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels([f'Layer {l}' for l in layers_with_data])
            ax.set_xlabel('Layer')
            ax.set_ylabel('Number of Features')
            ax.set_title(f'{benchmark.upper()}: Feature Response to Positive γ Steering')
            ax.legend(loc='best')
            
            plt.tight_layout()
            plt.savefig(steering_dir / f'{benchmark}_steering_summary.png', bbox_inches='tight')
            plt.close()


# =============================================================================
# Token Injection Plots
# =============================================================================

def plot_injection_summary(data: dict, output_dir: Path):
    """Plot token injection experiment summary."""
    injection_dir = output_dir / 'injection'
    injection_dir.mkdir(parents=True, exist_ok=True)
    
    if not data.get('injection_results'):
        print("  No injection results found")
        return
    
    layers = sorted(data['injection_results'].keys())
    
    # Collect summary statistics
    metrics = {
        'avg_cohens_d': [],
        'pct_token_driven': [],
        'pct_partially_driven': [],
        'pct_weakly_driven': [],
        'pct_context_dependent': [],
        'avg_baseline_activation': [],
        'avg_reasoning_activation': [],
    }
    
    for layer in layers:
        inj = data['injection_results'].get(layer, {})
        summary = inj.get('summary', {})
        counts = summary.get('classification_counts', {})
        n_features = summary.get('n_features', 1)
        
        # Backwards compatible: compute avg_cohens_d from features if not in summary
        metrics['avg_cohens_d'].append(summary.get('avg_cohens_d', 0))
        metrics['avg_baseline_activation'].append(summary.get('avg_baseline_activation', 0))
        metrics['avg_reasoning_activation'].append(summary.get('avg_reasoning_activation', 0))
        
        pct_token = 100 * counts.get('token_driven', 0) / n_features if n_features > 0 else 0
        pct_partial = 100 * counts.get('partially_token_driven', 0) / n_features if n_features > 0 else 0
        pct_weak = 100 * counts.get('weakly_token_driven', 0) / n_features if n_features > 0 else 0
        pct_context = 100 * counts.get('context_dependent', 0) / n_features if n_features > 0 else 0
        metrics['pct_token_driven'].append(pct_token)
        metrics['pct_partially_driven'].append(pct_partial)
        metrics['pct_weakly_driven'].append(pct_weak)
        metrics['pct_context_dependent'].append(pct_context)
    
    # Plot 1: Cohen's d and token-driven percentages
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Cohen's d (effect size)
    ax = axes[0]
    ax.bar(range(len(layers)), metrics['avg_cohens_d'], color='#C44E52', alpha=0.8)
    ax.axhline(0.8, color='black', linestyle='--', alpha=0.5, label='Large effect (d=0.8)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect (d=0.5)')
    ax.axhline(0.2, color='gray', linestyle='-.', alpha=0.5, label='Small effect (d=0.2)')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel("Average Cohen's d")
    ax.set_title('Token Injection Effect Size')
    ax.legend(fontsize=7)
    for i, v in enumerate(metrics['avg_cohens_d']):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # Percentage by classification
    ax = axes[1]
    x = np.arange(len(layers))
    width = 0.2
    ax.bar(x - 1.5*width, metrics['pct_token_driven'], width, label='Token-driven', color='#C44E52', alpha=0.8)
    ax.bar(x - 0.5*width, metrics['pct_partially_driven'], width, label='Partially token-driven', color='#DD8452', alpha=0.8)
    ax.bar(x + 0.5*width, metrics['pct_weakly_driven'], width, label='Weakly token-driven', color='#55A868', alpha=0.8)
    ax.bar(x + 1.5*width, metrics['pct_context_dependent'], width, label='Context dependent', color='#4C72B0', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Percentage of Features (%)')
    ax.set_title('Feature Classification by Token Dependency')
    ax.legend(fontsize=7, loc='upper right')
    
    # Activation comparison
    ax = axes[2]
    width_act = 0.35
    ax.bar(x - width_act/2, metrics['avg_baseline_activation'], width_act, label='Baseline', color='#4C72B0', alpha=0.8)
    ax.bar(x + width_act/2, metrics['avg_reasoning_activation'], width_act, label='Reasoning', color='#55A868', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Baseline vs Reasoning Activation')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(injection_dir / 'injection_summary.png', bbox_inches='tight')
    plt.close()


def plot_injection_per_feature(data: dict, output_dir: Path):
    """Plot injection results for each feature."""
    injection_dir = output_dir / 'injection'
    injection_dir.mkdir(parents=True, exist_ok=True)
    
    if not data.get('injection_results'):
        return
    
    for layer, inj_data in sorted(data['injection_results'].items()):
        features = inj_data.get('features', [])
        if not features:
            continue
        
        # Transfer ratios by strategy - get strategies from config
        config = inj_data.get('config', {})
        strategies = config.get('strategies', ['prepend', 'intersperse', 'replace'])
        
        # Color palette for all strategies
        strategy_colors = {
            'prepend': '#C44E52', 'append': '#DD8452', 'intersperse': '#4C72B0',
            'replace': '#55A868', 'inject_bigram': '#9370DB', 'inject_trigram': '#6A5ACD',
            'bigram_before': '#8172B3', 'bigram_after': '#CCB974',
            'trigram': '#64B5CD', 'comma_list': '#DA8BC3', 'active_trigram': '#8C8C8C',
        }
        
        # Create subplot grid based on number of strategies
        n_strategies = len(strategies)
        n_cols = min(3, n_strategies)
        n_rows = (n_strategies + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for ax_idx, strategy in enumerate(strategies):
            if ax_idx >= len(axes):
                break
            ax = axes[ax_idx]
            color = strategy_colors.get(strategy, '#808080')
            
            cohens_d_values = []
            feature_indices = []
            
            for f in features:
                strat_data = f.get('strategies', {}).get(strategy, {})
                cohens_d_values.append(strat_data.get('cohens_d', 0))
                feature_indices.append(f.get('feature_index', 0))
            
            ax.bar(range(len(cohens_d_values)), cohens_d_values, color=color, alpha=0.8)
            ax.axhline(0.8, color='black', linestyle='--', alpha=0.5, label='Large (d=0.8)')
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Medium (d=0.5)')
            ax.axhline(0.2, color='gray', linestyle='-.', alpha=0.5, label='Small (d=0.2)')
            ax.set_xticks(range(len(feature_indices)))
            ax.set_xticklabels([str(idx) for idx in feature_indices], rotation=45, fontsize=8)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel("Cohen's d")
            ax.set_title(f'Strategy: {strategy}')
            ax.legend(fontsize=7)
        
        # Hide unused subplots
        for ax_idx in range(len(strategies), len(axes)):
            axes[ax_idx].set_visible(False)
        
        plt.suptitle(f"Layer {layer}: Cohen's d by Injection Strategy", fontsize=14)
        plt.tight_layout()
        plt.savefig(injection_dir / f'layer{layer}_injection_strategies.png', bbox_inches='tight')
        plt.close()
        
        # Classification pie chart and distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Classification counts
        classifications = [f.get('classification', 'unknown') for f in features]
        class_counts = {}
        for c in classifications:
            class_counts[c] = class_counts.get(c, 0) + 1
        
        colors_pie = {
            'token_driven': '#C44E52', 
            'partially_token_driven': '#DD8452',
            'weakly_token_driven': '#55A868',
            'context_dependent': '#4C72B0',
        }
        
        ax = axes[0]
        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        pie_colors = [colors_pie.get(l, 'gray') for l in labels]
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Feature Classification Distribution')
        
        # Best Cohen's d distribution
        ax = axes[1]
        best_d_values = [f.get('best_cohens_d', 0) for f in features]
        ax.hist(best_d_values, bins=15, color='#C44E52', alpha=0.7, edgecolor='white')
        ax.axvline(0.8, color='black', linestyle='--', label='Large (d=0.8)')
        ax.axvline(0.5, color='gray', linestyle=':', label='Medium (d=0.5)')
        ax.axvline(0.2, color='gray', linestyle='-.', label='Small (d=0.2)')
        if best_d_values:
            ax.axvline(np.mean(best_d_values), color='red', linestyle='-', 
                       label=f'Mean: {np.mean(best_d_values):.2f}')
        ax.set_xlabel("Best Cohen's d")
        ax.set_ylabel('Count')
        ax.set_title("Distribution of Best Cohen's d")
        ax.legend(fontsize=8)
        
        plt.suptitle(f'Layer {layer}: Token Injection Results', fontsize=14)
        plt.tight_layout()
        plt.savefig(injection_dir / f'layer{layer}_injection_distribution.png', bbox_inches='tight')
        plt.close()


def plot_injection_activation_comparison(data: dict, output_dir: Path):
    """Plot activation comparison between baseline, injected, and reasoning."""
    injection_dir = output_dir / 'injection'
    injection_dir.mkdir(parents=True, exist_ok=True)
    
    if not data.get('injection_results'):
        return
    
    for layer, inj_data in sorted(data['injection_results'].items()):
        features = inj_data.get('features', [])
        if not features:
            continue
        
        # Activation comparison bar chart
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(features))
        width = 0.25
        
        baseline_means = [f.get('baseline_mean', 0) for f in features]
        reasoning_means = [f.get('reasoning_mean', 0) for f in features]
        # Use the best strategy (highest Cohen's d) for injected result
        injected_means = []
        for f in features:
            strategies = f.get('strategies', {})
            best_mean = 0
            for strat_data in strategies.values():
                if strat_data.get('injected_mean', 0) > best_mean:
                    best_mean = strat_data.get('injected_mean', 0)
            injected_means.append(best_mean if best_mean > 0 else f.get('strategies', {}).get('prepend', {}).get('injected_mean', 0))
        feature_indices = [f.get('feature_index', 0) for f in features]
        
        ax.bar(x - width, baseline_means, width, label='Non-reasoning (baseline)', color='#4C72B0', alpha=0.8)
        ax.bar(x, injected_means, width, label='Non-reasoning + injected tokens', color='#DD8452', alpha=0.8)
        ax.bar(x + width, reasoning_means, width, label='Reasoning text', color='#55A868', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(idx) for idx in feature_indices], rotation=45)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Mean Activation')
        ax.set_title(f'Layer {layer}: Activation Comparison')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(injection_dir / f'layer{layer}_activation_comparison.png', bbox_inches='tight')
        plt.close()


# =============================================================================
# Feature Interpretation Plots
# =============================================================================

def plot_interpretation_summary(data: dict, output_dir: Path):
    """Plot LLM feature interpretation results summary."""
    interp_dir = output_dir / 'interpretation'
    interp_dir.mkdir(parents=True, exist_ok=True)
    
    if not data.get('interpretation_results'):
        print("  No interpretation results found")
        return
    
    layers = sorted(data['interpretation_results'].keys())
    
    # Collect summary statistics per layer
    metrics = {
        'genuine_reasoning': [],
        'non_reasoning': [],
        'high_confidence': [],
        'medium_confidence': [],
        'low_confidence': [],
    }
    
    for layer in layers:
        interp = data['interpretation_results'].get(layer, {})
        summary = interp.get('summary', {})
        
        metrics['genuine_reasoning'].append(summary.get('genuine_reasoning_features', 0))
        metrics['non_reasoning'].append(summary.get('non_reasoning_features', 0))
        metrics['high_confidence'].append(summary.get('high_confidence', 0))
        metrics['medium_confidence'].append(summary.get('medium_confidence', 0))
        metrics['low_confidence'].append(summary.get('low_confidence', 0))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Genuine vs Non-Reasoning Features
    ax = axes[0]
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, metrics['genuine_reasoning'], width, 
           label='Genuine Reasoning', color='#55A868', alpha=0.8)
    ax.bar(x + width/2, metrics['non_reasoning'], width,
           label='Non-Reasoning (Confound)', color='#C44E52', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Features')
    ax.set_title('LLM Feature Classification: Genuine vs Confound')
    ax.legend()
    
    # Add value labels
    for i, (g, n) in enumerate(zip(metrics['genuine_reasoning'], metrics['non_reasoning'])):
        if g > 0:
            ax.text(i - width/2, g + 0.1, str(g), ha='center', fontsize=9)
        if n > 0:
            ax.text(i + width/2, n + 0.1, str(n), ha='center', fontsize=9)
    
    # Plot 2: Confidence Distribution
    ax = axes[1]
    width = 0.25
    
    ax.bar(x - width, metrics['high_confidence'], width,
           label='HIGH', color='#55A868', alpha=0.8)
    ax.bar(x, metrics['medium_confidence'], width,
           label='MEDIUM', color='#DD8452', alpha=0.8)
    ax.bar(x + width, metrics['low_confidence'], width,
           label='LOW', color='#C44E52', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Features')
    ax.set_title('LLM Interpretation Confidence')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(interp_dir / 'interpretation_summary.png', bbox_inches='tight')
    plt.close()


def plot_interpretation_per_layer(data: dict, output_dir: Path):
    """Plot interpretation results for each layer."""
    interp_dir = output_dir / 'interpretation'
    interp_dir.mkdir(parents=True, exist_ok=True)
    
    if not data.get('interpretation_results'):
        return
    
    for layer, interp_data in sorted(data['interpretation_results'].items()):
        features = interp_data.get('features', [])
        if not features:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Classification pie chart
        ax = axes[0]
        genuine_count = sum(1 for f in features if f.get('is_genuine_reasoning_feature'))
        confound_count = len(features) - genuine_count
        
        if genuine_count > 0 or confound_count > 0:
            labels = ['Genuine Reasoning', 'Confound']
            sizes = [genuine_count, confound_count]
            colors = ['#55A868', '#C44E52']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   startangle=90, explode=(0.02, 0))
            ax.set_title('Feature Classification')
        
        # Confidence pie chart
        ax = axes[1]
        conf_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for f in features:
            conf = f.get('confidence', 'LOW')
            if conf in conf_counts:
                conf_counts[conf] += 1
        
        labels = list(conf_counts.keys())
        sizes = list(conf_counts.values())
        colors = ['#55A868', '#DD8452', '#C44E52']
        
        # Only plot if there's data
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Interpretation Confidence')
        
        plt.suptitle(f'Layer {layer}: LLM Feature Interpretation Results', fontsize=14)
        plt.tight_layout()
        plt.savefig(interp_dir / f'layer{layer}_interpretation.png', bbox_inches='tight')
        plt.close()


# =============================================================================
# Summary Plot
# =============================================================================

def plot_summary(data: dict, output_dir: Path, experiment_name: str):
    """Create a summary plot with key metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = data['layers']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Feature counts
    ax = axes[0, 0]
    counts = [data['reasoning_features'].get(l, {}).get('summary', {}).get('reasoning_features_count', 0) 
              for l in layers]
    ax.bar(range(len(layers)), counts, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Count')
    ax.set_title('Reasoning Features Count')
    
    # 2. Mean AUC
    ax = axes[0, 1]
    aucs = [data['reasoning_features'].get(l, {}).get('summary', {}).get('mean_auc_reasoning_features', 0) 
            for l in layers]
    ax.bar(range(len(layers)), aucs, color='seagreen', alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean AUC')
    ax.set_title('Mean ROC-AUC')
    
    # 3. Token dependency
    ax = axes[0, 2]
    deps = [data['token_analysis'].get(l, {}).get('summary', {}).get('high_token_dependency_percentage', 0) 
            for l in layers]
    ax.bar(range(len(layers)), deps, color='coral', alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('High Token Dependency %')
    
    # 4. Mean Cohen's d
    ax = axes[1, 0]
    cohens = [data['reasoning_features'].get(l, {}).get('summary', {}).get('mean_cohens_d_reasoning_features', 0) 
              for l in layers]
    ax.bar(range(len(layers)), cohens, color='purple', alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel("Mean Cohen's d")
    ax.set_title("Mean Effect Size (Cohen's d)")
    
    # 5. Mean token concentration
    ax = axes[1, 1]
    concs = [data['token_analysis'].get(l, {}).get('summary', {}).get('mean_token_concentration', 0) 
             for l in layers]
    ax.bar(range(len(layers)), concs, color='darkorange', alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Concentration')
    ax.set_title('Mean Token Concentration')
    
    # 6. Mean normalized entropy
    ax = axes[1, 2]
    ents = [data['token_analysis'].get(l, {}).get('summary', {}).get('mean_normalized_entropy', 0) 
            for l in layers]
    ax.bar(range(len(layers)), ents, color='teal', alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Mean Normalized Entropy')
    
    plt.suptitle(f'Summary: {experiment_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Functions
# =============================================================================

def process_experiment(experiment_dir: Path, plots_dir: Path, 
                       plot_layer_stats: bool = True,
                       plot_distributions: bool = True,
                       plot_token: bool = True,
                       plot_scatter: bool = True,
                       plot_steering: bool = True,
                       plot_injection_: bool = True,
                       plot_interpretation_: bool = True,
                       plot_summary_: bool = True):
    """Process a single experiment and generate all plots.
    
    Expects directory structure: results/setting/model/dataset/layerX/
    """
    print(f"\nProcessing: {experiment_dir}")
    
    # Load data
    data = load_experiment_data(experiment_dir)
    
    if not data['layers']:
        print(f"  No layer data found in {experiment_dir}")
        return
    
    print(f"  Found {len(data['layers'])} layers: {data['layers']}")
    
    # Create output directory
    output_dir = plots_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_name = str(experiment_dir.relative_to(experiment_dir.parent.parent))
    
    # Generate plots based on options
    if plot_layer_stats:
        print("  Generating layer statistics plots...")
        plot_layer_feature_counts(data, output_dir)
        plot_layer_mean_statistics(data, output_dir)
        plot_layer_score_components(data, output_dir)
    
    if plot_distributions:
        print("  Generating distribution plots...")
        plot_feature_stat_distributions(data, output_dir)
        if data['feature_stats']:
            plot_all_features_distributions(data, output_dir)
    
    if plot_token:
        print("  Generating token dependency plots...")
        plot_token_dependency_by_layer(data, output_dir)
        plot_token_concentration_distribution(data, output_dir)
    
    if plot_scatter:
        print("  Generating scatter plots...")
        plot_reasoning_vs_token_scatter(data, output_dir)
        plot_per_layer_scatter(data, output_dir)
    
    if plot_steering:
        print("  Generating steering result plots...")
        plot_steering_results(data, output_dir)
    
    if plot_injection_:
        if data.get('injection_results'):
            print("  Generating injection experiment plots...")
            plot_injection_summary(data, output_dir)
            plot_injection_per_feature(data, output_dir)
            plot_injection_activation_comparison(data, output_dir)
    
    if plot_interpretation_:
        if data.get('interpretation_results'):
            print("  Generating interpretation plots...")
            plot_interpretation_summary(data, output_dir)
            plot_interpretation_per_layer(data, output_dir)
    
    if plot_summary_:
        print("  Generating summary plot...")
        plot_summary(data, output_dir, experiment_name)
    
    print(f"  Saved plots to {output_dir}")


def find_experiments(results_dir: Path) -> list[Path]:
    """Find all experiment directories containing layer data."""
    experiments = []
    
    # Check if this directory contains layer subdirs
    layer_dirs = [d for d in results_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('layer')]
    if layer_dirs:
        experiments.append(results_dir)
        return experiments
    
    # Recursively search
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            experiments.extend(find_experiments(subdir))
    
    return experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate plots from reasoning features analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--results-dir',
        type=Path,
        required=True,
        help='Path to results directory',
    )
    parser.add_argument(
        '--plots-dir',
        type=Path,
        default=Path('plots'),
        help='Base directory for output plots (default: plots/)',
    )
    parser.add_argument(
        '--all-experiments',
        action='store_true',
        help='Process all experiments found under results-dir',
    )
    
    # Plot category options
    parser.add_argument(
        '--only-layer-stats',
        action='store_true',
        help='Only generate layer-level statistics plots',
    )
    parser.add_argument(
        '--only-distributions',
        action='store_true',
        help='Only generate distribution plots',
    )
    parser.add_argument(
        '--only-token',
        action='store_true',
        help='Only generate token dependency plots',
    )
    parser.add_argument(
        '--only-scatter',
        action='store_true',
        help='Only generate scatter plots',
    )
    parser.add_argument(
        '--only-steering',
        action='store_true',
        help='Only generate steering result plots',
    )
    parser.add_argument(
        '--only-injection',
        action='store_true',
        help='Only generate injection experiment plots',
    )
    parser.add_argument(
        '--only-interpretation',
        action='store_true',
        help='Only generate LLM interpretation plots',
    )
    parser.add_argument(
        '--only-summary',
        action='store_true',
        help='Only generate summary plot',
    )
    
    # Exclude options
    parser.add_argument(
        '--no-layer-stats',
        action='store_true',
        help='Skip layer-level statistics plots',
    )
    parser.add_argument(
        '--no-distributions',
        action='store_true',
        help='Skip distribution plots',
    )
    parser.add_argument(
        '--no-token',
        action='store_true',
        help='Skip token dependency plots',
    )
    parser.add_argument(
        '--no-scatter',
        action='store_true',
        help='Skip scatter plots',
    )
    parser.add_argument(
        '--no-steering',
        action='store_true',
        help='Skip steering result plots',
    )
    parser.add_argument(
        '--no-injection',
        action='store_true',
        help='Skip injection experiment plots',
    )
    parser.add_argument(
        '--no-interpretation',
        action='store_true',
        help='Skip LLM interpretation plots',
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip summary plot',
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("REASONING FEATURES PLOTTING")
    print("=" * 60)
    
    # Determine which plots to generate
    only_flags = [args.only_layer_stats, args.only_distributions, args.only_token,
                  args.only_scatter, args.only_steering,
                  args.only_injection, args.only_interpretation, args.only_summary]
    
    if any(only_flags):
        # Only specified categories
        plot_layer_stats = args.only_layer_stats
        plot_distributions = args.only_distributions
        plot_token = args.only_token
        plot_scatter = args.only_scatter
        plot_steering = args.only_steering
        plot_injection = args.only_injection
        plot_interpretation = args.only_interpretation
        plot_summary = args.only_summary
    else:
        # All categories except excluded
        plot_layer_stats = not args.no_layer_stats
        plot_distributions = not args.no_distributions
        plot_token = not args.no_token
        plot_scatter = not args.no_scatter
        plot_steering = not args.no_steering
        plot_injection = not args.no_injection
        plot_interpretation = not args.no_interpretation
        plot_summary = not args.no_summary
    
    print(f"\nPlot categories enabled:")
    print(f"  Layer stats: {plot_layer_stats}")
    print(f"  Distributions: {plot_distributions}")
    print(f"  Token dependency: {plot_token}")
    print(f"  Scatter plots: {plot_scatter}")
    print(f"  Steering results: {plot_steering}")
    print(f"  Injection: {plot_injection}")
    print(f"  Interpretation: {plot_interpretation}")
    print(f"  Summary: {plot_summary}")
    
    # Find experiments
    if args.all_experiments:
        experiments = find_experiments(args.results_dir)
        print(f"\nFound {len(experiments)} experiments")
    else:
        experiments = [args.results_dir]
    
    # Process each experiment
    for exp_dir in experiments:
        # Compute relative path for output
        if args.all_experiments:
            try:
                rel_path = exp_dir.relative_to(args.results_dir)
            except ValueError:
                rel_path = exp_dir.name
            plots_dir = args.plots_dir / rel_path
        else:
            # Single experiment - use plots_dir directly
            plots_dir = args.plots_dir
        
        process_experiment(
            exp_dir, 
            plots_dir,
            plot_layer_stats=plot_layer_stats,
            plot_distributions=plot_distributions,
            plot_token=plot_token,
            plot_scatter=plot_scatter,
            plot_steering=plot_steering,
            plot_injection_=plot_injection,
            plot_interpretation_=plot_interpretation,
            plot_summary_=plot_summary,
        )
    
    print("\n" + "=" * 60)
    print("PLOTTING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
