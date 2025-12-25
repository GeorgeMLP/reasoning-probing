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
        
        # Load steering results
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
    """Plot steering experiment results across layers."""
    steering_dir = output_dir / 'steering'
    steering_dir.mkdir(parents=True, exist_ok=True)
    
    if not data['steering_results']:
        print("No steering results to plot")
        return
    
    for benchmark in ['aime24', 'gpqa_diamond', 'math500']:
        layers_with_data = []
        multipliers = set()
        results_by_layer = {}
        
        for layer, benchmarks in data['steering_results'].items():
            if benchmark in benchmarks:
                layers_with_data.append(layer)
                results = benchmarks[benchmark].get('results', {})
                results_by_layer[layer] = results
                multipliers.update(results.keys())
        
        if not layers_with_data:
            continue
        
        multipliers = sorted([float(m) for m in multipliers])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(layers_with_data))
        width = 0.15
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(multipliers)))
        
        for i, mult in enumerate(multipliers):
            accuracies = []
            for layer in layers_with_data:
                acc = results_by_layer[layer].get(str(mult), {}).get('accuracy', 0)
                accuracies.append(acc)
            
            offset = (i - len(multipliers)/2 + 0.5) * width
            ax.bar(x + offset, accuracies, width, label=f'mult={mult}', 
                  color=colors[i], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'Layer {l}' for l in layers_with_data])
        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{benchmark.upper()}: Steering Results by Layer and Multiplier')
        ax.legend(loc='best')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(steering_dir / f'{benchmark}_steering_results.png', bbox_inches='tight')
        plt.close()


# =============================================================================
# ANOVA Plots
# =============================================================================

def load_anova_data(experiment_dir: Path) -> Optional[dict]:
    """Load ANOVA results from experiment directory.
    
    Expects directory structure: results/setting/model/layerX/anova_results.json
    """
    anova_data = {}
    
    for layer_dir in experiment_dir.iterdir():
        if not layer_dir.is_dir() or not layer_dir.name.startswith('layer'):
            continue
        
        layer_idx = int(layer_dir.name.replace('layer', ''))
        results_path = layer_dir / 'anova_results.json'
        
        if results_path.exists():
            with open(results_path) as f:
                anova_data[layer_idx] = json.load(f)
    
    return anova_data if anova_data else None


def plot_anova_summary(anova_data: dict, output_dir: Path):
    """Plot ANOVA summary statistics across layers."""
    anova_dir = output_dir / 'anova'
    anova_dir.mkdir(parents=True, exist_ok=True)
    
    layers = sorted(anova_data.keys())
    
    # Extract summary statistics
    metrics = {
        'pct_token_dominated': [],
        'pct_context_dominated': [],
        'mean_eta_sq_token': [],
        'mean_eta_sq_context': [],
        'mean_eta_sq_interaction': [],
    }
    
    for layer in layers:
        summary = anova_data[layer].get('summary', {})
        for key in metrics:
            metrics[key].append(summary.get(key, 0))
    
    # Plot 1: Token vs Context Dominated Percentages
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, metrics['pct_token_dominated'], width, 
                   label='Token-Dominated', color='#C44E52', alpha=0.8)
    bars2 = ax.bar(x + width/2, metrics['pct_context_dominated'], width,
                   label='Context-Dominated', color='#55A868', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Percentage of Features (%)')
    ax.set_title('Token vs. Context Dominated Features by Layer')
    ax.legend()
    ax.set_ylim(0, 100)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(anova_dir / 'token_vs_context_dominated.png', bbox_inches='tight')
    plt.close()
    
    # Plot 2: Mean Eta-Squared by Layer
    fig, ax = plt.subplots(figsize=(12, 6))
    
    width = 0.25
    
    bars1 = ax.bar(x - width, metrics['mean_eta_sq_token'], width,
                   label='η² Token', color='#C44E52', alpha=0.8)
    bars2 = ax.bar(x, metrics['mean_eta_sq_context'], width,
                   label='η² Context', color='#55A868', alpha=0.8)
    bars3 = ax.bar(x + width, metrics['mean_eta_sq_interaction'], width,
                   label='η² Interaction', color='#4C72B0', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean η² (Variance Explained)')
    ax.set_title('Mean Eta-Squared by Factor and Layer')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(anova_dir / 'mean_eta_squared_by_layer.png', bbox_inches='tight')
    plt.close()


def plot_anova_distributions(anova_data: dict, output_dir: Path):
    """Plot distributions of ANOVA statistics."""
    dist_dir = output_dir / 'anova' / 'distributions'
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    layers = sorted(anova_data.keys())
    
    # Collect all eta-squared values
    all_eta_token = []
    all_eta_context = []
    all_dominant = []
    
    for layer in layers:
        features = anova_data[layer].get('features', [])
        for f in features:
            all_eta_token.append(f.get('eta_sq_token', 0))
            all_eta_context.append(f.get('eta_sq_context', 0))
            all_dominant.append(f.get('dominant_factor', 'none'))
    
    if not all_eta_token:
        return
    
    # Plot 1: Scatter plot of η²_token vs η²_context
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'token': '#C44E52', 'context': '#55A868', 
              'interaction': '#4C72B0', 'mixed': '#DD8452', 'none': 'gray'}
    
    for dom_type in ['token', 'context', 'interaction', 'mixed', 'none']:
        mask = [d == dom_type for d in all_dominant]
        x_vals = [all_eta_token[i] for i in range(len(mask)) if mask[i]]
        y_vals = [all_eta_context[i] for i in range(len(mask)) if mask[i]]
        if x_vals:
            ax.scatter(x_vals, y_vals, c=colors.get(dom_type, 'gray'),
                       label=f'{dom_type.capitalize()} ({len(x_vals)})', alpha=0.6, s=50)
    
    # Add diagonal line
    max_val = max(max(all_eta_token), max(all_eta_context))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal effect')
    
    ax.set_xlabel('η² Token')
    ax.set_ylabel('η² Context')
    ax.set_title('Token vs. Context Effect Sizes')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(dist_dir / 'eta_sq_scatter.png', bbox_inches='tight')
    plt.close()
    
    # Plot 2: Histograms of η² values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(all_eta_token, bins=30, color='#C44E52', alpha=0.7, edgecolor='white')
    axes[0].axvline(np.mean(all_eta_token), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_eta_token):.4f}')
    axes[0].set_xlabel('η² Token')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Token Effect Sizes')
    axes[0].legend()
    
    axes[1].hist(all_eta_context, bins=30, color='#55A868', alpha=0.7, edgecolor='white')
    axes[1].axvline(np.mean(all_eta_context), color='green', linestyle='--',
                    label=f'Mean: {np.mean(all_eta_context):.4f}')
    axes[1].set_xlabel('η² Context')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Context Effect Sizes')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(dist_dir / 'eta_sq_histograms.png', bbox_inches='tight')
    plt.close()
    
    # Plot 3: Dominant factor distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    
    dom_counts = {}
    for d in all_dominant:
        dom_counts[d] = dom_counts.get(d, 0) + 1
    
    categories = list(dom_counts.keys())
    counts = list(dom_counts.values())
    bar_colors = [colors.get(c, 'gray') for c in categories]
    
    bars = ax.bar(categories, counts, color=bar_colors, alpha=0.8)
    ax.set_xlabel('Dominant Factor')
    ax.set_ylabel('Number of Features')
    ax.set_title('Distribution of Dominant Factors')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(dist_dir / 'dominant_factor_distribution.png', bbox_inches='tight')
    plt.close()


def plot_anova_by_layer(anova_data: dict, output_dir: Path):
    """Plot ANOVA results separately for each layer."""
    layer_dir = output_dir / 'anova' / 'by_layer'
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    for layer, data in sorted(anova_data.items()):
        features = data.get('features', [])
        if not features:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        eta_token = [f.get('eta_sq_token', 0) for f in features]
        eta_context = [f.get('eta_sq_context', 0) for f in features]
        f_token = [f.get('f_token', 0) for f in features]
        f_context = [f.get('f_context', 0) for f in features]
        
        # Scatter: η² token vs context
        ax = axes[0, 0]
        ax.scatter(eta_token, eta_context, alpha=0.6, s=50, c='steelblue')
        max_val = max(max(eta_token), max(eta_context)) if eta_token else 1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        ax.set_xlabel('η² Token')
        ax.set_ylabel('η² Context')
        ax.set_title('Effect Sizes: Token vs Context')
        
        # Histogram: η² difference
        ax = axes[0, 1]
        diff = [t - c for t, c in zip(eta_token, eta_context)]
        ax.hist(diff, bins=25, color='coral', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(np.mean(diff), color='red', linestyle='--',
                   label=f'Mean: {np.mean(diff):.4f}')
        ax.set_xlabel('η² Token - η² Context')
        ax.set_ylabel('Count')
        ax.set_title('Difference in Effect Sizes')
        ax.legend()
        
        # Bar: Cell means
        ax = axes[1, 0]
        mean_A = np.mean([f.get('mean_A', 0) for f in features])
        mean_B = np.mean([f.get('mean_B', 0) for f in features])
        mean_C = np.mean([f.get('mean_C', 0) for f in features])
        mean_D = np.mean([f.get('mean_D', 0) for f in features])
        
        cells = ['A\n(R+T)', 'B\n(R-T)', 'C\n(NR+T)', 'D\n(NR-T)']
        means = [mean_A, mean_B, mean_C, mean_D]
        colors_cells = ['#C44E52', '#55A868', '#DD8452', '#4C72B0']
        
        ax.bar(cells, means, color=colors_cells, alpha=0.8)
        ax.set_xlabel('Condition')
        ax.set_ylabel('Mean Activation')
        ax.set_title('Average Cell Means Across Features')
        
        # Dominant factor pie
        ax = axes[1, 1]
        dom_counts = {}
        for f in features:
            d = f.get('dominant_factor', 'none')
            dom_counts[d] = dom_counts.get(d, 0) + 1
        
        colors_pie = {'token': '#C44E52', 'context': '#55A868',
                      'interaction': '#4C72B0', 'mixed': '#DD8452', 'none': 'gray'}
        
        labels = list(dom_counts.keys())
        sizes = list(dom_counts.values())
        pie_colors = [colors_pie.get(l, 'gray') for l in labels]
        
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%',
               startangle=90)
        ax.set_title('Dominant Factor Distribution')
        
        plt.suptitle(f'Layer {layer} ANOVA Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(layer_dir / f'layer{layer}_anova.png', bbox_inches='tight')
        plt.close()


def plot_anova_cell_means(anova_data: dict, output_dir: Path):
    """Plot cell means pattern across layers."""
    anova_dir = output_dir / 'anova'
    anova_dir.mkdir(parents=True, exist_ok=True)
    
    layers = sorted(anova_data.keys())
    
    # Collect mean cell values across features for each layer
    means = {'A': [], 'B': [], 'C': [], 'D': []}
    
    for layer in layers:
        features = anova_data[layer].get('features', [])
        if features:
            means['A'].append(np.mean([f.get('mean_A', 0) for f in features]))
            means['B'].append(np.mean([f.get('mean_B', 0) for f in features]))
            means['C'].append(np.mean([f.get('mean_C', 0) for f in features]))
            means['D'].append(np.mean([f.get('mean_D', 0) for f in features]))
        else:
            for key in means:
                means[key].append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(layers))
    width = 0.2
    
    colors = {'A': '#C44E52', 'B': '#55A868', 'C': '#DD8452', 'D': '#4C72B0'}
    labels = {
        'A': 'Reasoning + Tokens',
        'B': 'Reasoning - Tokens',
        'C': 'Non-Reasoning + Tokens',
        'D': 'Non-Reasoning - Tokens'
    }
    
    for i, (cell, values) in enumerate(means.items()):
        offset = (i - 1.5) * width
        ax.bar(x + offset, values, width, label=labels[cell], 
               color=colors[cell], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Average Cell Means by Layer and Condition')
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(anova_dir / 'cell_means_by_layer.png', bbox_inches='tight')
    plt.close()


def plot_all_anova(data: dict, output_dir: Path):
    """Generate all ANOVA-related plots."""
    # Try to load ANOVA data
    anova_data = None
    
    # Check various possible locations
    for layer in data.get('layers', []):
        possible_paths = [
            output_dir.parent / 'results' / output_dir.name / f'layer{layer}' / 'anova' / 'anova_results.json',
        ]
        for path in possible_paths:
            if path.exists():
                if anova_data is None:
                    anova_data = {}
                with open(path) as f:
                    anova_data[layer] = json.load(f)
    
    if anova_data is None:
        # Try loading from the data dict if available
        anova_data = data.get('anova', None)
    
    if anova_data is None:
        print("  No ANOVA data found, skipping ANOVA plots")
        return
    
    print("  Generating ANOVA plots...")
    plot_anova_summary(anova_data, output_dir)
    plot_anova_distributions(anova_data, output_dir)
    plot_anova_by_layer(anova_data, output_dir)
    plot_anova_cell_means(anova_data, output_dir)


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
        'avg_transfer_ratio': [],
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
        
        metrics['avg_transfer_ratio'].append(summary.get('avg_transfer_ratio', 0))
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
    
    # Plot 1: Transfer ratio and token-driven percentages
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Transfer ratio
    ax = axes[0]
    ax.bar(range(len(layers)), metrics['avg_transfer_ratio'], color='#C44E52', alpha=0.8)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Token-driven threshold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Transfer Ratio')
    ax.set_title('Token Injection Transfer Ratio')
    ax.legend(fontsize=8)
    for i, v in enumerate(metrics['avg_transfer_ratio']):
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
            
            transfer_ratios = []
            feature_indices = []
            
            for f in features:
                strat_data = f.get('strategies', {}).get(strategy, {})
                transfer_ratios.append(strat_data.get('transfer_ratio', 0))
                feature_indices.append(f.get('feature_index', 0))
            
            ax.bar(range(len(transfer_ratios)), transfer_ratios, color=color, alpha=0.8)
            ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Token-driven threshold')
            ax.axhline(0.2, color='gray', linestyle=':', alpha=0.5, label='Partial threshold')
            ax.set_xticks(range(len(feature_indices)))
            ax.set_xticklabels([str(idx) for idx in feature_indices], rotation=45, fontsize=8)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Transfer Ratio')
            ax.set_title(f'Strategy: {strategy}')
            ax.legend(fontsize=7)
        
        # Hide unused subplots
        for ax_idx in range(len(strategies), len(axes)):
            axes[ax_idx].set_visible(False)
        
        plt.suptitle(f'Layer {layer}: Transfer Ratios by Injection Strategy', fontsize=14)
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
        
        # Best transfer ratio distribution
        ax = axes[1]
        best_ratios = [f.get('best_transfer_ratio', 0) for f in features]
        ax.hist(best_ratios, bins=15, color='#C44E52', alpha=0.7, edgecolor='white')
        ax.axvline(0.5, color='black', linestyle='--', label='Token-driven threshold')
        ax.axvline(0.2, color='gray', linestyle=':', label='Partial threshold')
        ax.axvline(np.mean(best_ratios), color='red', linestyle='-', 
                   label=f'Mean: {np.mean(best_ratios):.2f}')
        ax.set_xlabel('Best Transfer Ratio')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Best Transfer Ratios')
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
        # Use the best strategy (highest transfer ratio) for injected result
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
                       plot_anova_: bool = True,
                       plot_injection_: bool = True,
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
    
    if plot_anova_:
        # Try to load and plot ANOVA data
        anova_data = load_anova_data(experiment_dir)
        if anova_data:
            print("  Generating ANOVA plots...")
            plot_anova_summary(anova_data, output_dir)
            plot_anova_distributions(anova_data, output_dir)
            plot_anova_by_layer(anova_data, output_dir)
            plot_anova_cell_means(anova_data, output_dir)
    
    if plot_injection_:
        if data.get('injection_results'):
            print("  Generating injection experiment plots...")
            plot_injection_summary(data, output_dir)
            plot_injection_per_feature(data, output_dir)
            plot_injection_activation_comparison(data, output_dir)
    
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
        '--only-anova',
        action='store_true',
        help='Only generate ANOVA plots',
    )
    parser.add_argument(
        '--only-injection',
        action='store_true',
        help='Only generate injection experiment plots',
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
        '--no-anova',
        action='store_true',
        help='Skip ANOVA plots',
    )
    parser.add_argument(
        '--no-injection',
        action='store_true',
        help='Skip injection experiment plots',
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
                  args.only_scatter, args.only_steering, args.only_anova, 
                  args.only_injection, args.only_summary]
    
    if any(only_flags):
        # Only specified categories
        plot_layer_stats = args.only_layer_stats
        plot_distributions = args.only_distributions
        plot_token = args.only_token
        plot_scatter = args.only_scatter
        plot_steering = args.only_steering
        plot_anova = args.only_anova
        plot_injection = args.only_injection
        plot_summary = args.only_summary
    else:
        # All categories except excluded
        plot_layer_stats = not args.no_layer_stats
        plot_distributions = not args.no_distributions
        plot_token = not args.no_token
        plot_scatter = not args.no_scatter
        plot_steering = not args.no_steering
        plot_anova = not args.no_anova
        plot_injection = not args.no_injection
        plot_summary = not args.no_summary
    
    print(f"\nPlot categories enabled:")
    print(f"  Layer stats: {plot_layer_stats}")
    print(f"  Distributions: {plot_distributions}")
    print(f"  Token dependency: {plot_token}")
    print(f"  Scatter plots: {plot_scatter}")
    print(f"  Steering results: {plot_steering}")
    print(f"  ANOVA: {plot_anova}")
    print(f"  Injection: {plot_injection}")
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
            plot_anova_=plot_anova,
            plot_injection_=plot_injection,
            plot_summary_=plot_summary,
        )
    
    print("\n" + "=" * 60)
    print("PLOTTING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
