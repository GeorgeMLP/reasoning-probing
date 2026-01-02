import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Computer Modern',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts,bm,mathtools}',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})


results_dir = Path("/home/exouser/reasoning-probing/results/cohens_d")
output_dir = Path("/home/exouser/reasoning-probing/figs")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Figure 1: Token concentration and normalized entropy across layers
# ============================================================================

fig1_dir = Path("/home/exouser/reasoning-probing/results/fig1")
model = "gemma-3-4b-it"
dataset = "s1k"

# Check all available layers
model_dataset_path = fig1_dir / model / dataset
available_layers = sorted([int(d.name.replace("layer", "")) 
                           for d in model_dataset_path.iterdir() 
                           if d.is_dir() and d.name.startswith("layer")])

layers_data = []
for layer in available_layers:
    layer_path = model_dataset_path / f"layer{layer}"
    token_analysis_path = layer_path / "token_analysis.json"
    
    if token_analysis_path.exists():
        with open(token_analysis_path) as f:
            data = json.load(f)
            summary = data.get("summary", {})
            
            layers_data.append({
                'layer': layer,
                'mean_token_concentration': summary.get('mean_token_concentration', 0),
                'mean_normalized_entropy': summary.get('mean_normalized_entropy', 0),
            })

if layers_data:
    layers = [d['layer'] for d in layers_data]
    concentrations = [d['mean_token_concentration'] for d in layers_data]
    entropies = [d['mean_normalized_entropy'] for d in layers_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
    
    # Token concentration
    ax1.plot(layers, concentrations, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax1.axvspan(17, 27, alpha=0.15, color='orange', label='Analyzed layers')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Token Concentration')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best')
    
    # Normalized entropy
    ax2.plot(layers, entropies, 'o-', color='darkgreen', linewidth=2, markersize=6)
    ax2.axvspan(17, 27, alpha=0.15, color='orange', label='Analyzed layers')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Mean Normalized Entropy')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    output_path = output_dir / "token_concentration_layers.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {output_path}")
else:
    print("  WARNING: No data found for Figure 1")


# ============================================================================
# Figure 2: Distribution of Cohen's d values across all features
# ============================================================================

layer = 22
feature_stats_path = results_dir / "gemma-3-12b-it" / "s1k" / f"layer{layer}" / "feature_stats.json"

if feature_stats_path.exists():
    with open(feature_stats_path) as f:
        all_features = json.load(f)
    
    # Extract Cohen's d values
    cohens_d_values = [f.get('cohens_d', 0) for f in all_features]
    cohens_d_values = sorted(cohens_d_values, reverse=True)
    
    # Load reasoning features to get the top 100
    reasoning_features_path = results_dir / "gemma-3-12b-it" / "s1k" / f"layer{layer}" / "reasoning_features.json"
    with open(reasoning_features_path) as f:
        reasoning_data = json.load(f)
        top_100_indices = set(reasoning_data.get('feature_indices', []))
    
    # Separate top 100 from others
    top_100_d = [f.get('cohens_d', 0) for f in all_features if f['feature_index'] in top_100_indices]
    other_d = [f.get('cohens_d', 0) for f in all_features if f['feature_index'] not in top_100_indices]
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Histogram with log scale
    bins = np.linspace(min(cohens_d_values), max(cohens_d_values), 50)
    ax.hist(other_d, bins=bins, alpha=0.6, color='gray', label='Other features', edgecolor='none')
    ax.hist(top_100_d, bins=bins, alpha=0.8, color='steelblue', label='Top 100 features', edgecolor='none')
    
    ax.axvline(0.3, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=r'Threshold ($d \geq 0.3$)')
    
    ax.set_xlabel(r"Cohen's $d$")
    ax.set_ylabel('Number of Features')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "cohens_d_distribution.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {output_path}")
else:
    print("  WARNING: Feature stats file not found for Figure 2")


# ============================================================================
# Figure 3: Stacked bar chart of injection classifications
# ============================================================================

# Load injection results for all configurations
configs = [
    ("gemma-3-12b-it", [17, 22, 27]),
    ("gemma-3-4b-it", [17, 22, 27]),
    ("deepseek-r1-distill-llama-8b", [19]),
]

datasets = ["s1k", "general_inquiry_cot"]

# Collect data
all_data = []
for model, layers in configs:
    for layer in layers:
        for dataset in datasets:
            inj_path = results_dir / model / dataset / f"layer{layer}" / "injection_results.json"
            if inj_path.exists():
                with open(inj_path) as f:
                    data = json.load(f)
                    summary = data.get("summary", {})
                    counts = summary.get("classification_counts", {})
                    n_features = summary.get("n_features", 1)
                    
                    # Shorten model names for display
                    model_short = model.replace("-it", "").replace("gemma-3-", "G3-").replace("gemma-3-", "G3-").replace("deepseek-r1-distill-llama-", "DS-")
                    dataset_short = "s1K" if dataset == "s1k" else "GenInq"
                    label = f"{model_short}\nL{layer}\n{dataset_short}"
                    
                    all_data.append({
                        'label': label,
                        'td': counts.get('token_driven', 0) / n_features * 100,
                        'ptd': counts.get('partially_token_driven', 0) / n_features * 100,
                        'wtd': counts.get('weakly_token_driven', 0) / n_features * 100,
                        'cd': counts.get('context_dependent', 0) / n_features * 100,
                    })

if all_data:
    labels = [d['label'] for d in all_data]
    td = [d['td'] for d in all_data]
    ptd = [d['ptd'] for d in all_data]
    wtd = [d['wtd'] for d in all_data]
    cd = [d['cd'] for d in all_data]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    x = np.arange(len(labels))
    width = 0.8
    
    # Stacked bars
    p1 = ax.bar(x, td, width, label='Token-driven', color='#d62728', alpha=0.9)
    p2 = ax.bar(x, ptd, width, bottom=td, label='Partially TD', color='#ff7f0e', alpha=0.9)
    p3 = ax.bar(x, wtd, width, bottom=np.array(td)+np.array(ptd), 
                label='Weakly TD', color='#ffbb78', alpha=0.9)
    p4 = ax.bar(x, cd, width, bottom=np.array(td)+np.array(ptd)+np.array(wtd), 
                label='Context-dependent', color='#1f77b4', alpha=0.9)
    
    ax.set_ylabel(r'Percentage of Features (\%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=7)
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "injection_classification.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {output_path}")
else:
    print("  WARNING: No data found for Figure 3")


print("\n" + "=" * 60)
print("Figure generation complete!")
print(f"All figures saved to: {output_dir}")
print("=" * 60)
print("\nGenerated figures:")
for fig_file in sorted(output_dir.glob("fig*.pdf")):
    print(f"  - {fig_file.name}")
