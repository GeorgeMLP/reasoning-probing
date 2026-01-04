import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Times',
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
    plt.savefig(output_path, bbox_inches='tight')
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
    plt.savefig(output_path, bbox_inches='tight')
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
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
else:
    print("  WARNING: No data found for Figure 3")


print("\n" + "=" * 60)
print("Figure generation complete!")
print(f"All figures saved to: {output_dir}")
print("=" * 60)
print("\nGenerated figures:")
for fig_file in sorted(output_dir.glob("*.pdf")):
    print(f"  - {fig_file.name}")


# ============================================================================
# APPENDIX FIGURES
# ============================================================================

print("\n" + "=" * 60)
print("GENERATING APPENDIX FIGURES")
print("=" * 60)

# ============================================================================
# Figure A1: Metric comparison rankings
# ============================================================================

# Load top features from each metric
metrics_data = {}
for metric, metric_name in [("cohens_d", "Cohen's d"), ("roc_auc", "ROC-AUC"), ("freq", "Frequency Ratio")]:
    if metric == "cohens_d":
        path = results_dir / "gemma-3-4b-it" / "s1k" / "layer22"
    else:
        path = Path(f"/home/exouser/reasoning-probing/results/{metric}/gemma-3-4b-it/s1k/layer22")
    
    rf_path = path / "reasoning_features.json"
    if rf_path.exists():
        with open(rf_path) as f:
            data = json.load(f)
            features = data.get("features", [])
            
            # Get feature indices and their ranks
            feature_indices = [f['feature_index'] for f in features[:100]]
            metrics_data[metric_name] = set(feature_indices)

# Calculate overlaps
if len(metrics_data) == 3:
    cohens = metrics_data["Cohen's d"]
    auc = metrics_data["ROC-AUC"]
    freq = metrics_data["Frequency Ratio"]
    
    # Compute pairwise Jaccard similarities
    cohens_auc = len(cohens & auc) / len(cohens | auc)
    cohens_freq = len(cohens & freq) / len(cohens | freq)
    auc_freq = len(auc & freq) / len(auc | freq)
    
    # Compute three-way overlap
    all_three = len(cohens & auc & freq)
    
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    metrics = ["Cohen's $d$\nvs AUC", "Cohen's $d$\nvs Freq", "AUC vs\nFreq"]
    similarities = [cohens_auc, cohens_freq, auc_freq]
    
    bars = ax.bar(range(len(metrics)), similarities, color='steelblue', alpha=0.8, width=0.6)
    ax.set_ylabel('Jaccard Similarity')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, similarities)):
        ax.text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=9)
    
    # Add annotation for three-way overlap
    ax.text(0.5, 0.92, f'Three-way overlap: {all_three} features', 
            transform=ax.transAxes, ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "metric_comparison_rankings.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

# ============================================================================
# Figure A2: Token concentration distributions (violin plots)
# ============================================================================

# Collect concentration data for all main configurations
concentration_data = []
labels_data = []

for model, layers in [("gemma-3-12b-it", [17, 22, 27]), ("gemma-3-4b-it", [17, 22, 27])]:
    for layer in layers:
        ta_path = results_dir / model / "s1k" / f"layer{layer}" / "token_analysis.json"
        if ta_path.exists():
            with open(ta_path) as f:
                data = json.load(f)
                features = data.get("features", [])
                concentrations = [f.get("token_concentration", 0) for f in features]
                
                if concentrations:
                    concentration_data.append(concentrations)
                    model_short = "G3-12B" if "12b" in model else "G3-4B"
                    labels_data.append(f"{model_short}\nL{layer}")

if concentration_data:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    parts = ax.violinplot(concentration_data, positions=range(len(concentration_data)),
                          showmeans=False, showmedians=True, widths=0.7)
    
    # Color the violins
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.6)
    
    # Add horizontal line at 0.5 threshold
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='High dependency threshold')
    
    ax.set_ylabel('Token Concentration')
    ax.set_xticks(range(len(labels_data)))
    ax.set_xticklabels(labels_data, fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "token_concentration_distributions.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

# ============================================================================
# Figure A3: Strategy comparison box plots
# ============================================================================

strategies_to_plot = ["prepend", "intersperse", "inject_bigram", "inject_trigram", 
                      "bigram_before", "trigram"]
strategy_labels = ["Prepend", "Intersperse", "Bigram", "Trigram", "Bigram+Ctx", "Trigram+Ctx"]

# Use Gemma-3-12B-IT layer 22 s1K as representative
inj_path = results_dir / "gemma-3-12b-it" / "s1k" / "layer22" / "injection_results.json"

if inj_path.exists():
    with open(inj_path) as f:
        data = json.load(f)
        features = data.get("features", [])
        
        strategy_values = {s: [] for s in strategies_to_plot}
        for feat in features:
            for strat, metrics in feat.get("strategies", {}).items():
                if strat in strategies_to_plot:
                    strategy_values[strat].append(metrics.get("cohens_d", 0))
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        
        data_to_plot = [strategy_values[s] for s in strategies_to_plot]
        bp = ax.boxplot(data_to_plot, tick_labels=strategy_labels, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.axhline(0.8, color='#d62728', linestyle='--', linewidth=1, alpha=0.7, label='Large effect')
        ax.axhline(0.5, color='#ff7f0e', linestyle='--', linewidth=1, alpha=0.7, label='Medium effect')
        ax.axhline(0.2, color='#ffbb78', linestyle='--', linewidth=1, alpha=0.7, label='Small effect')
        
        ax.set_ylabel(r"Cohen's $d$ (Injection Effect)")
        ax.set_xlabel('Injection Strategy')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(loc='upper right', fontsize=7)
        ax.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        output_path = output_dir / "strategy_comparison.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")

# ============================================================================
# Figure A4: LLM iterations distribution
# ============================================================================

# Collect iteration counts from all interpretations
all_iterations = []

configs = [
    ("gemma-3-12b-it", [17, 22, 27]),
    ("gemma-3-4b-it", [17, 22, 27]),
    ("deepseek-r1-distill-llama-8b", [19]),
]

for model, layers in configs:
    for layer in layers:
        for dataset in ["s1k", "general_inquiry_cot"]:
            interp_path = results_dir / model / dataset / f"layer{layer}" / "feature_interpretations.json"
            if interp_path.exists():
                with open(interp_path) as f:
                    data = json.load(f)
                    features = data.get("features", [])
                    for f in features:
                        iterations = f.get("iterations_used", 0)
                        all_iterations.append(iterations)

if all_iterations:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    # Histogram
    bins = np.arange(0.5, max(all_iterations) + 1.5, 1)
    ax.hist(all_iterations, bins=bins, color='steelblue', alpha=0.8, edgecolor='white')
    
    ax.set_xlabel('Iterations to Convergence')
    ax.set_ylabel('Number of Features')
    ax.set_xticks(range(1, max(all_iterations) + 1))
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add statistics
    mean_iter = np.mean(all_iterations)
    median_iter = np.median(all_iterations)
    ax.axvline(mean_iter, color='red', linestyle='--', linewidth=1.5, 
               label=f'Mean: {mean_iter:.1f}')
    ax.axvline(median_iter, color='orange', linestyle=':', linewidth=1.5,
               label=f'Median: {median_iter:.0f}')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / "llm_iterations_distribution.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

# ============================================================================
# Figure A5: Dataset overlap Venn-like visualization
# ============================================================================

# Calculate overlaps for each model and layer
overlap_data = []

for model, layers in [("gemma-3-12b-it", [17, 22, 27]), ("gemma-3-4b-it", [17, 22, 27])]:
    for layer in layers:
        s1k_path = results_dir / model / "s1k" / f"layer{layer}" / "reasoning_features.json"
        gen_path = results_dir / model / "general_inquiry_cot" / f"layer{layer}" / "reasoning_features.json"
        
        if s1k_path.exists() and gen_path.exists():
            with open(s1k_path) as f:
                s1k_indices = set(json.load(f).get("feature_indices", []))
            with open(gen_path) as f:
                gen_indices = set(json.load(f).get("feature_indices", []))
            
            intersection = len(s1k_indices & gen_indices)
            union = len(s1k_indices | gen_indices)
            jaccard = intersection / union if union > 0 else 0
            
            model_short = "G3-12B" if "12b" in model else "G3-4B"
            overlap_data.append({
                'label': f"{model_short} L{layer}",
                's1k_only': len(s1k_indices - gen_indices),
                'gen_only': len(gen_indices - s1k_indices),
                'shared': intersection,
                'jaccard': jaccard,
            })

if overlap_data:
    labels = [d['label'] for d in overlap_data]
    s1k_only = [d['s1k_only'] for d in overlap_data]
    shared = [d['shared'] for d in overlap_data]
    gen_only = [d['gen_only'] for d in overlap_data]
    jaccard_vals = [d['jaccard'] for d in overlap_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    # Stacked bar chart
    x = np.arange(len(labels))
    width = 0.6
    
    ax1.bar(x, s1k_only, width, label='s1K only', color='#1f77b4', alpha=0.8)
    ax1.bar(x, shared, width, bottom=s1k_only, label='Shared', color='#2ca02c', alpha=0.8)
    ax1.bar(x, gen_only, width, bottom=np.array(s1k_only)+np.array(shared), 
            label='Gen. Inq. only', color='#ff7f0e', alpha=0.8)
    
    ax1.set_ylabel('Number of Features')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax1.legend(loc='upper left', fontsize=7)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Jaccard similarity
    ax2.bar(x, jaccard_vals, width=0.6, color='steelblue', alpha=0.8)
    ax2.set_ylabel('Jaccard Similarity')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels
    for i, val in enumerate(jaccard_vals):
        ax2.text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / "dataset_overlap.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

print("\n" + "=" * 60)
print("All appendix figures generated!")
print("=" * 60)
