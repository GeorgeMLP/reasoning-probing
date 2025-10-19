# SAE Probing Experiments - Complete Guide

This guide explains how to use the probing experiment framework to investigate information loss in Sparse Autoencoder (SAE) reconstructions, with a focus on whether higher-level features like reasoning are captured by linear SAEs.

## Research Question

**Do linear SAEs capture higher-level nonlinear features (like reasoning)?**

We hypothesize that reasoning information may not be fully captured by the SAE's linear features and instead remains in the residue (reconstruction error). To test this, we train probes to classify reasoning vs. non-reasoning sequences using three types of activations:

1. **Original activations** - Baseline (all information)
2. **Reconstructed activations** - What the SAE captures
3. **Residue activations** - What the SAE misses (original - reconstructed)

## Quick Start

### 1. Validation Test (5-10 minutes)

Test that everything works:

```bash
cd /home/exouser/reasoning-probing
./probing/run_experiments.sh validate
```

### 2. Quick Experiment (10 minutes)

Run a small-scale experiment:

```bash
./probing/run_experiments.sh quick
```

### 3. Full Experiment (1-2 hours)

Run the complete experiment:

```bash
./probing/run_experiments.sh full
```

### 4. Analyze Results

```bash
python probing/analyze_results.py data/probing/experiments/full_exp --plot --save_plots results_plots
```

## Experiment Types

The `run_experiments.sh` script provides several pre-configured experiments:

| Type | Description | Time | Command |
|------|-------------|------|---------|
| `validate` | Validation tests | 5-10 min | `./probing/run_experiments.sh validate` |
| `quick` | Quick test (100 samples) | ~10 min | `./probing/run_experiments.sh quick` |
| `small` | Small exp (1K samples) | ~30 min | `./probing/run_experiments.sh small` |
| `full` | Full exp (5K samples) | ~2 hours | `./probing/run_experiments.sh full` |
| `mlp` | MLP probe comparison | ~2 hours | `./probing/run_experiments.sh mlp` |
| `fine_grained` | Fine-grained reasoning types | ~2 hours | `./probing/run_experiments.sh fine_grained` |
| `multi_layer` | Compare layers 4,8,12,16 | ~3 hours | `./probing/run_experiments.sh multi_layer` |

## Manual Usage

For more control, use the Python script directly:

```bash
python probing/run_probing_experiment.py \
    --model_name google/gemma-2-2b \
    --sae_name gemma-scope-2b-pt-res-canonical \
    --layer_index 8 \
    --normal_samples 5000 \
    --reasoning_samples 500 \
    --probe_type linear \
    --label_type binary \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --num_epochs 100 \
    --patience 10 \
    --device cuda \
    --save_dir data/probing/my_experiment
```

### Key Parameters

**Model Configuration:**
- `--model_name`: LLM to use (default: `google/gemma-2-2b`)
- `--sae_name`: SAE release (default: `gemma-scope-2b-pt-res-canonical`)
- `--layer_index`: Which layer to probe (default: `8`)

**Dataset:**
- `--normal_samples`: Non-reasoning samples from Pile (default: `5000`)
- `--reasoning_samples`: Reasoning samples (default: all available ~500)
- `--label_type`: `binary` (reasoning vs. non-reasoning) or `fine_grained` (6 reasoning types)

**Probe:**
- `--probe_type`: `linear`, `mlp_1`, `mlp_2`, or `mlp_3`
  - `linear`: Single linear layer (fastest, baseline)
  - `mlp_1`: 1 hidden layer (512 dims)
  - `mlp_2`: 2 hidden layers (512, 256 dims)
  - `mlp_3`: 3 hidden layers (512, 256, 128 dims)

**Training:**
- `--batch_size`: Batch size (default: `64`)
- `--learning_rate`: Learning rate (default: `1e-3`)
- `--num_epochs`: Max epochs (default: `100`)
- `--patience`: Early stopping patience (default: `10`)

**Other:**
- `--device`: `cuda` or `cpu` (default: `cuda`)
- `--save_dir`: Where to save results
- `--load_activations`: Reuse pre-computed activations (saves time)

## Output Structure

Each experiment creates the following structure:

```
data/probing/experiments/[exp_name]/
├── activations/
│   └── activations.pt              # Collected activations (can be reused)
├── probe_original.pt               # Trained probe on original activations
├── probe_reconstructed.pt          # Trained probe on reconstructed activations
├── probe_residue.pt                # Trained probe on residue activations
├── training_curves_*.png           # Training curves for each probe
├── confusion_matrix_*.png          # Confusion matrices
├── results.json                    # All metrics and results
└── config.json                     # Experiment configuration
```

## Interpreting Results

The experiment outputs a comparison table like:

```
Activation Type      Accuracy     F1 Score     Precision    Recall      
--------------------------------------------------------------------
Original             0.9500       0.9480       0.9520       0.9450      
Reconstructed        0.8200       0.8150       0.8300       0.8100      
Residue              0.7800       0.7750       0.7900       0.7650
```

### If Reconstructed Performs Best:
✅ **SAE captures reasoning information**
- The linear features in the SAE encode reasoning patterns
- Little information loss for reasoning
- SAE successfully decomposes reasoning into linear features

### If Residue Performs Best:
✅ **Reasoning NOT captured by SAE**
- Higher-level nonlinear features remain in reconstruction error
- The SAE's linear decomposition misses reasoning patterns
- **This supports the research hypothesis**

### If Original >> Both:
✅ **Information is distributed**
- Both reconstructed and residue contain partial information
- Compare their relative performance to see where more information is

## Advanced Experiments

### 1. Compare Different Layers

```bash
./probing/run_experiments.sh multi_layer
```

This runs experiments on layers 4, 8, 12, and 16, then generates comparison plots. Helps understand:
- Which layers contain more reasoning information
- How information is distributed across the network
- Where SAEs work better/worse

### 2. Compare Linear vs. MLP Probes

```bash
# First run with linear probe
./probing/run_experiments.sh full

# Then run with MLP probe (reuses activations)
./probing/run_experiments.sh mlp

# Compare
python probing/analyze_results.py \
    data/probing/experiments/full_exp \
    data/probing/experiments/mlp_exp \
    --plot --compare \
    --save_plots comparison_plots
```

If MLP significantly outperforms linear, it suggests nonlinear relationships in the activations.

### 3. Fine-Grained Reasoning Classification

Instead of binary (reasoning vs. non-reasoning), classify into 6 reasoning types:
- Initializing
- Deduction
- Adding knowledge
- Example testing
- Uncertainty estimation
- Backtracking

```bash
./probing/run_experiments.sh fine_grained
```

This tests whether different reasoning types are captured differently by the SAE.

### 4. Reuse Activations for Multiple Experiments

Activation collection is slow. Reuse them:

```bash
# First experiment: collect activations
python probing/run_probing_experiment.py \
    --normal_samples 5000 \
    --reasoning_samples 500 \
    --probe_type linear \
    --save_dir exp1

# Second experiment: reuse activations, different probe
python probing/run_probing_experiment.py \
    --load_activations exp1/activations/activations.pt \
    --probe_type mlp_2 \
    --save_dir exp2

# Third experiment: different hyperparameters
python probing/run_probing_experiment.py \
    --load_activations exp1/activations/activations.pt \
    --probe_type linear \
    --learning_rate 1e-4 \
    --save_dir exp3
```

## Analyzing Results

### Command Line

```bash
# Analyze single experiment
python probing/analyze_results.py data/probing/experiments/full_exp

# Generate plots
python probing/analyze_results.py data/probing/experiments/full_exp --plot --save_plots plots/

# Compare multiple experiments
python probing/analyze_results.py \
    data/probing/experiments/exp1 \
    data/probing/experiments/exp2 \
    data/probing/experiments/exp3 \
    --plot --compare \
    --save_plots comparison/
```

### In Python

```python
from probing.analyze_results import load_experiment_results, plot_single_experiment, print_summary

# Load experiment
exp = load_experiment_results('data/probing/experiments/full_exp')

# Print summary
print_summary(exp)

# Generate plots
plot_single_experiment(exp, save_path='my_plot.png')
```

## Tips and Best Practices

### 1. Start Small
Always test with small samples first to catch errors quickly:
```bash
./probing/run_experiments.sh quick
```

### 2. Balance Your Dataset
Use similar numbers of normal and reasoning samples to avoid class imbalance:
```bash
python probing/run_probing_experiment.py \
    --normal_samples 500 \
    --reasoning_samples 500
```

### 3. Check GPU Memory
If you run out of GPU memory:
- Reduce `--batch_size`
- Reduce `--normal_samples` or `--reasoning_samples`
- Use a smaller model

### 4. Early Stopping
The trainer uses early stopping automatically. If training stops early:
- ✅ Good: The model converged
- ⚠️ Check: If it stops at epoch 2-3, increase `--learning_rate`
- ⚠️ Check: If it never stops, increase `--patience` or reduce `--num_epochs`

### 5. Reproducibility
Results use fixed random seeds for reproducibility. Running the same experiment twice should give similar results.

### 6. Save Activations
For multiple experiments, save activations once and reuse:
```bash
# Collect once
python probing/run_probing_experiment.py \
    --normal_samples 5000 \
    --save_dir base_exp

# Reuse many times
python probing/run_probing_experiment.py \
    --load_activations base_exp/activations/activations.pt \
    --probe_type mlp_1 \
    --save_dir exp_mlp1

python probing/run_probing_experiment.py \
    --load_activations base_exp/activations/activations.pt \
    --probe_type mlp_2 \
    --save_dir exp_mlp2
```

## Troubleshooting

### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce `--batch_size` or `--normal_samples`

### No Reasoning Dataset
```
FileNotFoundError: reasoning_dataset/annotated_dataset.json
```
**Solution:** Ensure `reasoning_dataset/annotated_dataset.json` exists

### Poor Performance (all ~50% accuracy)
**Possible causes:**
- Dataset too small (increase `--normal_samples` and `--reasoning_samples`)
- Learning rate too high/low (try `--learning_rate 1e-4` or `1e-2`)
- Not enough epochs (increase `--num_epochs` or reduce `--patience`)

### Training Not Converging
**Solution:** 
- Try different `--learning_rate` (1e-4, 1e-3, 1e-2)
- Increase `--patience`
- Try MLP probe instead of linear

## Project Structure

```
probing/
├── dataset_interface.py         # Unified dataset handling
├── probe_model.py               # MLP probe models
├── trainer.py                   # Training pipeline with early stopping
├── run_probing_experiment.py    # Main experiment script
├── analyze_results.py           # Results analysis and visualization
├── test_implementation.py       # Validation tests
├── run_experiments.sh           # Convenience script for common experiments
└── README.md                    # Detailed documentation
```

## Support

For issues, questions, or contributions:
1. Check this guide first
2. Run validation tests: `./probing/run_experiments.sh validate`
3. Try a quick experiment: `./probing/run_experiments.sh quick`
4. Check the detailed README: `probing/README.md`

## Next Steps

1. **Run validation:** `./probing/run_experiments.sh validate`
2. **Quick test:** `./probing/run_experiments.sh quick`
3. **Full experiment:** `./probing/run_experiments.sh full`
4. **Analyze:** `python probing/analyze_results.py data/probing/experiments/full_exp --plot`
5. **Compare layers:** `./probing/run_experiments.sh multi_layer`
6. **Try MLP probes:** `./probing/run_experiments.sh mlp`

Happy probing! 🔬

