# SAE Probing Experiments

This directory contains tools for probing Sparse Autoencoder (SAE) activations to investigate information loss during reconstruction, with a focus on whether higher-level nonlinear features (like reasoning) are captured by the linear SAE or remain in the residue.

## Overview

The experiment trains three separate probes on:
1. **Original activations** - baseline performance
2. **Reconstructed activations** - what the SAE captures
3. **Residue activations** - what the SAE misses (original - reconstructed)

By comparing performance, we can determine where reasoning information is encoded.

## Quick Start

### Run a complete experiment:

```bash
python probing/run_probing_experiment.py \
    --model_name google/gemma-2-2b \
    --layer_index 8 \
    --normal_samples 5000 \
    --reasoning_samples 500 \
    --probe_type linear \
    --label_type binary \
    --device cuda
```

### Run a small test experiment:

```bash
python probing/run_probing_experiment.py \
    --model_name google/gemma-2-2b \
    --layer_index 8 \
    --normal_samples 100 \
    --reasoning_samples 50 \
    --probe_type linear \
    --num_epochs 20 \
    --save_dir data/probing/test
```

## Components

### 1. `dataset_interface.py`
Unified dataset interface that handles both normal (Pile) and reasoning datasets.

**Key Classes:**
- `UnifiedActivationDataset`: PyTorch Dataset that provides original, reconstructed, and residue activations with labels
- `ActivationDatasetBuilder`: Collects activations from both datasets using the model and SAE

**Usage:**
```python
from dataset_interface import ActivationDatasetBuilder

builder = ActivationDatasetBuilder(
    model_name='google/gemma-2-2b',
    layer_index=8,
)

dataset = builder.build_dataset(
    normal_samples=5000,
    reasoning_samples=500,
    label_type='binary',
    save_dir='data/probing/activations'
)
```

### 2. `probe_model.py`
MLP probing models with configurable depth.

**Probe Types:**
- `linear`: Single linear layer (default)
- `mlp_1`: 1 hidden layer (512 dims)
- `mlp_2`: 2 hidden layers (512, 256 dims)
- `mlp_3`: 3 hidden layers (512, 256, 128 dims)

**Usage:**
```python
from probe_model import create_probe

probe = create_probe(
    input_dim=2304,  # Gemma-2-2b hidden dim
    num_classes=2,   # Binary classification
    probe_type='linear'
)
```

### 3. `trainer.py`
Training pipeline with early stopping and metric tracking.

**Features:**
- Early stopping based on validation loss
- Learning rate scheduling (cosine annealing)
- Comprehensive metrics (accuracy, F1, precision, recall)
- Training curve visualization
- Confusion matrix plotting

**Usage:**
```python
from trainer import ProbeTrainer

trainer = ProbeTrainer(
    model=probe,
    learning_rate=1e-3,
    patience=10,
)

results = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    activation_type='residue',
    num_epochs=100,
)
```

### 4. `run_probing_experiment.py`
Main experiment script that ties everything together.

## Arguments

### Model Configuration
- `--model_name`: LLM name (default: `google/gemma-2-2b`)
- `--sae_name`: SAE name (default: `gemma-scope-2b-pt-res-canonical`)
- `--layer_index`: Layer to probe (default: `8`)

### Dataset Configuration
- `--normal_samples`: Number of non-reasoning samples (default: `5000`)
- `--reasoning_samples`: Number of reasoning samples (default: all available)
- `--label_type`: `binary` or `fine_grained` (default: `binary`)

### Training Configuration
- `--probe_type`: `linear`, `mlp_1`, `mlp_2`, or `mlp_3` (default: `linear`)
- `--batch_size`: Batch size (default: `64`)
- `--learning_rate`: Learning rate (default: `1e-3`)
- `--num_epochs`: Maximum epochs (default: `100`)
- `--patience`: Early stopping patience (default: `10`)
- `--train_split`: Training data fraction (default: `0.7`)
- `--val_split`: Validation data fraction (default: `0.15`)

### Other
- `--device`: Device to use (default: `cuda`)
- `--save_dir`: Results directory (default: `data/probing/experiments`)
- `--load_activations`: Load pre-computed activations (optional)

## Outputs

The experiment saves:
- `activations.pt`: Collected activations from both datasets
- `probe_{original|reconstructed|residue}.pt`: Trained probe models
- `training_curves_{original|reconstructed|residue}.png`: Training curves
- `confusion_matrix_{original|reconstructed|residue}.png`: Confusion matrices
- `results.json`: All metrics and results
- `config.json`: Experiment configuration

## Example Experiment

```bash
# Full experiment with linear probe
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
    --save_dir data/probing/exp1

# Compare with MLP probe
python probing/run_probing_experiment.py \
    --model_name google/gemma-2-2b \
    --layer_index 8 \
    --load_activations data/probing/exp1/activations/activations.pt \
    --probe_type mlp_2 \
    --save_dir data/probing/exp2
```

## Interpreting Results

The final comparison shows performance of probes on each activation type:

**If residue performs best:**
- Reasoning information is NOT fully captured by the SAE
- Higher-level nonlinear features remain in the reconstruction error
- The SAE's linear features miss important reasoning patterns

**If reconstructed performs best:**
- The SAE successfully captures reasoning information
- Linear features in the SAE encode reasoning patterns
- Little information loss for reasoning

**If original performs best (expected):**
- Performance gap indicates information loss
- Compare reconstructed vs. residue to see where information went

## Label Types

### Binary (`--label_type binary`)
- Class 0: Non-reasoning (from Pile dataset)
- Class 1: Reasoning (from reasoning dataset)

### Fine-grained (`--label_type fine_grained`)
- Class 0: Initializing
- Class 1: Deduction
- Class 2: Adding knowledge
- Class 3: Example testing
- Class 4: Uncertainty estimation
- Class 5: Backtracking

## Memory Management

The dataset interface uses efficient memory management:
- Activations are collected in batches
- Data can be saved and reloaded to avoid recomputation
- Use `--load_activations` to skip activation collection

## Tips

1. **Start small**: Test with `--normal_samples 100 --reasoning_samples 50` first
2. **Save activations**: Use `--save_dir` and then `--load_activations` for multiple experiments
3. **Try different probes**: Compare linear vs. MLP to see if nonlinearity helps
4. **Check multiple layers**: Run experiments on different `--layer_index` values
5. **Balance dataset**: Use similar numbers for `--normal_samples` and `--reasoning_samples`
