"""
SAE Probing Experiments Package

This package provides tools for investigating information loss in Sparse Autoencoder (SAE)
reconstructions through probing experiments.

Main components:
- dataset_interface: Unified dataset handling for normal and reasoning data
- probe_model: MLP probing models with configurable depth
- trainer: Training pipeline with early stopping and metrics
- run_probing_experiment: Complete experiment runner
- analyze_results: Results analysis and visualization
"""

from .dataset_interface import (
    UnifiedActivationDataset,
    ActivationDatasetBuilder,
)

from .probe_model import (
    MLPProbe,
    create_probe,
)

from .trainer import (
    ProbeTrainer,
    plot_training_curves,
    plot_confusion_matrix,
)

__version__ = '1.0.0'

__all__ = [
    'UnifiedActivationDataset',
    'ActivationDatasetBuilder',
    'MLPProbe',
    'create_probe',
    'ProbeTrainer',
    'plot_training_curves',
    'plot_confusion_matrix',
]

