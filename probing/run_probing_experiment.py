"""
Main script to run probing experiments comparing original, reconstructed, and residue activations.
"""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
import argparse
from typing import Literal

from dataset_interface import ActivationDatasetBuilder
from probe_model import create_probe
from trainer import ProbeTrainer, plot_training_curves, plot_confusion_matrix


def collate_fn(batch):
    """Collate function for DataLoader."""
    original = torch.stack([item['original'] for item in batch])
    reconstructed = torch.stack([item['reconstructed'] for item in batch])
    residue = torch.stack([item['residue'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    
    return {
        'original': original,
        'reconstructed': reconstructed,
        'residue': residue,
        'label': labels,
    }


def run_probing_experiment(
    model_name: str = 'google/gemma-2-2b',
    sae_name: str = 'gemma-scope-2b-pt-res-canonical',
    layer_index: int = 8,
    normal_samples: int = 5000,
    reasoning_samples: int = None,
    label_type: Literal['binary', 'fine_grained'] = 'binary',
    probe_type: str = 'linear',
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    patience: int = 10,
    train_split: float = 0.7,
    val_split: float = 0.15,
    device: str = 'cuda',
    save_dir: str = 'data/probing/experiments',
    load_activations: str = None,
    use_questions_as_control: bool = True,
):
    """
    Run complete probing experiment.
    
    This trains three separate probes on:
    1. Original activations
    2. Reconstructed activations
    3. Residue activations
    
    Then compares their performance to determine where reasoning information is encoded.
    
    Args:
        model_name: Name of the LLM
        sae_name: Name of the SAE
        layer_index: Which layer to probe
        normal_samples: Number of normal (non-reasoning) samples
        reasoning_samples: Number of reasoning samples (None = all available)
        label_type: 'binary' or 'fine_grained'
        probe_type: Type of probe ('linear', 'mlp_1', 'mlp_2', 'mlp_3')
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Maximum epochs
        patience: Early stopping patience
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        device: Device to use
        save_dir: Directory to save results
        load_activations: Path to load pre-computed activations (optional)
    """
    print("=" * 80)
    print("PROBING EXPERIMENT: Information Loss in SAE Reconstruction")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  SAE: {sae_name}")
    print(f"  Layer: {layer_index}")
    print(f"  Normal samples: {normal_samples}")
    print(f"  Reasoning samples: {reasoning_samples or 'all'}")
    print(f"  Label type: {label_type}")
    print(f"  Probe type: {probe_type}")
    print(f"  Device: {device}")
    print()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load or build dataset
    if load_activations:
        print(f"Loading pre-computed activations from {load_activations}...")
        dataset = ActivationDatasetBuilder.load_dataset(Path(load_activations))
    else:
        print("Building unified activation dataset...")
        builder = ActivationDatasetBuilder(
            model_name=model_name,
            sae_name=sae_name,
            layer_index=layer_index,
            device=device,
        )
        
        dataset = builder.build_dataset(
            normal_samples=normal_samples,
            reasoning_samples=reasoning_samples,
            label_type=label_type,
            save_dir=save_dir / 'activations',
            use_questions_as_control=use_questions_as_control,
        )
    
    # Step 2: Split dataset
    print(f"\nSplitting dataset (train={train_split}, val={val_split}, test={1-train_split-val_split})...")
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # Reproducibility
    )
    
    print(f"  Train: {train_size} samples")
    print(f"  Val: {val_size} samples")
    print(f"  Test: {test_size} samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Get input dimension from first sample
    sample = dataset[0]
    input_dim = sample['original'].shape[0]
    num_classes = 2 if label_type == 'binary' else 7  # 7 classes for fine-grained: 0=non-reasoning, 1-6=reasoning types
    
    print(f"\nProbe configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output classes: {num_classes}")
    print(f"  Probe type: {probe_type}")
    
    # Step 3: Train probes on each activation type
    activation_types = ['original', 'reconstructed', 'residue']
    results = {}
    
    for act_type in activation_types:
        print("\n" + "=" * 80)
        print(f"Training probe on {act_type.upper()} activations")
        print("=" * 80)
        
        # Create fresh model
        model = create_probe(
            input_dim=input_dim,
            num_classes=num_classes,
            probe_type=probe_type,
        )
        
        # Create trainer
        trainer = ProbeTrainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            patience=patience,
        )
        
        # Train
        train_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            activation_type=act_type,
            num_epochs=num_epochs,
            verbose=True,
        )
        
        # Evaluate on test set
        print(f"\nEvaluating on test set...")
        test_metrics = trainer.evaluate(test_loader, act_type, num_classes=num_classes)
        
        print(f"\nTest Results for {act_type}:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        
        # Save results
        results[act_type] = {
            'train_history': train_results['history'],
            'test_metrics': {
                'accuracy': test_metrics['accuracy'],
                'f1': test_metrics['f1'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
            },
            'stopped_epoch': train_results['stopped_epoch'],
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
        }
        
        # Save model
        model_save_path = save_dir / f'probe_{act_type}.pt'
        trainer.save(model_save_path)
        print(f"Saved model to {model_save_path}")
        
        # Plot training curves
        plot_path = save_dir / f'training_curves_{act_type}.png'
        plot_training_curves(train_results['history'], plot_path)
        
        # Plot confusion matrix
        class_names = ['Non-Reasoning', 'Reasoning'] if num_classes == 2 else [
            'Non-Reasoning', 'Initializing', 'Deduction', 'Adding Knowledge',
            'Example Testing', 'Uncertainty Estimation', 'Backtracking'
        ]
        cm_path = save_dir / f'confusion_matrix_{act_type}.png'
        plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, cm_path)
    
    # Step 4: Compare results
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"\n{'Activation Type':<20} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 68)
    
    for act_type in activation_types:
        metrics = results[act_type]['test_metrics']
        print(f"{act_type.capitalize():<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f}")
    
    # Determine which activation type performed best
    best_act_type = max(activation_types, key=lambda x: results[x]['test_metrics']['f1'])
    best_f1 = results[best_act_type]['test_metrics']['f1']
    
    print("\n" + "=" * 80)
    print(f"Best performing activation type: {best_act_type.upper()} (F1 = {best_f1:.4f})")
    print("=" * 80)
    
    # Interpretation
    print("\nInterpretation:")
    if best_act_type == 'residue':
        print("  → Residue activations perform best!")
        print("  → This suggests that reasoning information is NOT fully captured by the SAE.")
        print("  → Higher-level nonlinear features may remain in the reconstruction error.")
    elif best_act_type == 'reconstructed':
        print("  → Reconstructed activations perform best!")
        print("  → This suggests that the SAE successfully captures reasoning information.")
        print("  → Linear features in the SAE encode reasoning patterns.")
    else:
        print("  → Original activations perform best (as expected).")
        print("  → Performance gap between original and reconstructed indicates information loss.")
    
    # Save all results
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all results to {results_path}")
    
    # Save experiment config
    config = {
        'model_name': model_name,
        'sae_name': sae_name,
        'layer_index': layer_index,
        'normal_samples': normal_samples,
        'reasoning_samples': reasoning_samples,
        'label_type': label_type,
        'probe_type': probe_type,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'patience': patience,
        'train_split': train_split,
        'val_split': val_split,
    }
    
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved experiment config to {config_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run probing experiment on SAE activations')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='google/gemma-2-2b',
                       help='Name of the LLM')
    parser.add_argument('--sae_name', type=str, default='gemma-scope-2b-pt-res-canonical',
                       help='Name of the SAE')
    parser.add_argument('--layer_index', type=int, default=8,
                       help='Which layer to probe')
    
    # Dataset arguments
    parser.add_argument('--normal_samples', type=int, default=5000,
                       help='Number of normal (non-reasoning) samples')
    parser.add_argument('--reasoning_samples', type=int, default=None,
                       help='Number of reasoning samples (None = all available)')
    parser.add_argument('--label_type', type=str, default='binary', choices=['binary', 'fine_grained'],
                       help='Label type: binary or fine_grained')
    
    # Training arguments
    parser.add_argument('--probe_type', type=str, default='linear',
                       choices=['linear', 'mlp_1', 'mlp_2', 'mlp_3'],
                       help='Type of probe')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Data split arguments
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Fraction of data for training')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Fraction of data for validation')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='data/probing/experiments',
                       help='Directory to save results')
    parser.add_argument('--load_activations', type=str, default=None,
                       help='Path to load pre-computed activations')
    parser.add_argument('--use_pile_control', action='store_true',
                       help='Use Pile as control instead of questions')
    
    args = parser.parse_args()
    
    # Run experiment
    run_probing_experiment(
        model_name=args.model_name,
        sae_name=args.sae_name,
        layer_index=args.layer_index,
        normal_samples=args.normal_samples,
        reasoning_samples=args.reasoning_samples,
        label_type=args.label_type,
        probe_type=args.probe_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        train_split=args.train_split,
        val_split=args.val_split,
        device=args.device,
        save_dir=args.save_dir,
        load_activations=args.load_activations,
        use_questions_as_control=not args.use_pile_control,
    )


if __name__ == '__main__':
    main()
