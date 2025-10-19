"""
Test script to validate the probing experiment implementation.
Runs a small-scale test to ensure everything works.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from probing.dataset_interface import ActivationDatasetBuilder, UnifiedActivationDataset
from probing.probe_model import create_probe
from probing.trainer import ProbeTrainer
from torch.utils.data import DataLoader, random_split


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


def test_dataset_interface():
    """Test the dataset interface with small samples."""
    print("=" * 80)
    print("TEST 1: Dataset Interface")
    print("=" * 80)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("WARNING: Running on CPU. This will be slow.")
        print("Consider running on GPU for faster testing.")
    
    # Create builder
    builder = ActivationDatasetBuilder(
        model_name='google/gemma-2-2b',
        sae_name='gemma-scope-2b-pt-res-canonical',
        layer_index=8,
        device=device,
    )
    
    # Test with very small samples
    print("\nBuilding small test dataset...")
    try:
        dataset = builder.build_dataset(
            normal_samples=10,
            reasoning_samples=5,
            label_type='binary',
            save_dir=Path('data/probing/test_validation'),
        )
        
        print(f"✓ Dataset created successfully!")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Sample keys: {list(dataset[0].keys())}")
        print(f"  Activation shape: {dataset[0]['original'].shape}")
        
        return dataset, True
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_probe_model(dataset):
    """Test the probe model."""
    print("\n" + "=" * 80)
    print("TEST 2: Probe Model")
    print("=" * 80)
    
    if dataset is None:
        print("✗ Skipping (dataset not available)")
        return None, False
    
    try:
        # Get input dimension
        input_dim = dataset[0]['original'].shape[0]
        print(f"Input dimension: {input_dim}")
        
        # Test linear probe
        print("\nTesting linear probe...")
        linear_probe = create_probe(input_dim, num_classes=2, probe_type='linear')
        print(f"✓ Linear probe created")
        print(f"  Parameters: {sum(p.numel() for p in linear_probe.parameters())}")
        
        # Test MLP probe
        print("\nTesting MLP probe...")
        mlp_probe = create_probe(input_dim, num_classes=2, probe_type='mlp_2')
        print(f"✓ MLP probe created")
        print(f"  Parameters: {sum(p.numel() for p in mlp_probe.parameters())}")
        
        # Test forward pass
        x = dataset[0]['original'].unsqueeze(0)
        output = linear_probe(x)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        
        return linear_probe, True
    except Exception as e:
        print(f"✗ Probe model test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_training(dataset, model):
    """Test the training pipeline."""
    print("\n" + "=" * 80)
    print("TEST 3: Training Pipeline")
    print("=" * 80)
    
    if dataset is None or model is None:
        print("✗ Skipping (dependencies not available)")
        return False
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Split dataset
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"Train size: {train_size}, Val size: {val_size}")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)
        
        print(f"✓ DataLoaders created")
        
        # Create trainer
        trainer = ProbeTrainer(
            model=model,
            device=device,
            learning_rate=1e-3,
            patience=3,
        )
        print(f"✓ Trainer created")
        
        # Train for a few epochs
        print("\nTraining for 5 epochs on 'original' activations...")
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            activation_type='original',
            num_epochs=5,
            verbose=True,
        )
        
        print(f"✓ Training completed")
        print(f"  Stopped at epoch: {results['stopped_epoch'] + 1 if results['stopped_epoch'] >= 0 else 5}")
        print(f"  Best val loss: {results['best_val_loss']:.4f}")
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader, 'original', num_classes=2)
        print(f"✓ Evaluation completed")
        print(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete pipeline with minimal data."""
    print("\n" + "=" * 80)
    print("TEST 4: Full Pipeline (Minimal)")
    print("=" * 80)
    
    try:
        from probing.run_probing_experiment import run_probing_experiment
        
        print("\nRunning minimal probing experiment...")
        results = run_probing_experiment(
            model_name='google/gemma-2-2b',
            layer_index=8,
            normal_samples=10,
            reasoning_samples=5,
            label_type='binary',
            probe_type='linear',
            batch_size=4,
            num_epochs=5,
            patience=3,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            save_dir='data/probing/test_full_pipeline',
        )
        
        print(f"\n✓ Full pipeline test completed!")
        print(f"  Results for 3 activation types:")
        for act_type in ['original', 'reconstructed', 'residue']:
            metrics = results[act_type]['test_metrics']
            print(f"    {act_type}: F1 = {metrics['f1']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PROBING IMPLEMENTATION VALIDATION TESTS")
    print("=" * 80)
    print("\nThis will test the implementation with small samples.")
    print("Expected time: 2-5 minutes on GPU, longer on CPU.\n")
    
    # Track results
    test_results = {}
    
    # Test 1: Dataset Interface
    dataset, success = test_dataset_interface()
    test_results['dataset_interface'] = success
    
    # Test 2: Probe Model
    model, success = test_probe_model(dataset)
    test_results['probe_model'] = success
    
    # Test 3: Training
    success = test_training(dataset, model)
    test_results['training'] = success
    
    # Test 4: Full Pipeline (optional - can be slow)
    success = test_full_pipeline()
    test_results['full_pipeline'] = success
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in test_results.items():
        if result is None:
            status = "SKIPPED"
            symbol = "-"
        elif result:
            status = "PASSED"
            symbol = "✓"
        else:
            status = "FAILED"
            symbol = "✗"
        print(f"{symbol} {test_name.replace('_', ' ').title()}: {status}")
    
    # Overall result
    passed = sum(1 for r in test_results.values() if r is True)
    failed = sum(1 for r in test_results.values() if r is False)
    
    print("\n" + "=" * 80)
    if failed == 0:
        print("✓ ALL TESTS PASSED!")
        print("The implementation is ready to use.")
    else:
        print(f"✗ {failed} TEST(S) FAILED")
        print("Please check the errors above.")
    print("=" * 80)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
