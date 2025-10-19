"""
Training and evaluation pipeline for probing experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import tqdm


class ProbeTrainer:
    """
    Trainer for probing models with early stopping and metric tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        min_delta: float = 1e-4,
    ):
        """
        Args:
            model: Probe model to train
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in validation loss to be considered improvement
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
        }
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.stopped_epoch = -1
    
    def train_epoch(self, dataloader: DataLoader, activation_type: str) -> dict:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            activation_type: Which activation to use ('original', 'reconstructed', 'residue')
        
        Returns:
            Dict with training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            # Get the appropriate activation type
            x = batch[activation_type].to(self.device)
            y = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, activation_type: str, num_classes: int = None) -> dict:
        """
        Evaluate the model.
        
        Args:
            dataloader: Validation/test data loader
            activation_type: Which activation to use ('original', 'reconstructed', 'residue')
            num_classes: Number of classes (for confusion matrix shape)
        
        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in dataloader:
            x = batch[activation_type].to(self.device)
            y = batch['label'].to(self.device)
            
            # Forward pass
            logits = self.model(x)
            loss = self.criterion(logits, y)
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # If num_classes is provided, ensure confusion matrix has correct shape
        if num_classes is not None:
            import numpy as np
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        else:
            cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        activation_type: str,
        num_epochs: int = 100,
        verbose: bool = True,
    ) -> dict:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            activation_type: Which activation to use ('original', 'reconstructed', 'residue')
            num_epochs: Maximum number of epochs
            verbose: Whether to print progress
        
        Returns:
            Dict with training history and final metrics
        """
        # Learning rate scheduler
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        if verbose:
            pbar = tqdm.tqdm(range(num_epochs), desc='Training')
        else:
            pbar = range(num_epochs)
        
        for epoch in pbar:
            # Train
            train_metrics = self.train_epoch(train_loader, activation_type)
            
            # Validate (pass num_classes for proper confusion matrix shape)
            val_metrics = self.evaluate(val_loader, activation_type, num_classes=self.model.num_classes if hasattr(self.model, 'num_classes') else None)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Update learning rate
            scheduler.step()
            
            # Progress bar update
            if verbose:
                pbar.set_postfix({
                    'train_loss': f"{train_metrics['loss']:.4f}",
                    'val_loss': f"{val_metrics['loss']:.4f}",
                    'val_acc': f"{val_metrics['accuracy']:.4f}",
                    'val_f1': f"{val_metrics['f1']:.4f}",
                })
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.patience:
                self.stopped_epoch = epoch
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'history': self.history,
            'stopped_epoch': self.stopped_epoch,
            'best_val_loss': self.best_val_loss,
        }
    
    def save(self, save_path: Path):
        """Save model and training state."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
        }, save_path)
    
    def load(self, load_path: Path):
        """Load model and training state."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']


def plot_training_curves(history: dict, save_path: Optional[Path] = None):
    """
    Plot training curves.
    
    Args:
        history: Training history dict
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 curve
    axes[2].plot(history['train_f1'], label='Train')
    axes[2].plot(history['val_f1'], label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score Curve')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path: Optional[Path] = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()


if __name__ == '__main__':
    # Test the trainer
    from probe_model import create_probe
    from torch.utils.data import TensorDataset
    
    # Create dummy data
    batch_size = 32
    num_samples = 1000
    input_dim = 2304
    num_classes = 2
    
    # Create dummy activations and labels
    original = torch.randn(num_samples, input_dim)
    reconstructed = torch.randn(num_samples, input_dim)
    residue = original - reconstructed
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Create dataset
    dataset = TensorDataset(original, reconstructed, residue, labels)
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    def collate_fn(batch):
        orig, recon, resid, lab = zip(*batch)
        return {
            'original': torch.stack(orig),
            'reconstructed': torch.stack(recon),
            'residue': torch.stack(resid),
            'label': torch.stack(lab),
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Create model and trainer
    model = create_probe(input_dim, num_classes, probe_type='linear')
    trainer = ProbeTrainer(model, device='cpu', patience=5)
    
    # Train
    print("Training probe on original activations...")
    results = trainer.train(train_loader, val_loader, activation_type='original', num_epochs=20)
    
    print(f"\nTraining completed at epoch {results['stopped_epoch'] + 1}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    
    # Evaluate
    final_metrics = trainer.evaluate(val_loader, 'original', num_classes=num_classes)
    print(f"\nFinal validation metrics:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {final_metrics['f1']:.4f}")
