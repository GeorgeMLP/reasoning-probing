"""
MLP probing models for activation classification.
"""

import torch
import torch.nn as nn
from typing import Optional


class MLPProbe(nn.Module):
    """
    Multi-layer perceptron probe for classifying activations.
    
    Args:
        input_dim: Dimension of input activation vector
        num_classes: Number of output classes (2 for binary, 6 for fine-grained reasoning types)
        hidden_dims: List of hidden layer dimensions. Empty list = linear probe.
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'tanh')
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: list[int] = [],
        dropout: float = 0.1,
        activation: str = 'relu',
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes  # Store for later use
        self.hidden_dims = hidden_dims
        
        # Choose activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn,
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input activations [batch, input_dim]
        
        Returns:
            Logits [batch, num_classes]
        """
        return self.model(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels.
        
        Args:
            x: Input activations [batch, input_dim]
        
        Returns:
            Predicted class indices [batch]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class probabilities.
        
        Args:
            x: Input activations [batch, input_dim]
        
        Returns:
            Class probabilities [batch, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


def create_probe(
    input_dim: int,
    num_classes: int = 2,
    probe_type: str = 'linear',
    **kwargs
) -> MLPProbe:
    """
    Factory function to create different types of probes.
    
    Args:
        input_dim: Input dimension
        num_classes: Number of output classes
        probe_type: Type of probe ('linear', 'mlp_1', 'mlp_2', 'mlp_3')
        **kwargs: Additional arguments for MLPProbe
    
    Returns:
        MLPProbe instance
    """
    if probe_type == 'linear':
        hidden_dims = []
    elif probe_type == 'mlp_1':
        hidden_dims = [512]
    elif probe_type == 'mlp_2':
        hidden_dims = [512, 256]
    elif probe_type == 'mlp_3':
        hidden_dims = [512, 256, 128]
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    
    return MLPProbe(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        **kwargs
    )


if __name__ == '__main__':
    # Test the probe models
    batch_size = 32
    input_dim = 2304  # Gemma-2-2b hidden dim
    
    print("=== Testing Linear Probe ===")
    linear_probe = create_probe(input_dim, num_classes=2, probe_type='linear')
    print(f"Linear probe: {linear_probe}")
    print(f"Parameters: {sum(p.numel() for p in linear_probe.parameters())}")
    
    x = torch.randn(batch_size, input_dim)
    output = linear_probe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    predictions = linear_probe.predict(x)
    print(f"Predictions shape: {predictions.shape}")
    probs = linear_probe.predict_proba(x)
    print(f"Probabilities shape: {probs.shape}")
    
    print("\n=== Testing MLP Probe ===")
    mlp_probe = create_probe(input_dim, num_classes=2, probe_type='mlp_2')
    print(f"MLP probe: {mlp_probe}")
    print(f"Parameters: {sum(p.numel() for p in mlp_probe.parameters())}")
    
    output = mlp_probe(x)
    print(f"Output shape: {output.shape}")
