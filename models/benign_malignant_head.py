import torch
import torch.nn as nn
import yaml
from pathlib import Path

def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class BenignMalignantHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        """
        Binary classifier for malignant vs benign lesions.
        
        Args:
            input_dim (int): Input feature dimension (SAM features)
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Number of output classes (2 for binary)
            dropout (float): Dropout probability
        """
        super().__init__()
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x (torch.Tensor): Input tensor of shape [B, input_dim]
        Returns:
            torch.Tensor: Output tensor of shape [B, output_dim]
        """
        return self.model(x)
        
    def compute_class_weights(self, class_counts):
        """Compute class weights based on class distribution.
        
        Args:
            class_counts: Dictionary mapping class indices to their counts in dataset
            
        Returns:
            Tensor of class weights for balanced training
        """
        total_samples = sum(class_counts.values())
        weights = torch.zeros(len(class_counts))
        for idx, count in class_counts.items():
            weights[idx] = total_samples / (len(class_counts) * count)
        return weights 