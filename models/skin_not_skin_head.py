import torch
import torch.nn as nn
import yaml
from pathlib import Path

def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class SkinNotSkinClassifier(nn.Module):
    """MLP1: Classifier for distinguishing between skin and non-skin images."""
    
    def __init__(self):
        super().__init__()
        
        # Load model configuration
        config = load_config()
        model_config = config['models']['mlp1']
        
        # Get dimensions from config
        input_dim = model_config['input_dim']
        hidden_dims = model_config['hidden_dims']
        output_dim = model_config['output_dim']
        dropout_rate = model_config['dropout']
        
        # Build MLP layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        # Add final classification layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        return self.mlp(x)