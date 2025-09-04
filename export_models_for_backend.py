#!/usr/bin/env python3
"""
Export trained PyTorch models to TorchScript format for backend deployment.
This script loads the trained checkpoints and exports them as TorchScript models.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.multitask_model import MultiTaskHead


class TorchScriptMultiTaskWrapper(torch.nn.Module):
    """Wrapper class that returns tuple outputs for TorchScript compatibility."""
    
    def __init__(self, model: MultiTaskHead):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        return outputs['skin'], outputs['lesion'], outputs['bm']


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> MultiTaskHead:
    """Load model from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    input_dim = config.get('input_dim', 256)
    hidden_dims = config.get('hidden_dims', (512, 256))
    dropout = config.get('dropout', 0.3)
    
    # Create model with 8-class lesion head
    model = MultiTaskHead(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_classes_skin=2,
        num_classes_lesion=8,  # 8 fine-grained classes
        num_classes_bm=2
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def export_to_torchscript(model: MultiTaskHead, output_path: Path, device: torch.device):
    """Export model to TorchScript format."""
    # Create wrapper for TorchScript compatibility
    wrapper = TorchScriptMultiTaskWrapper(model)
    wrapper.eval()
    
    # Use script instead of trace for better compatibility
    scripted_model = torch.jit.script(wrapper)
    
    # Save TorchScript model
    scripted_model.save(str(output_path))
    print(f"Exported TorchScript model to: {output_path}")
    
    # Test the exported model
    dummy_input = torch.randn(1, 256, device=device)
    with torch.no_grad():
        skin_out, lesion_out, bm_out = scripted_model(dummy_input)
        print(f"  Test inference successful:")
        print(f"    Skin output shape: {skin_out.shape}")
        print(f"    Lesion output shape: {lesion_out.shape}")
        print(f"    BM output shape: {bm_out.shape}")


def main():
    # Environment variables
    model_dir = os.getenv('MODEL_DIR', 'backend/models')
    torch_device = os.getenv('TORCH_DEVICE', 'cpu')
    
    print(f"Environment variables:")
    print(f"  MODEL_DIR: {model_dir}")
    print(f"  TORCH_DEVICE: {torch_device}")
    
    # Setup device
    device = torch.device(torch_device)
    print(f"Using device: {device}")
    
    # Model paths mapping
    model_paths = {
        'multi': {
            'mlp1': 'backend/models/multi/mlp1.pt',
            'mlp2': 'backend/models/multi/mlp2.pt',
            'mlp3': 'backend/models/multi/mlp3.pt'
        },
        'parallel': {
            'mlp1': 'backend/models/parallel/mlp1.pt',
            'mlp2': 'backend/models/parallel/mlp2.pt',
            'mlp3': 'backend/models/parallel/mlp3.pt'
        }
    }
    
    # Export each model
    for architecture in ['multi', 'parallel']:
        print(f"\n=== Exporting {architecture} architecture ===")
        
        for mlp_name in ['mlp1', 'mlp2', 'mlp3']:
            print(f"\nExporting {mlp_name}...")
            
            # Use the specific path for each model
            checkpoint_path = Path(model_paths[architecture][mlp_name])
            
            if not checkpoint_path.exists():
                print(f"  ❌ Checkpoint not found: {checkpoint_path}")
                continue
                
            try:
                # Load and export model
                model = load_model_from_checkpoint(checkpoint_path, device)
                
                # Export to TorchScript in the same directory
                export_path = checkpoint_path.parent / f"{mlp_name}_torchscript.pt"
                
                # Export to TorchScript
                export_to_torchscript(model, export_path, device)
                print(f"  ✅ Successfully exported to: {export_path}")
                
                # Replace the original checkpoint with TorchScript model
                os.replace(export_path, checkpoint_path)
                print(f"  ✅ Replaced original checkpoint with TorchScript model")
                
            except Exception as e:
                print(f"  ❌ Failed to export {mlp_name}: {e}")
                continue
    
    print(f"\n🎉 Export complete! All models are now in TorchScript format.")
    print(f"Models are ready for backend deployment in: {model_dir}")
    
    # List exported files
    print("\nExported files:")
    for architecture in ['multi', 'parallel']:
        print(f"\n{architecture.upper()} architecture:")
        for mlp_name in ['mlp1', 'mlp2', 'mlp3']:
            model_path = Path(model_paths[architecture][mlp_name])
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"  {mlp_name}.pt: {size_mb:.1f} MB (TorchScript)")


if __name__ == '__main__':
    main()
