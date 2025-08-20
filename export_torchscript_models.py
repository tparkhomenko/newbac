#!/usr/bin/env python3
"""
Export multihead models to TorchScript format for backend deployment.
Exports exp1, exp2, exp3 multihead models as mlp1.pt, mlp2.pt, mlp3.pt
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
    
    # Create model
    model = MultiTaskHead(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
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
    preprocess_size = int(os.getenv('PREPROCESS_SIZE', '256'))
    preprocess_mean = float(os.getenv('PREPROCESS_MEAN', '0.0'))
    preprocess_std = float(os.getenv('PREPROCESS_STD', '1.0'))
    
    print(f"Environment variables:")
    print(f"  MODEL_DIR: {model_dir}")
    print(f"  TORCH_DEVICE: {torch_device}")
    print(f"  PREPROCESS_SIZE: {preprocess_size}")
    print(f"  PREPROCESS_MEAN: {preprocess_mean}")
    print(f"  PREPROCESS_STD: {preprocess_std}")
    
    # Setup device
    device = torch.device(torch_device)
    print(f"Using device: {device}")
    
    # Create output directory
    export_dir = Path(model_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {export_dir}")
    
    # Model mapping: experiment -> output filename
    model_mapping = {
        'exp1': 'mlp1.pt',
        'exp2': 'mlp2.pt', 
        'exp3': 'mlp3.pt',
        'exp1_parallel': 'mlp4.pt'
    }
    
    # Model paths mapping
    model_paths = {
        'exp1': 'models/exp1_multihead/20250820_134612/best.pt',
        'exp2': 'models/exp2_multihead/20250820_092326/best.pt',
        'exp3': 'models/exp3_multihead/20250820_092253/best.pt',
        'exp1_parallel': 'models/exp1_parallel/20250820_141116/best.pt'
    }
    
    # Export each model
    for exp_name, output_filename in model_mapping.items():
        print(f"\nExporting {exp_name} to {output_filename}...")
        
        # Use the specific path for each model
        checkpoint_path = Path(model_paths[exp_name])
        
        if not checkpoint_path.exists():
            print(f"  ❌ Checkpoint not found: {checkpoint_path}")
            continue
            
        try:
            # Load and export model
            model = load_model_from_checkpoint(checkpoint_path, device)
            export_path = export_dir / output_filename
            
            # Export to TorchScript
            export_to_torchscript(model, export_path, device)
            print(f"  ✅ Successfully exported to: {export_path}")
            
        except Exception as e:
            print(f"  ❌ Failed to export {exp_name}: {e}")
            continue
    
    print(f"\nExport complete! Models saved to: {export_dir}")
    
    # List exported files
    print("\nExported files:")
    for file in export_dir.glob('*.pt'):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name}: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
