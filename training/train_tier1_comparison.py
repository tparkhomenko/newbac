#!/usr/bin/env python3
"""
Tier 1 Training Script: Core comparison experiments with weighted CrossEntropy loss on Exp1 dataset.

This script establishes the baseline performance and the effect of balancing by training:
- Parallel vs Multi architectures
- Full vs Balanced Exp1 datasets
- All with weighted CrossEntropy loss
"""

import os
import sys
import time
import math
import gc
import logging
from io import TextIOBase
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import wandb
import numpy as np
import yaml

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.parallel_unified_dataset import ParallelUnifiedDataset
from models.multitask_model import MultiTaskHead
from training.train_parallel import train_parallel
from training.train_multihead import train_multihead

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


class _TeeStream(TextIOBase):
    """Tee stdout/stderr to both console and a file.

    This captures tqdm/progress bars and regular prints into the log file
    while still showing them live in the console.
    """

    def __init__(self, original_stream: TextIOBase, file_obj: TextIOBase):
        self.original_stream = original_stream
        self.file_obj = file_obj

    def write(self, data: str) -> int:
        # Write to console
        try:
            self.original_stream.write(data)
        except Exception:
            pass
        # Write to file
        try:
            self.file_obj.write(data)
            # Flush eagerly to ensure progress appears in file promptly
            self.file_obj.flush()
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        try:
            self.original_stream.flush()
        except Exception:
            pass
        try:
            self.file_obj.flush()
        except Exception:
            pass

def train_tier1_comparison(
    architecture: str = 'parallel',  # 'parallel' or 'multi'
    dataset: str = 'full',  # 'full' or 'balanced'
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    dropout: float = 0.3,
    hidden_dims: Tuple[int, ...] = (512, 256),
    wandb_project: str = 'exp1_comparison',
    wandb_name: Optional[str] = None,
):
    """
    Train a model for the Tier 1 comparison.
    
    Args:
        architecture: 'parallel' or 'multi'
        dataset: 'full' or 'balanced'
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay
        dropout: Dropout rate
        hidden_dims: Hidden dimensions for the model
        wandb_project: WandB project name
        wandb_name: WandB run name
    """
    
    # Determine experiment and run name
    experiment = 'exp1' if dataset == 'full' else 'exp1_balanced'
    if not wandb_name:
        wandb_name = f"exp1_{architecture}_weightedce_{dataset}"
    
    logger.info(f"=== Training {architecture} architecture on {dataset} Exp1 dataset ===")
    logger.info(f"Experiment: {experiment}")
    logger.info(f"WandB Project: {wandb_project}")
    logger.info(f"WandB Run: {wandb_name}")
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = project_root / 'backend' / 'models' / f'{architecture}' / 'exp1_comparison' / f'exp1_weightedce_{dataset}'
    ensure_dir(output_dir)
    
    # Set up logging
    log_file = output_dir / f'logs_{architecture}_exp1_weightedce_{dataset}.txt'
    
    # Redirect logging to both file and console
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Tee stdout/stderr so tqdm/progress bars and prints are captured in the file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_fp = open(log_file, 'a', buffering=1)
    sys.stdout = _TeeStream(original_stdout, log_fp)
    sys.stderr = _TeeStream(original_stderr, log_fp)

    try:
        if architecture == 'parallel':
            # Train parallel architecture
            train_parallel(
                experiment=experiment,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                dropout=dropout,
                hidden_dims=hidden_dims,
                wandb_project=wandb_project,
                wandb_name=wandb_name,
                log_file=str(log_file)
            )
        elif architecture == 'multi':
            # Train multi-head architecture
            train_multihead(
                experiment=experiment,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                dropout=dropout,
                hidden_dims=hidden_dims,
                wandb_project=wandb_project,
                wandb_name=wandb_name,
                log_file=str(log_file)
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
            
        logger.info(f"Training completed successfully for {architecture} on {dataset} dataset")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Remove handlers to avoid duplicate logging
        logger.removeHandler(file_handler)
        logger.removeHandler(console_handler)
        file_handler.close()
        # Restore std streams and close file
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        except Exception:
            pass
        try:
            log_fp.close()
        except Exception:
            pass

def main():
    """Main function to run all Tier 1 comparison experiments."""
    
    # Configuration
    config = {
        'epochs': 30,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'dropout': 0.3,
        'hidden_dims': (512, 256),
        'wandb_project': 'exp1_comparison'
    }
    
    # Experiments to run
    experiments = [
        ('parallel', 'full'),
        ('parallel', 'balanced'),
        ('multi', 'full'),
        ('multi', 'balanced')
    ]
    
    logger.info("=== Starting Tier 1 Comparison Experiments ===")
    logger.info(f"Configuration: {config}")
    logger.info(f"Experiments: {experiments}")
    
    # Run each experiment
    for arch, dataset in experiments:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting: {arch} architecture on {dataset} dataset")
            logger.info(f"{'='*60}")
            
            train_tier1_comparison(
                architecture=arch,
                dataset=dataset,
                **config
            )
            
            logger.info(f"✅ Completed: {arch} architecture on {dataset} dataset")
            
        except Exception as e:
            logger.error(f"❌ Failed: {arch} architecture on {dataset} dataset")
            logger.error(f"Error: {e}")
            continue
    
    logger.info("\n=== Tier 1 Comparison Experiments Completed ===")
    
    # Print summary
    logger.info("\nSummary of completed experiments:")
    for arch, dataset in experiments:
        output_dir = project_root / 'backend' / 'models' / f'{arch}' / 'exp1_comparison' / f'exp1_weightedce_{dataset}'
        if output_dir.exists():
            logger.info(f"✅ {arch} on {dataset}: {output_dir}")
        else:
            logger.info(f"❌ {arch} on {dataset}: Failed")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tier 1 Training: Core comparison experiments')
    parser.add_argument('--architecture', choices=['parallel', 'multi'], help='Architecture to train')
    parser.add_argument('--dataset', choices=['full', 'balanced'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--hidden_dims', type=str, default='512,256', help='Hidden dimensions (comma-separated)')
    parser.add_argument('--wandb_project', type=str, default='exp1_comparison', help='WandB project name')
    parser.add_argument('--wandb_name', type=str, help='WandB run name')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    
    args = parser.parse_args()
    
    # Parse hidden dimensions
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(',') if x)
    
    if args.all:
        # Run all experiments
        main()
    elif args.architecture and args.dataset:
        # Run single experiment
        train_tier1_comparison(
            architecture=args.architecture,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden_dims=hidden_dims,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name
        )
    else:
        parser.print_help()
        print("\nUse --all to run all experiments, or specify --architecture and --dataset for a single run.")

