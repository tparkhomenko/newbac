#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import yaml

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training.train_parallel import train_parallel
from training.train_multihead import train_multihead


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def set_config(lesion_loss_fn: str, mixup_enabled: bool, mixup_alpha: float = 0.2):
    cfg_path = project_root / 'config.yaml'
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    cfg['lesion_loss_fn'] = lesion_loss_fn
    bm = cfg.get('balanced_mixup', {}) or {}
    bm['enabled'] = bool(mixup_enabled)
    bm['alpha'] = float(mixup_alpha)
    cfg['balanced_mixup'] = bm
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def make_names(arch: str, dataset: str, loss: str, mixup: bool):
    dataset_tag = 'full' if dataset == 'full' else 'balanced'
    if mixup:
        run = f"exp1_{arch}_{loss}_mixup_{dataset_tag}"
        loss_dir = f"{loss}_mixup_{dataset_tag}"
        log_name = f"logs_{arch}_exp1_{loss}_mixup_{dataset_tag}.txt"
    else:
        run = f"exp1_{arch}_{loss}_{dataset_tag}"
        loss_dir = f"{loss}_{dataset_tag}"
        log_name = f"logs_{arch}_exp1_{loss}_{dataset_tag}.txt"
    return run, loss_dir, log_name


def train_one(architecture: str, dataset: str, loss: str, mixup: bool, *,
              epochs: int, batch_size: int, learning_rate: float, weight_decay: float,
              dropout: float, hidden_dims: Tuple[int, ...], wandb_project: str):
    experiment = 'exp1' if dataset == 'full' else 'exp1_balanced'
    wandb_name, loss_dir, log_name = make_names(architecture, dataset, loss, mixup)

    # Set config flags
    set_config(lesion_loss_fn='focal' if loss == 'focal' else 'weighted_ce', mixup_enabled=mixup, mixup_alpha=0.2)

    # Output dir (Tier 2)
    out_dir = project_root / 'backend' / 'models' / architecture / 'exp1_tier2_losses' / loss_dir
    ensure_dir(out_dir)
    log_file = out_dir / log_name

    if architecture == 'parallel':
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
    else:
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Tier 2: Exp1 advanced losses and mixup comparison')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--hidden_dims', type=str, default='512,256')
    parser.add_argument('--wandb_project', type=str, default='exp1_tier2_losses')
    parser.add_argument('--architecture', choices=['parallel', 'multi'])
    parser.add_argument('--dataset', choices=['full', 'balanced'])
    parser.add_argument('--loss', choices=['weightedce', 'focal'])
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--all', action='store_true', help='Run all 16 experiments')
    args = parser.parse_args()

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(',') if x)

    configs = []
    if args.all:
        for arch in ['parallel', 'multi']:
            for dataset in ['full', 'balanced']:
                for loss in ['weightedce', 'focal']:
                    for mix in [False, True]:
                        configs.append((arch, dataset, loss, mix))
    else:
        if not (args.architecture and args.dataset and args.loss is not None):
            parser.print_help()
            print('\nUse --all for the full 16-run sweep or specify a single config.')
            sys.exit(1)
        configs.append((args.architecture, args.dataset, args.loss, args.mixup))

    for arch, dataset, loss, mix in configs:
        print('\n' + '='*60)
        print(f"Starting: arch={arch}, dataset={dataset}, loss={loss}, mixup={mix}")
        print('='*60)
        train_one(
            architecture=arch,
            dataset=dataset,
            loss=loss,
            mixup=mix,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden_dims=hidden_dims,
            wandb_project=args.wandb_project,
        )


if __name__ == '__main__':
    main()








