#!/usr/bin/env python3
import os
from pathlib import Path
import yaml

from training.train_parallel import train_parallel

ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = ROOT / 'config.yaml'


def _write_config(odin_enabled: bool, temperature: float, epsilon: float, lesion_loss_fn: str = 'lmf'):
    # Load existing config and update
    cfg = {}
    if CONFIG_YAML.exists():
        with open(CONFIG_YAML, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    # Ensure sections
    cfg.setdefault('odin', {})
    cfg['odin']['enabled'] = bool(odin_enabled)
    cfg['odin']['temperature'] = float(temperature)
    cfg['odin']['epsilon'] = float(epsilon)

    cfg['lesion_loss_fn'] = lesion_loss_fn
    cfg['focal_gamma'] = cfg.get('focal_gamma', 2.0)
    cfg['cb_beta'] = cfg.get('cb_beta', 0.9999)

    with open(CONFIG_YAML, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    # Common params
    experiment = 'exp1'
    wandb_project = 'odin_comparison_exp1'

    # 1) No-ODIN baseline
    _write_config(odin_enabled=False, temperature=1000.0, epsilon=0.001, lesion_loss_fn='lmf')
    train_parallel(
        experiment=experiment,
        skin_weight=1.0,
        lesion_weight=1.0,
        bm_weight=1.0,
        hidden_dims=(512, 256),
        dropout=0.3,
        batch_size=32,
        epochs=30,
        learning_rate=1e-3,
        weight_decay=1e-2,
        wandb_project=wandb_project,
        wandb_name='exp1_parallel_lmf_no_odin',
        log_file=str(ROOT / 'backend' / 'models' / 'parallel' / 'exp1_odin_comparison' / 'no_odin' / 'logs_parallel_exp1_lmf_no_odin.txt'),
        output_base_dir=str(ROOT / 'backend' / 'models' / 'parallel' / 'exp1_odin_comparison' / 'no_odin')
    )

    # 2) With ODIN
    _write_config(odin_enabled=True, temperature=1000.0, epsilon=0.001, lesion_loss_fn='lmf')
    train_parallel(
        experiment=experiment,
        skin_weight=1.0,
        lesion_weight=1.0,
        bm_weight=1.0,
        hidden_dims=(512, 256),
        dropout=0.3,
        batch_size=32,
        epochs=30,
        learning_rate=1e-3,
        weight_decay=1e-2,
        wandb_project=wandb_project,
        wandb_name='exp1_parallel_lmf_with_odin',
        log_file=str(ROOT / 'backend' / 'models' / 'parallel' / 'exp1_odin_comparison' / 'with_odin' / 'logs_parallel_exp1_lmf_with_odin.txt'),
        output_base_dir=str(ROOT / 'backend' / 'models' / 'parallel' / 'exp1_odin_comparison' / 'with_odin')
    )


if __name__ == '__main__':
    main()
