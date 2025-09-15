import os
import sys
import math
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import wandb
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.parallel_unified_dataset import ParallelUnifiedDataset
from models.multitask_model import MultiTaskHead
from datasets.balanced_mixup import BalancedMixupDataset
from utils.odin import compute_odin_scores, compute_msp_scores


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compute_split_class_counts(metadata_csv: Path, experiment_col: str = 'exp1') -> Dict[str, Dict[str, int]]:
    import pandas as pd
    df = pd.read_csv(metadata_csv)
    class_names = ['UNKNOWN', 'NV', 'MEL', 'BCC', 'BKL', 'AKIEC', 'NOT_SKIN', 'SCC', 'VASC', 'DF']
    stats: Dict[str, Dict[str, int]] = {}
    for split in ['train', 'val', 'test']:
        subset = df[df[experiment_col] == split]
        counts = subset['unified_diagnosis'].astype(str).str.upper().value_counts().reindex(class_names, fill_value=0)
        stats[split] = counts.to_dict()
    return stats


def save_class_distribution_bar(stats: Dict[str, Dict[str, int]], out_path: Path, title: str):
    class_names = ['UNKNOWN', 'NV', 'MEL', 'BCC', 'BKL', 'AKIEC', 'NOT_SKIN', 'SCC', 'VASC', 'DF']
    splits = ['train', 'val', 'test']
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, split in enumerate(splits):
        counts = [stats.get(split, {}).get(cls, 0) for cls in class_names]
        ax.bar(x + (i - 1) * width, counts, width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylabel('Count')
    ax.legend()
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches='tight')
    plt.close(fig)


def print_split_counts(stats: Dict[str, Dict[str, int]]):
    class_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NOT_SKIN', 'NV', 'SCC', 'UNKNOWN', 'VASC']
    for split in ['train', 'val', 'test']:
        dist = stats.get(split, {})
        for cls in class_names:
            cnt = dist.get(cls, 0)
            print(f"wandb: {split}_count/{cls} {cnt}")


def compute_task_class_weights(dataset: ParallelUnifiedDataset, device: torch.device) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, Dict[int, int]]]:
    counts: Dict[str, Dict[int, int]] = {
        'skin': {0: 0, 1: 0},
        'lesion': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},  # 8 fine-grained lesion classes (no NOT_SKIN)
        'bm': {0: 0, 1: 0},
    }
    # Iterate once through dataset to gather counts using masks
    for idx in range(len(dataset)):
        _, labels, masks, _ = dataset[idx]
        for task in ('skin', 'lesion', 'bm'):
            if masks[task].item() == 1:
                label_idx = int(labels[task].item())
                if label_idx in counts[task]:
                    counts[task][label_idx] += 1

    weights: Dict[str, Optional[torch.Tensor]] = {}
    for task, class_counts in counts.items():
        total = sum(class_counts.values())
        if total == 0:
            weights[task] = None
            continue
        num_classes = len(class_counts)
        w = torch.zeros(num_classes, dtype=torch.float32, device=device)
        for cls_idx, cnt in class_counts.items():
            if cnt > 0:
                w[cls_idx] = total / (num_classes * cnt)
            else:
                # If class absent, set weight to 0 so it doesn't affect loss
                w[cls_idx] = 0.0
        weights[task] = w
    return weights, counts
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class CBFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        alpha_t = self.alpha.gather(0, targets)
        loss = -alpha_t * ((1 - pt) ** self.gamma) * log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        return loss.mean()


def plot_and_save_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], title: str, save_path: Path):
    if len(y_true) == 0:
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float32)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names, title=title, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fmt = '.2f'
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, format(cm_norm[i, j], fmt), ha='center', va='center', color='white' if cm_norm[i, j] > thresh else 'black')
    fig.tight_layout()
    ensure_dir(save_path.parent)
    plt.savefig(str(save_path), bbox_inches='tight')
    plt.close(fig)


def save_confusion_matrix_raw(y_true: List[int], y_pred: List[int], class_names: List[str], title: str, save_path: Path):
    if len(y_true) == 0:
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names, title=title, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    ensure_dir(save_path.parent)
    plt.savefig(str(save_path), bbox_inches='tight')
    plt.close(fig)


def save_per_class_metrics_bar(y_true: List[int], y_pred: List[int], class_names: List[str], out_path: Path, title: str):
    if len(y_true) == 0:
        return
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    x = np.arange(len(class_names))
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, prec, width, label='Precision')
    plt.bar(x, rec, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim(0.0, 1.0)
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()


def save_test_metrics_bar(acc: float, f1_macro: float, out_path: Path, title: str):
    fig = plt.figure(figsize=(4, 3))
    bars = plt.bar(['Accuracy', 'F1 (macro)'], [acc, f1_macro], color=['#1f77b4', '#ff7f0e'])
    plt.ylim(0.0, 1.0)
    for b, val in zip(bars, [acc, f1_macro]):
        plt.text(b.get_x() + b.get_width()/2, val + 0.02, f"{val:.3f}", ha='center', va='bottom', fontsize=10)
    plt.title(title)
    plt.ylabel('Score')
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), bbox_inches='tight')
    plt.close(fig)


def save_loss_accuracy_curves_lesion(metrics_csv: Path, out_path: Path):
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    epochs = df['epoch'].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Accuracy subplot for lesion head
    axes[0].plot(epochs, df.get('train_lesion_acc', pd.Series([0]*len(epochs))).values, label='Train Acc')
    axes[0].plot(epochs, df.get('val_lesion_acc', pd.Series([0]*len(epochs))).values, label='Val Acc')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].set_title('Accuracy (Train vs Val)'); axes[0].legend()
    # Loss subplot
    axes[1].plot(epochs, df['train_loss'].values, label='Train Loss')
    axes[1].plot(epochs, df['val_loss'].values, label='Val Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].set_title('Loss (Train vs Val)'); axes[1].legend()
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), bbox_inches='tight')
    plt.close(fig)


def train_multihead(
        batch_size: int = 64,
        epochs: int = 40,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        hidden_dims: Tuple[int, int] = (512, 256),
        skin_weight: float = 1.0,
        lesion_weight: float = 1.0,
        bm_weight: float = 1.0,
        wandb_project: str = 'skin-lesion-classification',
        wandb_name: Optional[str] = None,
        num_workers: int = 0,
        experiment: str = 'exp1',
        log_file: Optional[str] = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up file logging if log_file is provided
    if log_file:
        import logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    # Output dirs
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_tag = f"{experiment}_multihead"
    # For exp6 runs, save under Tier 5 paths
    if experiment == 'exp6':
        checkpoints_dir = project_root / 'backend' / 'models' / 'multi' / 'exp6_tier5_cb_focal' / run_id
        logs_dir = project_root / 'backend' / 'models' / 'multi'
        ensure_dir(logs_dir)
    else:
        checkpoints_dir = project_root / 'models' / exp_tag / run_id
        logs_dir = project_root / 'logs' / exp_tag / run_id
    cm_dir_val = logs_dir / 'confusion_matrices' / 'val'
    cm_dir_test = logs_dir / 'confusion_matrices' / 'test'
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)
    ensure_dir(cm_dir_val)
    ensure_dir(cm_dir_test)

    # Datasets and loaders
    train_dataset = ParallelUnifiedDataset(experiment=experiment, split='train')
    val_dataset = ParallelUnifiedDataset(experiment=experiment, split='val')
    test_dataset = ParallelUnifiedDataset(experiment=experiment, split='test')

    # Optional Balanced-MixUp and ODIN
    balanced_mixup_cfg = {'enabled': False, 'alpha': 0.2}
    odin_cfg = {'enabled': False, 'temperature': 1000.0, 'epsilon': 0.001}
    config_path = project_root / 'config.yaml'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            bm = cfg.get('balanced_mixup', {}) or {}
            balanced_mixup_cfg['enabled'] = bool(bm.get('enabled', False))
            balanced_mixup_cfg['alpha'] = float(bm.get('alpha', 0.2))
            odin_in = cfg.get('odin', {}) or {}
            odin_cfg['enabled'] = bool(odin_in.get('enabled', False))
            odin_cfg['temperature'] = float(odin_in.get('temperature', 1000.0))
            odin_cfg['epsilon'] = float(odin_in.get('epsilon', 0.001))
        except Exception as e:
            print(f"[WARN] Failed to read balanced_mixup from config.yaml: {e}")
    if balanced_mixup_cfg['enabled']:
        print(f"Using Balanced MixUp augmentation (alpha={balanced_mixup_cfg['alpha']})")
        train_dataset = BalancedMixupDataset(train_dataset, num_lesion_classes=8, alpha=balanced_mixup_cfg['alpha'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model - ensure lesion head has 8 classes for fine-grained classification (no NOT_SKIN)
    model = MultiTaskHead(
        input_dim=256, 
        hidden_dims=hidden_dims, 
        dropout=dropout,
        num_classes_skin=2,
        num_classes_lesion=8,  # 8 fine-grained lesion classes (no NOT_SKIN)
        num_classes_bm=2
    )
    model = model.to(device)

    # Class weights per task (use base dataset if wrapped by BalancedMixupDataset)
    base_for_counts = train_dataset.base if hasattr(train_dataset, 'base') else train_dataset
    task_class_weights, class_counts = compute_task_class_weights(base_for_counts, device=device)
    
    # Read global config to select lesion loss function and curriculum
    config_path = project_root / 'config.yaml'
    lesion_loss_fn = 'weighted_ce'
    focal_gamma: float = 2.0
    cb_beta: float = 0.9999
    curriculum_cfg = {'enabled': False, 'lesion_start': 1.0, 'lesion_end': 1.0, 'bm_start': 1.0, 'bm_end': 1.0, 'epochs': 20}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            lesion_loss_fn = str(cfg.get('lesion_loss_fn', lesion_loss_fn)).strip().lower()
            focal_gamma = float(cfg.get('focal_gamma', focal_gamma))
            cb_beta = float(cfg.get('cb_beta', cb_beta))
            cur_in = cfg.get('curriculum', {}) or {}
            if bool(cur_in.get('enabled', False)):
                curriculum_cfg.update({
                    'enabled': True,
                    'lesion_start': float(cur_in.get('lesion_start', 0.5)),
                    'lesion_end': float(cur_in.get('lesion_end', 1.0)),
                    'bm_start': float(cur_in.get('bm_start', 0.5)),
                    'bm_end': float(cur_in.get('bm_end', 1.0)),
                    'epochs': int(cur_in.get('epochs', 20)),
                })
        except Exception as e:
            print(f"[WARN] Failed to read config.yaml: {e}")

    # Print class counts and weights for debugging
    print("Lesion class counts:", [class_counts['lesion'].get(i, 0) for i in range(8)])
    print("Lesion class weights:", task_class_weights['lesion'].tolist() if task_class_weights['lesion'] is not None else "None")
    print(f"Using lesion_loss_fn: {lesion_loss_fn}")

    # Select lesion criterion
    if lesion_loss_fn == 'focal':
        lesion_criterion = FocalLoss(alpha=task_class_weights['lesion'], gamma=focal_gamma, reduction='mean')
        print(f"Focal alpha: {task_class_weights['lesion'].detach().cpu().tolist() if isinstance(task_class_weights['lesion'], torch.Tensor) else None}, gamma: {focal_gamma}")
    elif lesion_loss_fn == 'lmf':
        from losses.lmf_loss import LMFLoss
        # Get class counts for LMF loss without overwriting the dict used later
        lesion_class_counts_list = [class_counts['lesion'].get(i, 0) for i in range(8)]
        lesion_criterion = LMFLoss(cls_num_list=lesion_class_counts_list, alpha=0.5, beta=0.5, gamma=2.0)
        print(f"LMF loss with alpha=0.5, beta=0.5, gamma=2.0")
    elif lesion_loss_fn == 'cb_focal':
        # Class-balanced alpha from counts using beta
        lesion_class_counts_list = [class_counts['lesion'].get(i, 0) for i in range(8)]
        counts_np = np.array(lesion_class_counts_list, dtype=np.float64)
        effective_num = 1.0 - np.power(cb_beta, counts_np)
        weights_cb = (1.0 - cb_beta) / np.maximum(effective_num, 1e-12)
        weights_cb = weights_cb / weights_cb.sum()
        alpha_cb = torch.tensor(weights_cb, dtype=torch.float32, device=device)
        lesion_criterion = CBFocalLoss(alpha=alpha_cb, gamma=focal_gamma)
        print(f"CB-Focal alpha: {weights_cb.tolist()}, beta: {cb_beta}, gamma: {focal_gamma}")
    else:
        lesion_criterion = nn.CrossEntropyLoss(weight=task_class_weights['lesion']) if task_class_weights['lesion'] is not None else nn.CrossEntropyLoss()

    # Losses
    criteria: Dict[str, nn.Module] = {
        'skin': nn.CrossEntropyLoss(weight=task_class_weights['skin']) if task_class_weights['skin'] is not None else nn.CrossEntropyLoss(),
        'lesion': lesion_criterion,
        'bm': nn.CrossEntropyLoss(weight=task_class_weights['bm']) if task_class_weights['bm'] is not None else nn.CrossEntropyLoss(),
    }
    task_weights = {'skin': skin_weight, 'lesion': lesion_weight, 'bm': bm_weight}

    # Optimizer & scheduler (Adam, CosineAnnealingLR)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # wandb
    if wandb_name is None:
        wandb_name = f'{exp_tag}_{run_id}'
    if balanced_mixup_cfg['enabled'] and wandb_name is not None and not wandb_name.endswith('_balanced_mixup'):
        wandb_name = f"{wandb_name}_balanced_mixup"
    wandb.init(project=wandb_project, name=wandb_name, config={
        'experiment': experiment,
        'architecture': 'shared_trunk_multi_head',
        'hidden_dims': list(hidden_dims),
        'dropout': dropout,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': learning_rate,
        'weight_decay': weight_decay,
        'task_weights': task_weights,
        'class_weights': {k: (v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else None) for k, v in task_class_weights.items()},
        'lesion_class_counts': [class_counts['lesion'].get(i, 0) for i in range(8)]
        ,
        'lesion_loss_fn': lesion_loss_fn,
        'focal_gamma': focal_gamma if lesion_loss_fn in ('focal','cb_focal') else None,
        'cb_beta': cb_beta if lesion_loss_fn == 'cb_focal' else None,
        'balanced_mixup': balanced_mixup_cfg,
        'curriculum': curriculum_cfg,
        'odin': odin_cfg,
    })
    wandb.watch(model, log='all', log_freq=10)

    # Log class counts and weights for lesion task
    print(f"Lesion class counts: {[class_counts['lesion'].get(i, 0) for i in range(8)]}")
    print(f"Lesion class weights: {task_class_weights['lesion'].tolist() if task_class_weights['lesion'] is not None else 'None'}")

    # CSV logger
    # For exp6, maintain CSV summary in backend/experiments_summary.csv too
    csv_path = logs_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'val_loss',
            'train_skin_acc', 'train_lesion_acc', 'train_bm_acc',
            'val_skin_acc', 'val_lesion_acc', 'val_bm_acc',
            'val_skin_f1_macro', 'val_lesion_f1_macro', 'val_bm_f1_macro',
            'lr'
        ])

    best_val_loss = math.inf
    best_checkpoint_path = checkpoints_dir / 'best.pt'

    def compute_masked_metrics(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {t: {} for t in ('skin', 'lesion', 'bm')}
        y_true: Dict[str, List[int]] = {t: [] for t in ('skin', 'lesion', 'bm')}
        y_pred: Dict[str, List[int]] = {t: [] for t in ('skin', 'lesion', 'bm')}
        for task in ('skin', 'lesion', 'bm'):
            mask = masks[task]
            valid_idx = mask.nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            preds = outputs[task].argmax(dim=1)[valid_idx].detach().cpu().numpy().tolist()
            tgts = labels[task][valid_idx].detach().cpu().numpy().tolist()
            y_true[task].extend(tgts)
            y_pred[task].extend(preds)
        for task in ('skin', 'lesion', 'bm'):
            if len(y_true[task]) > 0:
                metrics[task]['acc'] = accuracy_score(y_true[task], y_pred[task])
                metrics[task]['f1_macro'] = f1_score(y_true[task], y_pred[task], average='macro')
            else:
                metrics[task]['acc'] = 0.0
                metrics[task]['f1_macro'] = 0.0
        return metrics

    # Training loop
    for epoch in range(1, epochs + 1):
        # Curriculum weights per epoch
        cur_task_weights = task_weights.copy()
        if curriculum_cfg['enabled']:
            t = min(1.0, (epoch - 1) / max(1, curriculum_cfg['epochs']))
            cur_task_weights['lesion'] = curriculum_cfg['lesion_start'] + (curriculum_cfg['lesion_end'] - curriculum_cfg['lesion_start']) * t
            cur_task_weights['bm'] = curriculum_cfg['bm_start'] + (curriculum_cfg['bm_end'] - curriculum_cfg['bm_start']) * t
        wandb.log({'curriculum_lesion_w': cur_task_weights.get('lesion', 1.0), 'curriculum_bm_w': cur_task_weights.get('bm', 1.0), 'epoch': epoch})
        model.train()
        train_losses: List[float] = []
        train_collect = {'skin': {'preds': [], 'tgts': []}, 'lesion': {'preds': [], 'tgts': []}, 'bm': {'preds': [], 'tgts': []}}
        for features, labels, masks, _ in tqdm(train_loader, desc=f'Epoch {epoch} - train'):
            features = features.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                outputs = model(features)
                loss_total = 0.0
                for task in ('skin', 'lesion', 'bm'):
                    mask = masks[task]
                    valid_idx = mask.nonzero(as_tuple=True)[0]
                    if valid_idx.numel() == 0:
                        continue
                    logits = outputs[task][valid_idx]
                    if task == 'lesion' and isinstance(labels['lesion'], torch.Tensor) and labels['lesion'].dtype.is_floating_point:
                        targets_soft = labels['lesion'][valid_idx]
                        log_probs = torch.log_softmax(logits, dim=1)
                        loss_task = -(targets_soft * log_probs).sum(dim=1).mean()
                    else:
                        targets = labels[task][valid_idx]
                        loss_task = criteria[task](logits, targets)
                    loss_total = loss_total + cur_task_weights[task] * loss_task
            loss_total.backward()
            optimizer.step()
            train_losses.append(loss_total.item())

            # Train metrics collection
            for task in ('skin', 'lesion', 'bm'):
                mask = masks[task]
                valid_idx = mask.nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                pred = outputs[task].argmax(dim=1)[valid_idx].detach().cpu().numpy().tolist()
                # If lesion uses soft labels (mixup), harden for metrics
                if task == 'lesion' and isinstance(labels['lesion'], torch.Tensor) and labels['lesion'].dtype.is_floating_point:
                    trg_tensor = labels['lesion'][valid_idx]
                    trg = trg_tensor.argmax(dim=1).detach().cpu().numpy().tolist()
                else:
                    trg = labels[task][valid_idx].detach().cpu().numpy().tolist()
                train_collect[task]['preds'].extend(pred)
                train_collect[task]['tgts'].extend(trg)

        train_log: Dict[str, float] = {'epoch': epoch, 'train_loss': float(sum(train_losses) / max(1, len(train_losses)))}
        for task in ('skin', 'lesion', 'bm'):
            preds = train_collect[task]['preds']
            tgts = train_collect[task]['tgts']
            if len(preds) > 0:
                train_log[f'train_{task}_acc'] = accuracy_score(tgts, preds)
            else:
                train_log[f'train_{task}_acc'] = 0.0

        # Validation
        model.eval()
        val_losses: List[float] = []
        val_preds: Dict[str, List[int]] = {'skin': [], 'lesion': [], 'bm': []}
        val_tgts: Dict[str, List[int]] = {'skin': [], 'lesion': [], 'bm': []}
        with torch.no_grad():
            for features, labels, masks, _ in tqdm(val_loader, desc=f'Epoch {epoch} - val'):
                features = features.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                masks = {k: v.to(device) for k, v in masks.items()}
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(features)
                    loss_total = 0.0
                    for task in ('skin', 'lesion', 'bm'):
                        mask = masks[task]
                        valid_idx = mask.nonzero(as_tuple=True)[0]
                        if valid_idx.numel() == 0:
                            continue
                        logits = outputs[task][valid_idx]
                        if task == 'lesion' and isinstance(labels['lesion'], torch.Tensor) and labels['lesion'].dtype.is_floating_point:
                            targets_soft = labels['lesion'][valid_idx]
                            log_probs = torch.log_softmax(logits, dim=1)
                            loss_task = -(targets_soft * log_probs).sum(dim=1).mean()
                        else:
                            targets = labels[task][valid_idx]
                            loss_task = criteria[task](logits, targets)
                        loss_total = loss_total + task_weights[task] * loss_task
                val_losses.append(loss_total.item())
                for task in ('skin', 'lesion', 'bm'):
                    mask = masks[task]
                    valid_idx = mask.nonzero(as_tuple=True)[0]
                    if valid_idx.numel() == 0:
                        continue
                    pred = outputs[task].argmax(dim=1)[valid_idx].detach().cpu().numpy().tolist()
                    if task == 'lesion' and isinstance(labels['lesion'], torch.Tensor) and labels['lesion'].dtype.is_floating_point:
                        trg_tensor = labels['lesion'][valid_idx]
                        trg = trg_tensor.argmax(dim=1).detach().cpu().numpy().tolist()
                    else:
                        trg = labels[task][valid_idx].detach().cpu().numpy().tolist()
                    val_preds[task].extend(pred)
                    val_tgts[task].extend(trg)

        val_log: Dict[str, float] = {'epoch': epoch}
        if len(val_losses) > 0:
            val_log['val_loss'] = float(sum(val_losses) / len(val_losses))
        for task in ('skin', 'lesion', 'bm'):
            preds = val_preds[task]
            tgts = val_tgts[task]
            if len(preds) > 0:
                val_log[f'val_{task}_acc'] = accuracy_score(tgts, preds)
                val_log[f'val_{task}_f1_macro'] = f1_score(tgts, preds, average='macro')
            else:
                val_log[f'val_{task}_acc'] = 0.0
                val_log[f'val_{task}_f1_macro'] = 0.0

        # ODIN/MSP on validation (MLP1/skin)
        val_odin_log: Dict[str, float] = {}
        val_msp_log: Dict[str, float] = {}
        # Always compute MSP baseline
        all_msp_scores: List[torch.Tensor] = []
        all_msp_skin_targets: List[torch.Tensor] = []
        for features, labels, masks, _ in tqdm(val_loader, desc='MSP Val (skin)'):
            features = features.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            valid_idx = masks['skin'].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            batch_feats = features[valid_idx]
            msp_scores, _ = compute_msp_scores(
                model=model,
                features=batch_feats,
                device=device,
                head='skin',
            )
            all_msp_scores.append(msp_scores.detach().cpu())
            all_msp_skin_targets.append(labels['skin'][valid_idx].detach().cpu())
        if all_msp_scores:
            msp_scores_cat = torch.cat(all_msp_scores).numpy()
            y_true_skin_msp = torch.cat(all_msp_skin_targets).numpy()
            msp_ood_labels = (y_true_skin_msp == 0).astype(np.int32)
            try:
                val_msp_log['val_msp_auroc'] = float(roc_auc_score(msp_ood_labels, msp_scores_cat))
                val_msp_log['val_msp_aupr'] = float(average_precision_score(msp_ood_labels, msp_scores_cat))
            except Exception:
                pass
        if odin_cfg['enabled']:
            all_odin_scores: List[torch.Tensor] = []
            all_skin_targets: List[torch.Tensor] = []
            for features, labels, masks, _ in tqdm(val_loader, desc='ODIN Val (skin)'):
                features = features.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                masks = {k: v.to(device) for k, v in masks.items()}
                valid_idx = masks['skin'].nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                batch_feats = features[valid_idx]
                with torch.enable_grad():
                    odin_scores, _ = compute_odin_scores(
                        model=model,
                        features=batch_feats,
                        device=device,
                        head='skin',
                        temperature=odin_cfg['temperature'],
                        epsilon=odin_cfg['epsilon'],
                    )
                all_odin_scores.append(odin_scores.detach().cpu())
                all_skin_targets.append(labels['skin'][valid_idx].detach().cpu())
            if all_odin_scores:
                odin_scores_cat = torch.cat(all_odin_scores).numpy()
                y_true_skin = torch.cat(all_skin_targets).numpy()
                ood_labels = (y_true_skin == 0).astype(np.int32)
                try:
                    val_odin_log['val_odin_auroc'] = float(roc_auc_score(ood_labels, odin_scores_cat))
                    val_odin_log['val_odin_aupr'] = float(average_precision_score(ood_labels, odin_scores_cat))
                except Exception:
                    pass

        # Scheduler step (cosine)
        scheduler.step()

        # Log to CSV and wandb
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_log['train_loss'],
                val_log.get('val_loss', 0.0),
                train_log.get('train_skin_acc', 0.0), train_log.get('train_lesion_acc', 0.0), train_log.get('train_bm_acc', 0.0),
                val_log.get('val_skin_acc', 0.0), val_log.get('val_lesion_acc', 0.0), val_log.get('val_bm_acc', 0.0),
                val_log.get('val_skin_f1_macro', 0.0), val_log.get('val_lesion_f1_macro', 0.0), val_log.get('val_bm_f1_macro', 0.0),
                optimizer.param_groups[0]['lr']
            ])

        if val_msp_log:
            print(f"MSP (val): AUROC={val_msp_log.get('val_msp_auroc', float('nan')):.5f}, AUPR={val_msp_log.get('val_msp_aupr', float('nan')):.5f}")
        if val_odin_log:
            print(f"ODIN (val): AUROC={val_odin_log.get('val_odin_auroc', float('nan')):.5f}, AUPR={val_odin_log.get('val_odin_aupr', float('nan')):.5f}")
        wandb.log({**train_log, **val_log, **val_msp_log, **val_odin_log, 'lr': optimizer.param_groups[0]['lr']})

        # Print epoch summary similar to parallel
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss {train_log['train_loss']:.5f} | "
            f"train_skin_acc {train_log.get('train_skin_acc', 0.0):.5f} | "
            f"train_lesion_acc {train_log.get('train_lesion_acc', 0.0):.5f} | "
            f"train_bm_acc {train_log.get('train_bm_acc', 0.0):.5f} | "
            f"val_loss {val_log.get('val_loss', 0.0):.5f} | "
            f"val_skin_acc {val_log.get('val_skin_acc', 0.0):.5f} | "
            f"val_lesion_acc {val_log.get('val_lesion_acc', 0.0):.5f} | "
            f"val_bm_acc {val_log.get('val_bm_acc', 0.0):.5f} | "
            f"val_skin_f1_macro {val_log.get('val_skin_f1_macro', 0.0):.5f} | "
            f"val_lesion_f1_macro {val_log.get('val_lesion_f1_macro', 0.0):.5f} | "
            f"val_bm_f1_macro {val_log.get('val_bm_f1_macro', 0.0):.5f} | "
            f"lr {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save best checkpoint
        if 'val_loss' in val_log and val_log['val_loss'] < best_val_loss:
            best_val_loss = val_log['val_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_dim': 256,
                    'hidden_dims': list(hidden_dims),
                    'dropout': dropout,
                    'experiment': experiment,
                },
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, best_checkpoint_path)

        # Clean up
        torch.cuda.empty_cache()

    # Test evaluation
    model.eval()
    # Load best model if available
    if best_checkpoint_path.exists():
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    test_losses: List[float] = []
    test_preds: Dict[str, List[int]] = {'skin': [], 'lesion': [], 'bm': []}
    test_tgts: Dict[str, List[int]] = {'skin': [], 'lesion': [], 'bm': []}
    with torch.no_grad():
        for features, labels, masks, _ in tqdm(test_loader, desc='Test'):
            features = features.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            with torch.amp.autocast(device_type=device.type):
                outputs = model(features)
                loss_total = 0.0
                for task in ('skin', 'lesion', 'bm'):
                    mask = masks[task]
                    valid_idx = mask.nonzero(as_tuple=True)[0]
                    if valid_idx.numel() == 0:
                        continue
                    logits = outputs[task][valid_idx]
                    if task == 'lesion' and isinstance(labels['lesion'], torch.Tensor) and labels['lesion'].dtype.is_floating_point:
                        targets_soft = labels['lesion'][valid_idx]
                        log_probs = torch.log_softmax(logits, dim=1)
                        loss_task = -(targets_soft * log_probs).sum(dim=1).mean()
                    else:
                        targets = labels[task][valid_idx]
                        loss_task = criteria[task](logits, targets)
                    loss_total = loss_total + task_weights[task] * loss_task
            test_losses.append(loss_total.item())
            for task in ('skin', 'lesion', 'bm'):
                mask = masks[task]
                valid_idx = mask.nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                pred = outputs[task].argmax(dim=1)[valid_idx].detach().cpu().numpy().tolist()
                trg = labels[task][valid_idx].detach().cpu().numpy().tolist()
                test_preds[task].extend(pred)
                test_tgts[task].extend(trg)

    # ODIN/MSP evaluation for MLP1 (skin) on test split
    odin_log: Dict[str, float] = {}
    msp_log: Dict[str, float] = {}
    # Always compute MSP baseline on test
    all_msp_scores: List[torch.Tensor] = []
    all_msp_skin_targets: List[torch.Tensor] = []
    for features, labels, masks, _ in tqdm(test_loader, desc='MSP Test (skin)'):
        features = features.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        masks = {k: v.to(device) for k, v in masks.items()}
        valid_idx = masks['skin'].nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue
        batch_feats = features[valid_idx]
        msp_scores, _ = compute_msp_scores(
            model=model,
            features=batch_feats,
            device=device,
            head='skin',
        )
        all_msp_scores.append(msp_scores.detach().cpu())
        all_msp_skin_targets.append(labels['skin'][valid_idx].detach().cpu())
    if all_msp_scores:
        msp_scores_cat = torch.cat(all_msp_scores).numpy()
        y_true_skin_msp = torch.cat(all_msp_skin_targets).numpy()
        msp_ood_labels = (y_true_skin_msp == 0).astype(np.int32)
        try:
            msp_log['test_msp_auroc'] = float(roc_auc_score(msp_ood_labels, msp_scores_cat))
            msp_log['test_msp_aupr'] = float(average_precision_score(msp_ood_labels, msp_scores_cat))
        except Exception:
            pass
    if odin_cfg['enabled']:
        model.eval()
        all_odin_scores: List[torch.Tensor] = []
        all_skin_targets: List[torch.Tensor] = []
        for features, labels, masks, _ in tqdm(test_loader, desc='ODIN Test (skin)'):
            features = features.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            valid_idx = masks['skin'].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            batch_feats = features[valid_idx]
            with torch.enable_grad():
                odin_scores, max_probs = compute_odin_scores(
                    model=model,
                    features=batch_feats,
                    device=device,
                    head='skin',
                    temperature=odin_cfg['temperature'],
                    epsilon=odin_cfg['epsilon'],
                )
            all_odin_scores.append(odin_scores.detach().cpu())
            all_skin_targets.append(labels['skin'][valid_idx].detach().cpu())
        if all_odin_scores:
            odin_scores_cat = torch.cat(all_odin_scores).numpy()
            y_true_skin = torch.cat(all_skin_targets).numpy()
            ood_labels = (y_true_skin == 0).astype(np.int32)
            try:
                odin_log['test_skin_ood_auroc'] = float(roc_auc_score(ood_labels, odin_scores_cat))
                odin_log['test_skin_ood_aupr'] = float(average_precision_score(ood_labels, odin_scores_cat))
            except Exception:
                pass

    test_log: Dict[str, float] = {}
    if len(test_losses) > 0:
        test_log['test_loss'] = float(sum(test_losses) / len(test_losses))
    for task in ('skin', 'lesion', 'bm'):
        preds = test_preds[task]
        tgts = test_tgts[task]
        if len(preds) > 0:
            test_log[f'test_{task}_acc'] = accuracy_score(tgts, preds)
            test_log[f'test_{task}_f1_macro'] = f1_score(tgts, preds, average='macro')
        else:
            test_log[f'test_{task}_acc'] = 0.0
            test_log[f'test_{task}_f1_macro'] = 0.0

    # Save confusion matrices (raw and normalized) per head
    skin_names = ['not_skin', 'skin']
    lesion_names = ['melanocytic', 'non_melanocytic_carcinoma', 'keratosis_like', 'fibrous', 'vascular']
    bm_names = ['benign', 'malignant']
    # Raw
    save_confusion_matrix_raw(test_tgts['skin'], test_preds['skin'], skin_names, 'Skin vs Not-Skin (Test)', cm_dir_test / 'confusion_matrix_skin_test.png')
    save_confusion_matrix_raw(test_tgts['lesion'], test_preds['lesion'], lesion_names, 'Lesion Group (Test)', cm_dir_test / 'confusion_matrix_lesion_test.png')
    save_confusion_matrix_raw(test_tgts['bm'], test_preds['bm'], bm_names, 'Benign vs Malignant (Test)', cm_dir_test / 'confusion_matrix_bm_test.png')
    # Normalized
    plot_and_save_confusion_matrix(test_tgts['skin'], test_preds['skin'], skin_names, 'Skin vs Not-Skin (Test)', cm_dir_test / 'confusion_matrix_skin_test_normalized.png')
    plot_and_save_confusion_matrix(test_tgts['lesion'], test_preds['lesion'], lesion_names, 'Lesion Group (Test)', cm_dir_test / 'confusion_matrix_lesion_test_normalized.png')
    plot_and_save_confusion_matrix(test_tgts['bm'], test_preds['bm'], bm_names, 'Benign vs Malignant (Test)', cm_dir_test / 'confusion_matrix_bm_test_normalized.png')

    # Save lesion confusion matrix as CSV
    try:
        import csv as _csv
        cm = confusion_matrix(test_tgts['lesion'], test_preds['lesion'], labels=list(range(8)))
        cm_csv = checkpoints_dir / 'confusion_matrix_lesion_test.csv'
        with open(cm_csv, 'w', newline='') as f:
            w = _csv.writer(f)
            for row in cm.tolist():
                w.writerow(row)
    except Exception as _e:
        print(f"[WARN] Failed to save lesion confusion matrix CSV: {_e}")

    # Exactly six plots under plots/exp5 to mirror parallel style
    plots_dir = project_root / 'plots' / experiment
    ensure_dir(plots_dir)
    # 1-2: Lesion confusion matrices (raw + normalized)
    save_confusion_matrix_raw(test_tgts['lesion'], test_preds['lesion'], lesion_names, 'Lesion Group (Test)', plots_dir / 'confusion_matrix_test.png')
    plot_and_save_confusion_matrix(test_tgts['lesion'], test_preds['lesion'], lesion_names, 'Lesion Group (Test)', plots_dir / 'confusion_matrix_test_normalized.png')
    # 3: Per-class metrics (lesion)
    save_per_class_metrics_bar(test_tgts['lesion'], test_preds['lesion'], lesion_names, plots_dir / 'per_class_metrics_test.png', 'Per-Class Metrics (Test) - Lesion')
    # 4: Test metrics bar (lesion)
    save_test_metrics_bar(test_log.get('test_lesion_acc', 0.0), test_log.get('test_lesion_f1_macro', 0.0), plots_dir / 'test_metrics_bar.png', 'Test Metrics - Lesion')

    # Also save validation CMs for reference
    # Recompute from last cached preds if needed; simplest is to run a quick val pass
    val_preds = {'skin': [], 'lesion': [], 'bm': []}
    val_tgts = {'skin': [], 'lesion': [], 'bm': []}
    with torch.no_grad():
        for features, labels, masks, _ in tqdm(val_loader, desc='Val CM'):
            features = features.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            with torch.amp.autocast(device_type=device.type):
                outputs = model(features)
            for task in ('skin', 'lesion', 'bm'):
                valid_idx = masks[task].nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                pred = outputs[task].argmax(dim=1)[valid_idx].detach().cpu().numpy().tolist()
                trg = labels[task][valid_idx].detach().cpu().numpy().tolist()
                val_preds[task].extend(pred)
                val_tgts[task].extend(trg)

    plot_and_save_confusion_matrix(val_tgts['skin'], val_preds['skin'], skin_names, 'Skin vs Not-Skin (Val)', cm_dir_val / 'confusion_matrix_skin_val_normalized.png')
    plot_and_save_confusion_matrix(val_tgts['lesion'], val_preds['lesion'], lesion_names, 'Lesion Group (Val)', cm_dir_val / 'confusion_matrix_lesion_val_normalized.png')
    plot_and_save_confusion_matrix(val_tgts['bm'], val_preds['bm'], bm_names, 'Benign vs Malignant (Val)', cm_dir_val / 'confusion_matrix_bm_val_normalized.png')

    # Final logs and run-summary style prints
    if msp_log:
        print(f"MSP (test): AUROC={msp_log.get('test_msp_auroc', float('nan')):.5f}, AUPR={msp_log.get('test_msp_aupr', float('nan')):.5f}")
    if odin_log:
        print(f"ODIN (test): AUROC={odin_log.get('test_skin_ood_auroc', float('nan')):.5f}, AUPR={odin_log.get('test_skin_ood_aupr', float('nan')):.5f}")
    wandb.log({**test_log, **msp_log, **odin_log})
    # Print split counts like parallel summary
    try:
        split_stats = compute_split_class_counts(project_root / 'data/metadata/metadata.csv', experiment_col=experiment)
        print_split_counts(split_stats)
    except Exception as e:
        print(f"[WARN] Failed to print split counts: {e}")
    # Print wandb-like run summary
    print("Run summary:")
    print(f"wandb: epoch {epochs}")
    print(f"wandb: lr {optimizer.param_groups[0]['lr']:.5f}")
    print(f"wandb: test_skin_acc {test_log.get('test_skin_acc', 0.0):.5f}")
    print(f"wandb: test_lesion_acc {test_log.get('test_lesion_acc', 0.0):.5f}")
    print(f"wandb: test_bm_acc {test_log.get('test_bm_acc', 0.0):.5f}")
    print(f"wandb: test_skin_f1_macro {test_log.get('test_skin_f1_macro', 0.0):.5f}")
    print(f"wandb: test_lesion_f1_macro {test_log.get('test_lesion_f1_macro', 0.0):.5f}")
    print(f"wandb: test_bm_f1_macro {test_log.get('test_bm_f1_macro', 0.0):.5f}")
    # Save final checkpoint
    final_ckpt_path = checkpoints_dir / 'final.pt'
    torch.save({'model_state_dict': model.state_dict(), 'config': {'input_dim': 256, 'hidden_dims': list(hidden_dims), 'dropout': dropout, 'experiment': experiment}}, final_ckpt_path)
    # Save class distribution per split plot similar to parallel
    try:
        split_stats = compute_split_class_counts(project_root / 'data/metadata/metadata.csv', experiment_col=experiment)
        plots_dir = project_root / 'plots' / experiment
        ensure_dir(plots_dir)
        # 5: Class distribution per split
        save_class_distribution_bar(split_stats, plots_dir / 'class_distribution_splits.png', title=f'Class Distribution per Split ({experiment})')
        # 6: Loss/accuracy curves for lesion head + loss
        save_loss_accuracy_curves_lesion(csv_path, plots_dir / 'loss_accuracy_curves.png')
    except Exception as e:
        print(f"[WARN] Failed to create class distribution plot: {e}")
    wandb.finish()
    
    # Clean up file handler if it was added
    if log_file and 'file_handler' in locals():
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    # Append summary metrics to backend/experiments_summary.csv
    try:
        backend_csv = project_root / 'backend' / 'experiments_summary.csv'
        ensure_dir(backend_csv.parent)
        write_header = not backend_csv.exists()
        with open(backend_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['timestamp','experiment','architecture','wandb_project','wandb_name','best_val_skin_acc','best_val_lesion_acc','best_val_bm_acc','test_skin_acc','test_skin_f1_macro','test_lesion_acc','test_lesion_f1_macro','test_bm_acc','test_bm_f1_macro','odin_enabled','odin_temperature','odin_epsilon','val_msp_auroc','val_msp_aupr','test_msp_auroc','test_msp_aupr','val_odin_auroc','val_odin_aupr','test_skin_ood_auroc','test_skin_ood_aupr'])
            writer.writerow([
                datetime.now().isoformat(timespec='seconds'),
                experiment,
                'multi',
                wandb_project,
                wandb_name,
                best_val_metrics.get('skin',0.0),
                best_val_metrics.get('lesion',0.0),
                best_val_metrics.get('bm',0.0),
                test_log.get('test_skin_acc',0.0),
                test_log.get('test_skin_f1_macro',0.0),
                test_log.get('test_lesion_acc',0.0),
                test_log.get('test_lesion_f1_macro',0.0),
                test_log.get('test_bm_acc',0.0),
                test_log.get('test_bm_f1_macro',0.0),
                bool(odin_cfg.get('enabled', False)),
                float(odin_cfg.get('temperature', 1000.0)),
                float(odin_cfg.get('epsilon', 0.001)),
                val_msp_log.get('val_msp_auroc', None) if 'val_msp_log' in locals() else None,
                val_msp_log.get('val_msp_aupr', None) if 'val_msp_log' in locals() else None,
                msp_log.get('test_msp_auroc', None) if 'msp_log' in locals() else None,
                msp_log.get('test_msp_aupr', None) if 'msp_log' in locals() else None,
                val_odin_log.get('val_odin_auroc', None) if 'val_odin_log' in locals() else None,
                val_odin_log.get('val_odin_aupr', None) if 'val_odin_log' in locals() else None,
                odin_log.get('test_skin_ood_auroc', None) if 'odin_log' in locals() else None,
                odin_log.get('test_skin_ood_aupr', None) if 'odin_log' in locals() else None,
            ])
    except Exception as e:
        print(f"[WARN] Failed to append to backend/experiments_summary.csv: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train multi-head model on SAM features for a selected experiment split')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--hidden_dims', type=str, default='512,256')
    parser.add_argument('--skin_weight', type=float, default=1.0)
    parser.add_argument('--lesion_weight', type=float, default=1.0)
    parser.add_argument('--bm_weight', type=float, default=1.0)
    parser.add_argument('--wandb_project', type=str, default='skin-lesion-classification')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--experiment', type=str, default='exp1')
    args = parser.parse_args()

    hidden_dims_tuple = tuple(int(x) for x in args.hidden_dims.split(',') if x)
    train_multihead(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_dims=hidden_dims_tuple,
        skin_weight=args.skin_weight,
        lesion_weight=args.lesion_weight,
        bm_weight=args.bm_weight,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        num_workers=args.num_workers,
        experiment=args.experiment,
    )


