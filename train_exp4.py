import os
import sys
import csv
import math
import time
import gc
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb


# Ensure project root on path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CLASS_NAMES_10 = [
    'UNKNOWN', 'NV', 'MEL', 'BCC', 'BKL', 'AKIEC', 'NOT_SKIN', 'SCC', 'VASC', 'DF'
]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES_10)}


class UnifiedExp4Dataset(Dataset):
    """Unified 10-class dataset backed by SAM feature stores and metadata exp4 splits."""

    def __init__(
        self,
        split: str,
        metadata_csv: Optional[str] = None,
        features_dir: Optional[str] = None,
        feature_dim: int = 256,
        experiment_col: str = 'exp4'
    ):
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.feature_dim = feature_dim
        self.experiment_col = experiment_col

        self.metadata_csv = metadata_csv or str(project_root / 'data/metadata/metadata.csv')
        self.features_dir = Path(features_dir) if features_dir else (project_root / 'data/processed/features')

        self.df = pd.read_csv(self.metadata_csv)
        if self.experiment_col not in self.df.columns:
            raise ValueError(f"Missing column {self.experiment_col} in metadata: {self.metadata_csv}")

        # Filter by split and non-empty assignment
        mask_non_empty = self.df[self.experiment_col].astype(str) != ''
        mask_split = self.df[self.experiment_col].astype(str) == split
        self.df = self.df[mask_non_empty & mask_split].reset_index(drop=True)

        # Build quick lookup of labels from row
        # Expect column 'unified_diagnosis' with values among CLASS_NAMES_10
        if 'unified_diagnosis' not in self.df.columns:
            # Backward compatibility
            alt_cols = [c for c in self.df.columns if c.lower().startswith('unified')]
            raise ValueError(f"Metadata must contain 'unified_diagnosis' column. Found alternatives: {alt_cols}")

        # Load feature stores (try split-specific, then 'all')
        self.feature_stores: Dict[str, Dict[str, np.ndarray]] = {}
        self._load_feature_store('all')
        self._load_feature_store(self.split)

    def _load_feature_store(self, name: str):
        pkl_path = self.features_dir / f"sam_features_{name}.pkl"
        if not pkl_path.exists():
            inc_path = self.features_dir / f"sam_features_{name}_incremental.pkl"
            if not inc_path.exists():
                print(f"[WARN] Feature store not found: {pkl_path} or {inc_path}")
                return
            pkl_path = inc_path
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                store = pickle.load(f)
                if isinstance(store, dict):
                    self.feature_stores[name] = store
                    print(f"[INFO] Loaded features: {name} ({len(store)} items) from {pkl_path}")
                else:
                    print(f"[WARN] Unexpected feature store format in {pkl_path}")
        except Exception as e:
            print(f"[WARN] Failed to load features from {pkl_path}: {e}")

    def __len__(self) -> int:
        return len(self.df)

    def _get_features(self, image_name: str) -> np.ndarray:
        # Try split-specific first, then 'all'
        for key in (self.split, 'all'):
            store = self.feature_stores.get(key)
            if store is not None:
                feat = store.get(image_name)
                if feat is not None:
                    return feat
        # Fallback to zeros
        return np.zeros(self.feature_dim, dtype=np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_name = str(row['image_name']) if 'image_name' in row else str(row.get('image'))
        if image_name == 'nan' or image_name is None or image_name == 'None':
            image_name = str(row.get('image'))

        features = self._get_features(image_name)
        x = torch.from_numpy(features).float()

        label_str = str(row['unified_diagnosis']).upper()
        if label_str not in CLASS_TO_INDEX:
            # Map any unexpected label to UNKNOWN
            y = CLASS_TO_INDEX['UNKNOWN']
        else:
            y = CLASS_TO_INDEX[label_str]

        return x, torch.tensor(y, dtype=torch.long)


class UnifiedClassifier(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dims: Tuple[int, int] = (512, 256), dropout: float = 0.3, num_classes: int = 10):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mlp(x) if len(self.mlp) > 0 else x
        return self.classifier(z)


def compute_class_weights(train_dataset: UnifiedExp4Dataset, device: torch.device) -> Optional[torch.Tensor]:
    labels = [CLASS_TO_INDEX.get(str(lbl).upper(), CLASS_TO_INDEX['UNKNOWN']) for lbl in train_dataset.df['unified_diagnosis']]
    counts = np.bincount(labels, minlength=len(CLASS_NAMES_10)).astype(np.float32)
    with np.errstate(divide='ignore'):
        inv = 1.0 / np.clip(counts, a_min=1.0, a_max=None)
    weights = inv / inv.sum() * len(CLASS_NAMES_10)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def ensure_dirs():
    (project_root / 'models').mkdir(parents=True, exist_ok=True)
    (project_root / 'logs').mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path, title: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()


def save_confusion_matrix_normalized(cm: np.ndarray, class_names: List[str], out_path: Path, title: str):
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, np.maximum(row_sums, 1), where=row_sums!=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(norm * 100.0, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title + ' (Row-normalized %)')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()


def save_per_class_metrics_bar(metrics: Dict[str, np.ndarray], class_names: List[str], out_path: Path, title: str):
    # metrics keys: 'precision', 'recall', 'f1'
    x = np.arange(len(class_names))
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, metrics['precision'], width, label='Precision')
    plt.bar(x, metrics['recall'], width, label='Recall')
    plt.bar(x + width, metrics['f1'], width, label='F1')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim(0.0, 1.0)
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()


def save_loss_accuracy_curves(metrics_csv: Path, out_path: Path, title: str):
    df = pd.read_csv(metrics_csv)
    epochs = df['epoch'].values
    train_acc = df['train_acc'].values
    val_acc = df['val_acc'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_acc, label='Train Acc')
    axes[0].plot(epochs, val_acc, label='Val Acc')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy (Train vs Val)')
    axes[0].legend()
    axes[1].plot(epochs, train_loss, label='Train Loss')
    axes[1].plot(epochs, val_loss, label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss (Train vs Val)')
    axes[1].legend()
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches='tight')
    plt.close(fig)


def save_class_distribution_bar(split_stats: Dict[str, Dict[str, int]], out_path: Path, title: str):
    # Prepare dataframe
    rows = []
    for split, dist in split_stats.items():
        for cls in CLASS_NAMES_10:
            rows.append({'split': split, 'class': cls, 'count': dist.get(cls, 0)})
    df_plot = pd.DataFrame(rows)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='class', y='count', hue='split')
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()


def log_split_distributions(df: pd.DataFrame, experiment_col: str = 'exp4') -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    # Overall split sizes from experiment column
    print('[INFO] exp4 value_counts (overall):')
    try:
        print(df[experiment_col].value_counts(dropna=False))
    except Exception as e:
        print(f'[WARN] Could not compute value_counts for {experiment_col}: {e}')
    for split in ['train', 'val', 'test']:
        subset = df[df[experiment_col] == split]
        counts = subset['unified_diagnosis'].str.upper().value_counts().reindex(CLASS_NAMES_10, fill_value=0)
        stats[split] = counts.to_dict()
    # Log empties summary
    non_empty = (df[experiment_col].astype(str) != '').sum()
    total = len(df)
    empty = total - non_empty
    print(f"[INFO] {experiment_col} total: {total} (non-empty: {non_empty}, empty: {empty})")
    for split, dist in stats.items():
        total_split = sum(dist.values())
        print(f"[INFO] {split} count: {total_split}")
        print("       " + ", ".join([f"{k}: {v}" for k, v in dist.items()]))
        # Verify all 10 classes represented in this split
        missing = [cls for cls in CLASS_NAMES_10 if dist.get(cls, 0) == 0]
        if missing:
            print(f"[WARN] Missing classes in {split}: {missing}")
        else:
            print(f"[INFO] All 10 classes present in {split}")
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 4: Unified 10-class classifier on SAM features')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--scheduler', type=str, default='reduce', choices=['reduce', 'step', 'none'])
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--experiment_col', type=str, default='exp4')
    parser.add_argument('--verify_only', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    set_global_seed(42)
    ensure_dirs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Load metadata once to log distributions
    metadata_csv = str(project_root / 'data/metadata/metadata.csv')
    df_all = pd.read_csv(metadata_csv)
    split_stats = log_split_distributions(df_all, experiment_col=args.experiment_col)

    if args.verify_only:
        # Print per-split unified_diagnosis value_counts explicitly
        for split in ['train', 'val', 'test']:
            subset = df_all[df_all[args.experiment_col] == split]
            print(f"\n[INFO] unified_diagnosis value_counts for split={split}:")
            print(subset['unified_diagnosis'].str.upper().value_counts())
        return

    # Datasets and loaders
    train_ds = UnifiedExp4Dataset('train', metadata_csv=metadata_csv, feature_dim=args.feature_dim, experiment_col=args.experiment_col)
    val_ds = UnifiedExp4Dataset('val', metadata_csv=metadata_csv, feature_dim=args.feature_dim, experiment_col=args.experiment_col)
    test_ds = UnifiedExp4Dataset('test', metadata_csv=metadata_csv, feature_dim=args.feature_dim, experiment_col=args.experiment_col)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = UnifiedClassifier(input_dim=args.feature_dim, hidden_dims=(512, 256), dropout=args.dropout, num_classes=len(CLASS_NAMES_10)).to(device)

    # Loss
    if args.use_class_weights and len(train_ds) > 0:
        class_weights = compute_class_weights(train_ds, device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, patience=3)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    # WandB
    wandb.init(project='skin-lesion-classification', name=f'{args.experiment_col}_unified_classifier', config={
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'dropout': args.dropout,
        'feature_dim': args.feature_dim,
        'experiment_col': args.experiment_col,
        'class_names': CLASS_NAMES_10,
        'split_counts': {k: sum(v.values()) for k, v in split_stats.items()}
    })
    # Also log per-split class distributions at start
    initial_log = {}
    for split_name, dist in split_stats.items():
        for cls, cnt in dist.items():
            initial_log[f'{split_name}_count/{cls}'] = cnt
    if initial_log:
        wandb.log(initial_log)

    # CSV logger
    metrics_csv_path = project_root / 'logs' / f'{args.experiment_col}_train_log.csv'
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1_macro', 'lr'])

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_state = None

    # If eval-only, load checkpoint now
    if args.eval_only:
        ckpt_path = project_root / 'models' / f'{args.experiment_col}_unified_classifier.pth'
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(state)
            best_state = state
            print(f"[INFO] Loaded checkpoint for eval_only from {ckpt_path}")
        else:
            print(f"[WARN] No checkpoint found at {ckpt_path}; evaluating current model state.")

    for epoch in ([] if args.eval_only else range(1, args.epochs + 1)):
        epoch_start = time.time()
        model.train()
        train_losses: List[float] = []
        train_preds: List[int] = []
        train_targets: List[int] = []

        for x, y in tqdm(train_loader, desc=f'Epoch {epoch} - Train'):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            targs = y.detach().cpu().numpy().tolist()
            train_preds.extend(preds)
            train_targets.extend(targs)

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = float(accuracy_score(train_targets, train_preds)) if train_targets else 0.0

        # Validation
        model.eval()
        val_losses: List[float] = []
        val_preds: List[int] = []
        val_targets: List[int] = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f'Epoch {epoch} - Val'):
                x = x.to(device)
                y = y.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, y)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
                targs = y.detach().cpu().numpy().tolist()
                val_preds.extend(preds)
                val_targets.extend(targs)

        val_loss = float(np.mean(val_losses)) if val_losses else math.inf
        val_acc = float(accuracy_score(val_targets, val_preds)) if val_targets else 0.0
        val_f1_macro = float(f1_score(val_targets, val_preds, average='macro')) if val_targets else 0.0
        # Validation confusion matrix to wandb
        if val_targets:
            val_cm = confusion_matrix(val_targets, val_preds, labels=list(range(len(CLASS_NAMES_10))))
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(val_cm, annot=False, cmap='Blues', xticklabels=CLASS_NAMES_10, yticklabels=CLASS_NAMES_10)
            plt.title(f'Validation Confusion Matrix - Epoch {epoch}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            wandb.log({
                'val_confusion_matrix': wandb.Image(fig)
            })
            plt.close(fig)

        # Step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        # Log
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1_macro': val_f1_macro,
            'lr': current_lr
        })

        with open(metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_f1_macro, current_lr])

        # Track best
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            improved = True
        if val_f1_macro >= best_val_f1 and (val_acc == best_val_acc):
            best_val_f1 = val_f1_macro
            improved = True
        if improved:
            best_state = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} | time {epoch_time:.1f}s | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | val_f1_macro {val_f1_macro:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    # Save best model
    model_out = project_root / 'models' / f'{args.experiment_col}_unified_classifier.pth'
    if best_state is not None:
        torch.save(best_state, str(model_out))
    else:
        torch.save(model.state_dict(), str(model_out))
    print(f"[INFO] Saved checkpoint to {model_out}")

    # Test evaluation
    model.eval()
    if best_state is not None:
        model.load_state_dict(best_state)
    test_preds: List[int] = []
    test_targets: List[int] = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Test'):
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(x)
            preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            targs = y.detach().cpu().numpy().tolist()
            test_preds.extend(preds)
            test_targets.extend(targs)

    if len(test_targets) > 0:
        test_acc = float(accuracy_score(test_targets, test_preds))
        test_f1_macro = float(f1_score(test_targets, test_preds, average='macro'))
        cm = confusion_matrix(test_targets, test_preds, labels=list(range(len(CLASS_NAMES_10))))

        wandb.log({'test_acc': test_acc, 'test_f1_macro': test_f1_macro})
        print(f"[TEST] acc {test_acc:.4f} | f1_macro {test_f1_macro:.4f}")

        # Prepare plots output dir
        plots_dir = project_root / 'plots' / args.experiment_col
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrices (raw and normalized)
        cm_out = plots_dir / 'confusion_matrix_test.png'
        save_confusion_matrix(cm, CLASS_NAMES_10, cm_out, title='Test Confusion Matrix')
        cm_norm_out = plots_dir / 'confusion_matrix_test_normalized.png'
        save_confusion_matrix_normalized(cm, CLASS_NAMES_10, cm_norm_out, title='Test Confusion Matrix')
        wandb.log({'test_confusion_matrix': wandb.Image(str(cm_out))})
        wandb.log({'test_confusion_matrix_normalized': wandb.Image(str(cm_norm_out))})

        # Plot and log a small figure for test accuracy and F1
        fig = plt.figure(figsize=(4, 3))
        bars = plt.bar(['Accuracy', 'F1 (macro)'], [test_acc, test_f1_macro], color=['#1f77b4', '#ff7f0e'])
        plt.ylim(0.0, 1.0)
        for b, val in zip(bars, [test_acc, test_f1_macro]):
            plt.text(b.get_x() + b.get_width()/2, val + 0.02, f"{val:.3f}", ha='center', va='bottom', fontsize=10)
        plt.title('Test Metrics')
        plt.ylabel('Score')
        test_metrics_plot_path = plots_dir / 'test_metrics_bar.png'
        fig.savefig(str(test_metrics_plot_path), bbox_inches='tight')
        wandb.log({'test_metrics_plot': wandb.Image(fig)})
        plt.close(fig)

        # Per-class metrics (precision/recall/F1) on test set
        # Compute per-class metrics safely
        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, f1, _ = precision_recall_fscore_support(test_targets, test_preds, labels=list(range(len(CLASS_NAMES_10))), zero_division=0)
        per_class = {
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        per_class_plot = plots_dir / 'per_class_metrics_test.png'
        save_per_class_metrics_bar(per_class, CLASS_NAMES_10, per_class_plot, title='Per-Class Metrics (Test)')
        wandb.log({'per_class_metrics_test': wandb.Image(str(per_class_plot))})

        # Loss & Accuracy curves from CSV
        metrics_csv_path = project_root / 'logs' / f'{args.experiment_col}_train_log.csv'
        if metrics_csv_path.exists():
            curves_out = plots_dir / 'loss_accuracy_curves.png'
            save_loss_accuracy_curves(metrics_csv_path, curves_out, title='Training Curves (Acc/Loss)')
            wandb.log({'loss_accuracy_curves': wandb.Image(str(curves_out))})

        # Class distribution bar plot
        split_stats = log_split_distributions(pd.read_csv(project_root / 'data/metadata/metadata.csv'), experiment_col=args.experiment_col)
        dist_out = plots_dir / 'class_distribution_splits.png'
        save_class_distribution_bar(split_stats, dist_out, title='Class Distribution per Split')
        wandb.log({'class_distribution_splits': wandb.Image(str(dist_out))})

    wandb.finish()


if __name__ == '__main__':
    main()


