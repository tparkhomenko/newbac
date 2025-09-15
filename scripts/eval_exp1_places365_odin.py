#!/usr/bin/env python3
"""
Evaluate ODIN on Exp1 test and Places365 OOD for two checkpoints (no_odin, with_odin).
Also compute standard Exp1 classification metrics and save lesion confusion matrices.
Append results to backend/experiments_summary.csv and log to W&B.
"""

import os
import sys
import csv
import pickle
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, average_precision_score

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from datasets.parallel_unified_dataset import ParallelUnifiedDataset
from models.multitask_model import MultiTaskHead
from utils.odin import compute_odin_scores


def _find_latest_final(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob('*/final.pt'))
    if not candidates:
        # allow direct final.pt in dir
        direct = model_dir / 'final.pt'
        if direct.exists():
            return direct
        raise FileNotFoundError(f"No final.pt under {model_dir}")
    return candidates[-1]


def _load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultiTaskHead(
        input_dim=256,
        hidden_dims=(512, 256),
        dropout=0.3,
        num_classes_skin=2,
        num_classes_lesion=8,
        num_classes_bm=2,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def _exp1_metrics(model: torch.nn.Module, device: torch.device) -> Dict[str, float]:
    ds = ParallelUnifiedDataset(experiment='exp1', split='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    preds: Dict[str, list] = {k: [] for k in ('skin', 'lesion', 'bm')}
    tgts: Dict[str, list] = {k: [] for k in ('skin', 'lesion', 'bm')}
    with torch.no_grad():
        for features, labels, masks, _ in tqdm(dl, desc='Exp1 test'):            
            features = features.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            outputs = model(features)
            for task in ('skin','lesion','bm'):
                valid_idx = masks[task].nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                p = outputs[task].argmax(dim=1)[valid_idx].detach().cpu().numpy().tolist()
                t = labels[task][valid_idx].detach().cpu().numpy().tolist()
                preds[task].extend(p)
                tgts[task].extend(t)
    out: Dict[str, float] = {}
    for task in ('skin','lesion','bm'):
        if len(preds[task]) == 0:
            out[f'{task}_acc'] = 0.0
            out[f'{task}_f1'] = 0.0
            continue
        t = np.array(tgts[task])
        p = np.array(preds[task])
        out[f'{task}_acc'] = float((t == p).mean())
        out[f'{task}_f1'] = float(f1_score(t, p, average='macro'))
    # Save lesion confusion matrix CSV
    cm = confusion_matrix(np.array(tgts['lesion']), np.array(preds['lesion']), labels=list(range(8))) if len(tgts['lesion'])>0 else None
    return out, cm


def _odin_metrics_skin(model: torch.nn.Module, device: torch.device, dataset, desc: str) -> Tuple[float, float]:
    dl = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    scores = []
    labels_skin = []
    for batch in tqdm(dl, desc=desc):
        if isinstance(batch, dict):
            feats = batch['features'].to(device)
            y_skin = torch.zeros(len(feats), dtype=torch.long, device=device)
            valid_idx = torch.arange(len(feats), device=device)
        else:
            feats, labels, masks, _ = batch
            feats = feats.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            valid_idx = masks['skin'].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            y_skin = labels['skin'][valid_idx]
        x = feats[valid_idx].float().detach().clone()
        x.requires_grad_(True)
        with torch.enable_grad():
            odin, _ = compute_odin_scores(model, x, device, head='skin', temperature=1000.0, epsilon=0.001)
        scores.append(odin.detach().cpu())
        labels_skin.append(y_skin.detach().cpu())
    if not scores:
        return 0.0, 0.0
    s = torch.cat(scores).numpy()
    y = torch.cat(labels_skin).numpy()
    ood = (y == 0).astype(np.int32)
    return float(roc_auc_score(ood, s)), float(average_precision_score(ood, s))


def _places365_dataset() -> torch.utils.data.Dataset:
    pkl = root / 'data' / 'processed' / 'features' / 'sam_features_places365_test.pkl'
    class Places(torch.utils.data.Dataset):
        def __init__(self, p):
            with open(p, 'rb') as f:
                self.store = pickle.load(f)
            self.names = list(self.store.keys())
        def __len__(self):
            return len(self.names)
        def __getitem__(self, idx):
            name = self.names[idx]
            feat = torch.tensor(self.store[name], dtype=torch.float32)
            return {'features': feat}
    return Places(pkl)


def _log_wandb(run_name: str, project: str, metrics: Dict[str, float]) -> None:
    try:
        import wandb
        wandb.init(project=project, name=run_name)
        wandb.log(metrics)
        wandb.finish()
    except Exception as e:
        print(f"[WARN] WandB logging failed: {e}")


def _append_csv(row: list) -> None:
    out_csv = root / 'backend' / 'experiments_summary.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ['timestamp','experiment','architecture','wandb_project','wandb_name','test_skin_acc','test_skin_f1','test_lesion_acc','test_lesion_f1','test_bm_acc','test_bm_f1','exp1_test_odin_auroc','exp1_test_odin_aupr','places365_test_odin_auroc','places365_test_odin_aupr']
    write_header = not out_csv.exists()
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def evaluate_one(model_dir: Path, run_name: str, wandb_project: str) -> None:
    import datetime
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = _find_latest_final(model_dir)
    print(f"\n== Evaluating {run_name} ==\nCheckpoint: {ckpt}")
    model = _load_model(ckpt, device)

    # Exp1 metrics
    exp1_metrics, lesion_cm = _exp1_metrics(model, device)
    print(f"Exp1 test: skin acc={exp1_metrics['skin_acc']:.4f}, f1={exp1_metrics['skin_f1']:.4f} | lesion acc={exp1_metrics['lesion_acc']:.4f}, f1={exp1_metrics['lesion_f1']:.4f} | bm acc={exp1_metrics['bm_acc']:.4f}, f1={exp1_metrics['bm_f1']:.4f}")

    # ODIN on Exp1 and Places365
    exp1_test_ds = ParallelUnifiedDataset(experiment='exp1', split='test')
    places_ds = _places365_dataset()
    exp1_auroc, exp1_aupr = _odin_metrics_skin(model, device, exp1_test_ds, desc='ODIN on Exp1 test')
    places_auroc, places_aupr = _odin_metrics_skin(model, device, places_ds, desc='ODIN on Places365')
    print(f"ODIN (Exp1):    AUROC={exp1_auroc:.4f}, AUPR={exp1_aupr:.4f}")
    print(f"ODIN (Places):  AUROC={places_auroc:.4f}, AUPR={places_aupr:.4f}")

    # Save lesion confusion matrix under model folder
    if lesion_cm is not None:
        cm_path = ckpt.parent / 'confusion_matrix_lesion_exp1_test.csv'
        with open(cm_path, 'w', newline='') as f:
            w = csv.writer(f)
            for row in lesion_cm.tolist():
                w.writerow(row)
        print(f"Saved lesion confusion matrix to: {cm_path}")

    # W&B
    metrics = {
        'exp1_test_skin_acc': exp1_metrics['skin_acc'],
        'exp1_test_skin_f1': exp1_metrics['skin_f1'],
        'exp1_test_lesion_acc': exp1_metrics['lesion_acc'],
        'exp1_test_lesion_f1': exp1_metrics['lesion_f1'],
        'exp1_test_bm_acc': exp1_metrics['bm_acc'],
        'exp1_test_bm_f1': exp1_metrics['bm_f1'],
        'exp1_test_odin_auroc': exp1_auroc,
        'exp1_test_odin_aupr': exp1_aupr,
        'places365_test_odin_auroc': places_auroc,
        'places365_test_odin_aupr': places_aupr,
    }
    _log_wandb(run_name, wandb_project, metrics)

    # Append CSV
    _append_csv([
        datetime.datetime.now().isoformat(timespec='seconds'),
        'exp1',
        'parallel',
        wandb_project,
        run_name,
        exp1_metrics['skin_acc'],
        exp1_metrics['skin_f1'],
        exp1_metrics['lesion_acc'],
        exp1_metrics['lesion_f1'],
        exp1_metrics['bm_acc'],
        exp1_metrics['bm_f1'],
        exp1_auroc,
        exp1_aupr,
        places_auroc,
        places_aupr,
    ])


def main() -> None:
    # Checkpoints
    no_odin_dir = root / 'backend' / 'models' / 'parallel' / 'exp1_odin_fix' / 'no_odin'
    with_odin_dir = root / 'backend' / 'models' / 'parallel' / 'exp1_odin_fix' / 'with_odin'
    # Evaluate
    evaluate_one(no_odin_dir, 'exp1_parallel_lmf_no_odin_fix_eval', 'odin_fix_exp1')
    evaluate_one(with_odin_dir, 'exp1_parallel_lmf_with_odin_fix_eval', 'odin_fix_exp1')


if __name__ == '__main__':
    main()


