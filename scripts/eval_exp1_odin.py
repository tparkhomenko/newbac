#!/usr/bin/env python3
import os
import sys
import csv
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.multitask_model import MultiTaskHead
from utils.odin import compute_odin_scores, compute_msp_scores


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
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


def load_eval_features() -> Tuple[np.ndarray, np.ndarray, list]:
    p = ROOT / 'data' / 'processed' / 'features' / 'sam_features_exp1_odin_eval.pkl'
    with open(p, 'rb') as f:
        obj = pickle.load(f)
    store = obj['features'] if isinstance(obj, dict) and 'features' in obj else obj
    names = list(store.keys())
    X = np.stack([store[n] for n in names], axis=0)
    # Labels dictionary may not exist; derive from names if needed later
    y = None
    if isinstance(obj, dict) and 'labels' in obj:
        y = np.array([obj['labels'].get(n, 0) for n in names], dtype=np.int64)
    return X, y, names


def save_curves(scores: np.ndarray, labels: np.ndarray, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(labels, scores):.3f}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'{prefix.upper()} ROC (Exp1 vs Places365)')
    plt.legend(); plt.grid(True)
    plt.savefig(out_dir / f'{prefix}_roc_exp1_vs_places.png', dpi=200, bbox_inches='tight')
    plt.close()
    # PR
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f'AP={average_precision_score(labels, scores):.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{prefix.upper()} PR (Exp1 vs Places365)')
    plt.legend(); plt.grid(True)
    plt.savefig(out_dir / f'{prefix}_pr_exp1_vs_places.png', dpi=200, bbox_inches='tight')
    plt.close()


def append_csv_row(run_name: str, project: str, auroc_odin: float, aupr_odin: float, auroc_msp: float, aupr_msp: float) -> None:
    out_csv = ROOT / 'backend' / 'experiments_summary.csv'
    header = ['timestamp','experiment','architecture','wandb_project','wandb_name','exp1_odin_eval_auroc_odin','exp1_odin_eval_aupr_odin','exp1_odin_eval_auroc_msp','exp1_odin_eval_aupr_msp']
    import datetime
    write_header = not out_csv.exists()
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([
            datetime.datetime.now().isoformat(timespec='seconds'),
            'exp1','parallel', project, run_name, auroc_odin, aupr_odin, auroc_msp, aupr_msp
        ])


def log_wandb(run_name: str, project: str, metrics: Dict[str, float]) -> None:
    try:
        import wandb
        wandb.init(project=project, name=run_name)
        wandb.log(metrics)
        wandb.finish()
    except Exception as e:
        print(f"[WARN] wandb logging failed: {e}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='Path to final.pt')
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--wandb_project', default='odin_fix_exp1')
    ap.add_argument('--temperature', type=float, default=1000.0)
    ap.add_argument('--epsilon', type=float, default=0.001)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(Path(args.checkpoint), device)

    X, y, names = load_eval_features()
    if y is None:
        # Build labels via metadata if needed (fallback not expected normally)
        print('[WARN] Labels missing; defaulting all-zero (OOD).')
        y = np.zeros(len(names), dtype=np.int64)

    # Compute ODIN and MSP scores
    odin_ood_scores = []  # conventional ODIN: higher means OOD
    msp_ood_scores = []   # MSP returns OOD score (1 - maxprob)
    bs = 256
    for i in tqdm(range(0, len(X), bs), desc='Computing ODIN on eval set'):
        feats = torch.tensor(X[i:i+bs], dtype=torch.float32, device=device)
        feats.requires_grad_(True)
        with torch.enable_grad():
            s_odin, _ = compute_odin_scores(model, feats, device, head='skin', temperature=args.temperature, epsilon=args.epsilon)
        # MSP does not need grad
        feats_msp = feats.detach()
        msp_s, _ = compute_msp_scores(model, feats_msp, device, head='skin')
        odin_ood_scores.append(s_odin.detach().cpu().numpy())
        msp_ood_scores.append(msp_s.detach().cpu().numpy())
    odin_ood_scores = np.concatenate(odin_ood_scores, axis=0)
    msp_ood_scores = np.concatenate(msp_ood_scores, axis=0)

    # Flip orientation to ID scores to match labels (ID=1, OOD=0)
    odin_id_scores = 1.0 - odin_ood_scores
    msp_id_scores = 1.0 - msp_ood_scores

    # AUROC/AUPR (labels: 1=ISIC ID, 0=Places OOD)
    auroc_odin = float(roc_auc_score(y, odin_id_scores))
    aupr_odin = float(average_precision_score(y, odin_id_scores))
    auroc_msp = float(roc_auc_score(y, msp_id_scores))
    aupr_msp = float(average_precision_score(y, msp_id_scores))
    print(f"ODIN Eval (fixed orientation) AUROC={auroc_odin:.4f}, AUPR={aupr_odin:.4f}")
    print(f"MSP  Eval AUROC={auroc_msp:.4f}, AUPR={aupr_msp:.4f}")

    # Threshold by Youden J on ROC using ID scores
    fpr, tpr, thr = roc_curve(y, odin_id_scores)
    youden = (tpr - fpr)
    best_idx = int(np.argmax(youden))
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5
    y_pred_odin = (odin_id_scores >= best_thr).astype(int)
    cm_odin = confusion_matrix(y, y_pred_odin, labels=[0,1])
    print(f"ODIN confusion matrix at thr={best_thr:.4f} (rows: OOD,ID):\n{cm_odin}")

    # MSP threshold by Youden J
    fpr_m, tpr_m, thr_m = roc_curve(y, msp_id_scores)
    youden_m = (tpr_m - fpr_m)
    best_idx_m = int(np.argmax(youden_m))
    best_thr_m = float(thr_m[best_idx_m]) if best_idx_m < len(thr_m) else 0.5
    y_pred_msp = (msp_id_scores >= best_thr_m).astype(int)
    cm_msp = confusion_matrix(y, y_pred_msp, labels=[0,1])
    print(f"MSP  confusion matrix at thr={best_thr_m:.4f} (rows: OOD,ID):\n{cm_msp}")

    # Save curves and confusion matrix under checkpoint folder
    out_dir = Path(args.checkpoint).parent
    save_curves(odin_id_scores, y, out_dir, prefix='odin')
    save_curves(msp_id_scores, y, out_dir, prefix='msp')
    with open(out_dir / 'confusion_matrix_odin.csv', 'w', newline='') as f:
        w = csv.writer(f); [w.writerow(row.tolist()) for row in cm_odin]
    with open(out_dir / 'confusion_matrix_msp.csv', 'w', newline='') as f:
        w = csv.writer(f); [w.writerow(row.tolist()) for row in cm_msp]

    # CSV + W&B
    append_csv_row(args.run_name, args.wandb_project, auroc_odin, aupr_odin, auroc_msp, aupr_msp)
    log_wandb(args.run_name, args.wandb_project, {
        'exp1_odin_eval_auroc_odin': auroc_odin,
        'exp1_odin_eval_aupr_odin': aupr_odin,
        'exp1_odin_eval_auroc_msp': auroc_msp,
        'exp1_odin_eval_aupr_msp': aupr_msp,
        'temperature': args.temperature,
        'epsilon': args.epsilon,
    })


if __name__ == '__main__':
    main()
