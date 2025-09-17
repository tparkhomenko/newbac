#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT))

from datasets.parallel_unified_dataset import ParallelUnifiedDataset
from models.multitask_model import MultiTaskHead
from utils.lesion_mapping import LESION_CLASS_MAPPING, lesion_requires_bm


IDX_TO_LESION = {0:'melanoma',1:'nevus',2:'bcc',3:'scc',4:'seborrheic_keratosis',5:'akiec',6:'lentigo',7:'vasc'}


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultiTaskHead(
        input_dim=256,
        hidden_dims=(512, 256),
        dropout=0.3,
        num_classes_skin=2,
        num_classes_lesion=8,
        num_classes_bm=2,
        num_classes_final=11,  # Add final head for exp_finalmulticlass
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def compute_lesion_confusion(model: torch.nn.Module, dataloader, device: torch.device) -> Tuple[np.ndarray, List[int], List[int]]:
    all_preds: List[int] = []
    all_tgts: List[int] = []
    with torch.no_grad():
        for features, labels, masks, _ in tqdm(dataloader, desc='Lesion head eval'):
            features = features.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            outputs = model(features)
            valid_idx = masks['lesion'].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            logits = outputs['lesion'][valid_idx]
            pred = logits.argmax(dim=1)
            tgt = labels['lesion'][valid_idx]
            all_preds.extend(pred.cpu().numpy().tolist())
            all_tgts.extend(tgt.cpu().numpy().tolist())
    if not all_tgts:
        return np.zeros((8, 8), dtype=int), all_tgts, all_preds
    cm = confusion_matrix(all_tgts, all_preds, labels=list(range(8)))
    return cm, all_tgts, all_preds


def lesion_mapping_requires_bm(lesion: str, override_map: Dict[str,str]) -> bool:
    if override_map:
        return (override_map.get(lesion, 'bm') == 'bm')
    return lesion_requires_bm(lesion)


def compute_pipeline_confusion(model: torch.nn.Module, dataloader, device: torch.device, lesion_map: Dict[str,str]) -> Tuple[np.ndarray, Dict[str,int]]:
    label_to_idx: Dict[str,int] = {}
    preds_idx: List[int] = []
    tgts_idx: List[int] = []
    with torch.no_grad():
        for features, labels, masks, _ in tqdm(dataloader, desc='Pipeline eval'):
            features = features.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            outputs = model(features)
            skin_logits = outputs['skin']
            lesion_logits = outputs['lesion']
            bm_logits = outputs['bm']

            B = features.shape[0]
            for i in range(B):
                if masks['skin'][i].item() == 0:
                    continue
                # Prediction path
                skin_pred = skin_logits[i].argmax().item()
                if skin_pred == 0:
                    pred_label = 'not_skin'
                else:
                    lesion_idx = lesion_logits[i].argmax().item()
                    lesion_name = IDX_TO_LESION.get(lesion_idx, 'unknown')
                    if lesion_mapping_requires_bm(lesion_name, lesion_map):
                        if masks['bm'][i].item() == 0:
                            bm_name = 'benign'
                        else:
                            bm_pred = bm_logits[i].argmax().item()
                            bm_name = 'malignant' if bm_pred == 1 else 'benign'
                        pred_label = f"{bm_name}_{lesion_name}"
                    else:
                        pred_label = f"benign_{lesion_name}"

                # Ground-truth path (best-effort)
                gt_skin = int(labels['skin'][i].item())
                if gt_skin == 0:
                    gt_label = 'not_skin'
                else:
                    if masks['lesion'][i].item() == 1:
                        gt_lesion_idx = int(labels['lesion'][i].item())
                        gt_lesion_name = IDX_TO_LESION.get(gt_lesion_idx, 'unknown')
                        if lesion_mapping_requires_bm(gt_lesion_name, lesion_map) and masks['bm'][i].item() == 1:
                            bm_val = int(labels['bm'][i].item())
                            bm_name = 'malignant' if bm_val == 1 else 'benign'
                            gt_label = f"{bm_name}_{gt_lesion_name}"
                        else:
                            gt_label = f"benign_{gt_lesion_name}"
                    else:
                        gt_label = 'unknown'

                for lab in (pred_label, gt_label):
                    if lab not in label_to_idx:
                        label_to_idx[lab] = len(label_to_idx)
                preds_idx.append(label_to_idx[pred_label])
                tgts_idx.append(label_to_idx[gt_label])

    if not tgts_idx:
        return np.zeros((1, 1), dtype=int), label_to_idx
    cm = confusion_matrix(tgts_idx, preds_idx, labels=list(range(len(label_to_idx))))
    return cm, label_to_idx


def save_confusion_csv(cm: np.ndarray, labels: List[str], out_csv: Path) -> None:
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([''] + labels)
        for i, row in enumerate(cm.tolist()):
            w.writerow([labels[i]] + row)


def plot_confusion_png(cm: np.ndarray, labels: List[str], title: str, out_png: Path, normalize_rows: bool = True) -> None:
    if normalize_rows and cm.sum(axis=1).any():
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = (cm / row_sums) * 100.0
    else:
        cm_norm = cm.astype(float)
    df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    plt.figure(figsize=(max(8, len(labels)*0.6), max(6, len(labels)*0.6)))
    ax = sns.heatmap(df, annot=True, fmt='.1f', cmap='Blues', cbar=True)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description='Plot lesion and pipeline confusion matrices on Exp1 test split')
    ap.add_argument('--checkpoint', required=True, help='Path to model .pt file')
    ap.add_argument('--wandb_project', default=None, help='Optional W&B project to log images')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(args.checkpoint)
    ckpt_dir = ckpt_path.parent
    model = load_model(ckpt_path, device)

    dataset = ParallelUnifiedDataset(experiment='exp1', split='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    # Lesion head confusion
    lesion_cm, lesion_tgts, lesion_preds = compute_lesion_confusion(model, dataloader, device)
    lesion_labels = [IDX_TO_LESION[i] for i in range(8)]
    lesion_csv = ckpt_dir / 'lesion_confusion.csv'
    lesion_png = ckpt_dir / 'lesion_confusion.png'
    save_confusion_csv(lesion_cm, lesion_labels, lesion_csv)
    plot_confusion_png(lesion_cm, lesion_labels, 'Lesion Head Confusion Matrix', lesion_png, normalize_rows=True)

    # Pipeline confusion
    pipeline_cm, label_to_idx = compute_pipeline_confusion(model, dataloader, device, LESION_CLASS_MAPPING)
    pipeline_labels = [k for k,_ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
    pipeline_csv = ckpt_dir / 'pipeline_confusion.csv'
    pipeline_png = ckpt_dir / 'pipeline_confusion.png'
    save_confusion_csv(pipeline_cm, pipeline_labels, pipeline_csv)
    plot_confusion_png(pipeline_cm, pipeline_labels, 'Pipeline Confusion Matrix', pipeline_png, normalize_rows=True)

    # Compute simple summaries
    lesion_acc = float((lesion_cm.trace() / max(1, lesion_cm.sum()))) if lesion_cm.size else 0.0
    lesion_f1 = float(f1_score(lesion_tgts, lesion_preds, average='weighted')) if lesion_tgts else 0.0
    pipeline_acc = float(pipeline_cm.trace() / max(1, pipeline_cm.sum())) if pipeline_cm.size else 0.0
    # Macro F1 for pipeline requires numeric labels already computed
    if pipeline_cm.size:
        # Reconstruct preds/tgts from cm is non-trivial; skip F1 if not computed directly
        pipeline_f1 = None
    else:
        pipeline_f1 = None

    # Log to W&B if requested
    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=f'plot_confusions_{ckpt_dir.name}')
            wandb.log({
                'lesion_confusion_png': wandb.Image(str(lesion_png)),
                'pipeline_confusion_png': wandb.Image(str(pipeline_png)),
                'lesion_acc_from_cm': lesion_acc,
                'lesion_f1_weighted': lesion_f1,
                'pipeline_acc_from_cm': pipeline_acc,
            })
            run.finish()
        except Exception as e:
            print(f"[WARN] W&B logging failed: {e}")

    # Append to experiments_summary.csv
    out_csv = ROOT / 'backend' / 'experiments_summary.csv'
    write_header = not out_csv.exists()
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                'timestamp','task','checkpoint','lesion_acc','lesion_f1_weighted','pipeline_acc','pipeline_f1',
                'lesion_confusion_png','lesion_confusion_csv','pipeline_confusion_png','pipeline_confusion_csv'
            ])
        import datetime
        w.writerow([
            datetime.datetime.now().isoformat(timespec='seconds'),
            'plot_confusions',
            str(ckpt_path),
            lesion_acc,
            lesion_f1,
            pipeline_acc,
            pipeline_f1,
            str(lesion_png),
            str(lesion_csv),
            str(pipeline_png),
            str(pipeline_csv),
        ])


if __name__ == '__main__':
    main()



