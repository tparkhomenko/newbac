#!/usr/bin/env python3
"""
Export stats from WandB:
- Summary table (tab-delimited) in the format used for comparisons
- Per-epoch logs (easy to plot) for each run
- Confusion matrix CSVs (if present as run files)

Usage:
  python scripts/export_stats_from_wandb.py \
    --project taisiyaparkhomenko-technische-universit-t-graz/exp6_balanced_partial \
    --outdir exports/exp6_balanced_partial

Requires: pip install wandb
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Any

from wandb.apis import public


def infer_architecture(run_name: str) -> str:
    name = (run_name or '').lower()
    if 'parallel' in name:
        return 'parallel'
    if 'multi' in name:
        return 'multi'
    return 'unknown'


def infer_augmentation(run_name: str, cfg: Dict[str, Any]) -> str:
    name = (run_name or '').lower()
    if 'mixup' in name:
        return 'Balanced MixUp (alpha=0.2)'
    bm = (cfg or {}).get('balanced_mixup', {}) or {}
    if bool(bm.get('enabled', False)):
        return f"Balanced MixUp (alpha={bm.get('alpha', 0.2)})"
    return 'None'


def infer_loss(cfg: Dict[str, Any]) -> str:
    loss = (cfg or {}).get('lesion_loss_fn', '')
    loss = (str(loss) if loss is not None else '').lower().strip()
    if loss == 'cb_focal':
        beta = (cfg or {}).get('cb_beta', 0.9999)
        gamma = (cfg or {}).get('focal_gamma', 2.0)
        return f"cb_focal (beta={beta}, gamma={gamma})"
    if loss == 'focal':
        gamma = (cfg or {}).get('focal_gamma', 2.0)
        return f"focal (gamma={gamma})"
    if loss == 'weighted_ce':
        return 'weighted CE'
    if loss == 'lmf':
        return 'LMF (alpha=0.5, beta=0.5, gamma=2.0)'
    return loss or 'base'


def export(project: str, outdir: Path) -> Path:
    api = public.Api()
    runs = list(api.runs(project))
    outdir.mkdir(parents=True, exist_ok=True)

    # Summary table
    summary_headers = [
        'Run ID', 'Architecture', 'Dataset Setup', 'Loss Function', 'Augmentation',
        'Test Skin Acc', 'Test Skin F1', 'Test Lesion Acc', 'Test Lesion F1', 'Test BM Acc', 'Test BM F1', 'Notes'
    ]
    project_tag = project.split('/')[-1]
    summary_path = outdir / f"{project_tag}_summary_fixed.csv"
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(summary_headers)
        for r in runs:
            cfg = r.config or {}
            summ = r.summary or {}
            arch = infer_architecture(r.name or r.id)
            dataset_setup = project_tag
            loss_fn = infer_loss(cfg)
            aug = infer_augmentation(r.name or '', cfg)
            skin_acc = float(summ.get('test_skin_acc', 0.0) or 0.0)
            skin_f1 = float(summ.get('test_skin_f1_macro', summ.get('test_skin_f1', 0.0)) or 0.0)
            lesion_acc = float(summ.get('test_lesion_acc', 0.0) or 0.0)
            lesion_f1 = float(summ.get('test_lesion_f1_macro', summ.get('test_lesion_f1', 0.0)) or 0.0)
            bm_acc = float(summ.get('test_bm_acc', 0.0) or 0.0)
            bm_f1 = float(summ.get('test_bm_f1_macro', summ.get('test_bm_f1', 0.0)) or 0.0)
            w.writerow([
                r.name or r.id,
                arch,
                dataset_setup,
                loss_fn,
                aug,
                f"{skin_acc:.5f}",
                f"{skin_f1:.5f}",
                f"{lesion_acc:.5f}",
                f"{lesion_f1:.5f}",
                f"{bm_acc:.5f}",
                f"{bm_f1:.5f}",
                ''
            ])

    # Per-epoch logs and confusion matrices per run
    for r in runs:
        run_dir = outdir / (r.name or r.id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Export full history with all keys (easy plotting)
        hist_path = run_dir / 'history.csv'
        try:
            history = r.history(pandas=True)
            if history is not None and not history.empty:
                history.to_csv(hist_path, index=False)
        except Exception:
            pass

        # Attempt to fetch any confusion matrix CSVs saved as files
        try:
            files = list(r.files())
            for fobj in files:
                fname = fobj.name or ''
                if 'confusion_matrix' in fname and fname.endswith('.csv'):
                    dest = run_dir / os.path.basename(fname)
                    fobj.download(root=str(run_dir), replace=True)
        except Exception:
            pass

    return summary_path


def main():
    parser = argparse.ArgumentParser(description='Export stats and logs from a WandB project.')
    parser.add_argument('--project', required=True, help='Entity/Project, e.g., user/project_name')
    parser.add_argument('--outdir', default=None, help='Output directory (default: exports/<project_name>)')
    args = parser.parse_args()

    project_tag = args.project.split('/')[-1]
    outdir = Path(args.outdir) if args.outdir else Path('exports') / project_tag
    out_path = export(args.project, outdir)
    print(f"Wrote summary: {out_path}")
    print(f"Per-run logs and confusion matrices saved under: {outdir}/<run_name>/")


if __name__ == '__main__':
    main()


