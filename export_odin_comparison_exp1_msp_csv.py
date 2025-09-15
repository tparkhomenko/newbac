#!/usr/bin/env python3
"""
Export a CSV for the W&B project odin_comparison_exp1_msp with requested columns.

Columns:
  Run ID, Architecture, Dataset Setup, Loss Function, Augmentation,
  Test Skin Acc, Test Skin F1, Test Lesion Acc, Test Lesion F1, Test BM Acc, Test BM F1
"""

import os
import csv
from datetime import datetime

def main() -> None:
    try:
        import wandb
        try:
            api = wandb.Api()
        except Exception:
            from wandb import Api
            api = Api()
    except Exception as e:
        print("wandb not available. Please install and login.")
        return

    project = "taisiyaparkhomenko-technische-universit-t-graz/odin_comparison_exp1_msp"
    try:
        runs = list(api.runs(project))
    except Exception as e:
        print(f"Failed to access project {project}: {e}")
        return

    want_names = {"exp1_parallel_lmf_no_odin_msp", "exp1_parallel_lmf_with_odin_msp"}
    filtered = [r for r in runs if r.name in want_names]

    out_path = os.path.join("backend", "experiments_summary_msp.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = [
        "Run ID","Architecture","Dataset Setup","Loss Function","Augmentation",
        "Test Skin Acc","Test Skin F1","Test Lesion Acc","Test Lesion F1","Test BM Acc","Test BM F1",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for run in sorted(filtered, key=lambda r: r.name):
            cfg = run.config or {}
            arch = cfg.get("architecture") or "parallel"
            dataset = cfg.get("experiment") or cfg.get("dataset") or "exp1"
            loss_name = (cfg.get("loss_function") or cfg.get("lesion_loss_fn") or ("LMF" if "lmf" in run.name else "")).lower()
            if loss_name == "lmf":
                loss_disp = "LMF (alpha=0.5, beta=0.5, gamma=2.0)"
            elif loss_name == "weighted_ce":
                loss_disp = "weighted CE"
            elif loss_name == "focal":
                gamma = cfg.get("focal_gamma", 2.0)
                loss_disp = f"focal (gamma={gamma})"
            else:
                loss_disp = cfg.get("loss_function") or cfg.get("lesion_loss_fn") or ""
            bm = cfg.get("balanced_mixup", {})
            aug = "Balanced MixUp (alpha=0.2)" if isinstance(bm, dict) and bm.get("enabled") else "None"
            summ = run.summary or {}
            row = {
                "Run ID": run.name,
                "Architecture": arch,
                "Dataset Setup": dataset,
                "Loss Function": loss_disp,
                "Augmentation": aug,
                "Test Skin Acc": float(summ.get("test_skin_acc", 0.0)),
                "Test Skin F1": float(summ.get("test_skin_f1", 0.0)),
                "Test Lesion Acc": float(summ.get("test_lesion_acc", 0.0)),
                "Test Lesion F1": float(summ.get("test_lesion_f1", 0.0)),
                "Test BM Acc": float(summ.get("test_bm_acc", 0.0)),
                "Test BM F1": float(summ.get("test_bm_f1", 0.0)),
            }
            w.writerow(row)
    print(f"Wrote: {out_path} ({len(filtered)} runs)")

if __name__ == "__main__":
    if not os.path.exists(os.path.expanduser("~/.netrc")) and not os.environ.get("WANDB_API_KEY"):
        print("⚠️  WandB not configured. Please run 'wandb login' first or set WANDB_API_KEY.")
    else:
        main()


