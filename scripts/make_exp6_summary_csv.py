#!/usr/bin/env python3
"""
Generate a summary CSV (tab-delimited) for the latest Exp6 runs, matching
the format of losses_with_mixup_comparison_summary_fixed.csv.
"""

import csv
from datetime import datetime
import wandb


def main():
    project = "taisiyaparkhomenko-technische-universit-t-graz/exp6_balanced_comparison"
    run_names = [
        "exp6_parallel_weightedce_balanced_mixup",
        "exp6_multi_weightedce_balanced_mixup",
    ]

    headers = [
        "Run ID",
        "Architecture",
        "Dataset Setup",
        "Loss Function",
        "Augmentation",
        "Test Skin Acc",
        "Test Skin F1",
        "Test Lesion Acc",
        "Test Lesion F1",
        "Test BM Acc",
        "Test BM F1",
        "Notes",
    ]

    api = wandb.Api()
    runs = {r.name: r for r in api.runs(project)}

    rows = []
    for name in run_names:
        r = runs.get(name)
        if r is None:
            continue
        summ = r.summary._json_dict

        # Infer architecture and dataset setup
        arch = "parallel" if "parallel" in name else ("multi" if "multi" in name else "unknown")
        dataset_setup = "exp6_balanced"

        # Loss function and augmentation (from config/known settings)
        # Both runs used focal according to logs; keep explicit gamma for clarity
        loss_fn = "focal (gamma=2.0)"
        aug = "Balanced MixUp (alpha=0.2)"

        # Pull metrics (macro F1 if available, else plain key)
        skin_acc = float(summ.get("test_skin_acc", 0.0))
        skin_f1 = float(summ.get("test_skin_f1", summ.get("test_skin_f1_macro", 0.0)))
        lesion_acc = float(summ.get("test_lesion_acc", 0.0))
        lesion_f1 = float(summ.get("test_lesion_f1", summ.get("test_lesion_f1_macro", 0.0)))
        bm_acc = float(summ.get("test_bm_acc", 0.0))
        bm_f1 = float(summ.get("test_bm_f1", summ.get("test_bm_f1_macro", 0.0)))

        rows.append([
            name,
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
            "",
        ])

    out_path = "exp6_balanced_comparison_summary_fixed.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(headers)
        for row in rows:
            w.writerow(row)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()


