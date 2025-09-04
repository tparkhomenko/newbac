import re
import csv
import os
import glob


def collect_rows() -> list[dict]:
    base = os.path.join(os.path.dirname(__file__), "..")
    
    # Search in multiple directories for logs
    log_paths = []
    
    # Primary logs from backend/models
    log_paths.extend(sorted(
        glob.glob(os.path.join(base, "backend/models/multi/logs/*.txt"))
        + glob.glob(os.path.join(base, "backend/models/parallel/logs/*.txt"))
    ))
    
    # Additional logs from logs/runs directory
    log_paths.extend(sorted(
        glob.glob(os.path.join(base, "logs/runs/*.log"))
    ))
    
    # Wandb output logs
    log_paths.extend(sorted(
        glob.glob(os.path.join(base, "wandb/run-*/files/output.log"))
    ))

    rows: list[dict] = []
    
    # Patterns for different log file naming conventions
    patterns = [
        # With MixUp (6 runs)
        r"^logs_(multi|parallel)_exp(\d+)_8classes_(weighted_ce|focal|lmf)_balanced_mixup\.txt$",
        # Without MixUp (6 runs) 
        r"^logs_(multi|parallel)_exp(\d+)_8classes_(weighted_ce|focal|lmf)\.txt$",
        # Base runs (6 runs)
        r"^logs_(multi|parallel)_exp(\d+)_8classes\.txt$",
        # Alternative naming patterns from logs/runs
        r"^exp(\d+)_(multihead|parallel)_e50_.*\.log$",
        r"^exp(\d+)_(multihead|parallel)_retrain_.*\.log$",
        r"^retrain_all_8classes_.*\.log$",
    ]
    
    loss_map = {
        "weighted_ce": "weighted CE",
        "focal": "focal (gamma=2.0)",
        "lmf": "LMF (alpha=0.5, beta=0.5, gamma=2.0)",
    }
    
    # Map multihead -> multi for consistency
    arch_map = {
        "multihead": "multi",
        "parallel": "parallel"
    }

    for path in log_paths:
        fname = os.path.basename(path)
        matched = False
        
        for pattern in patterns:
            m = re.match(pattern, fname)
            if m:
                matched = True
                
                if "logs_" in fname:
                    # Pattern 1-3: backend/models logs
                    arch, exp_num = m.groups()[:2]
                    exp = f"exp{exp_num}"
                    
                    # Determine loss function and augmentation
                    if "_balanced_mixup" in fname:
                        # Pattern 1: with mixup
                        loss_key = m.group(3)
                        augmentation = "Balanced MixUp (alpha=0.2)"
                        run_id = f"{exp}_{arch}_8classes_{loss_key}_balanced_mixup"
                    elif len(m.groups()) >= 3 and m.group(3) in loss_map:
                        # Pattern 2: without mixup
                        loss_key = m.group(3)
                        augmentation = "None"
                        run_id = f"{exp}_{arch}_8classes_{loss_key}"
                    else:
                        # Pattern 3: base runs
                        loss_key = "base"
                        augmentation = "None"
                        run_id = f"{exp}_{arch}_8classes"
                    
                    loss_friendly = loss_map.get(loss_key, "base")
                    
                elif "exp" in fname and ("multihead" in fname or "parallel" in fname):
                    # Pattern 4-5: logs/runs directory
                    exp_num, arch_raw = m.groups()[:2]
                    exp = f"exp{exp_num}"
                    arch = arch_map.get(arch_raw, arch_raw)
                    loss_key = "base"
                    augmentation = "None"
                    run_id = f"{exp}_{arch}_8classes"
                    loss_friendly = "base"
                    
                elif "retrain_all_8classes" in fname:
                    # Pattern 6: retrain log (contains multiple runs)
                    # This is a special case - we'll extract multiple runs from it
                    continue  # Handle separately
                
                # Extract test metrics
                metrics = {
                    "skin_acc": None,
                    "skin_f1": None,
                    "lesion_acc": None,
                    "lesion_f1": None,
                    "bm_acc": None,
                    "bm_f1": None,
                }

                with open(path, "r", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        # Handle both standard and macro F1 formats
                        m1 = re.search(r"test_skin_acc\s+([0-9.]+)", line)
                        if m1:
                            metrics["skin_acc"] = float(m1.group(1))
                            continue
                        m1 = re.search(r"test_skin_f1(?:_macro)?\s+([0-9.]+)", line)
                        if m1:
                            metrics["skin_f1"] = float(m1.group(1))
                            continue
                        m1 = re.search(r"test_lesion_acc\s+([0-9.]+)", line)
                        if m1:
                            metrics["lesion_acc"] = float(m1.group(1))
                            continue
                        m1 = re.search(r"test_lesion_f1(?:_macro)?\s+([0-9.]+)", line)
                        if m1:
                            metrics["lesion_f1"] = float(m1.group(1))
                            continue
                        m1 = re.search(r"test_bm_acc\s+([0-9.]+)", line)
                        if m1:
                            metrics["bm_acc"] = float(m1.group(1))
                            continue
                        m1 = re.search(r"test_bm_f1(?:_macro)?\s+([0-9.]+)", line)
                        if m1:
                            metrics["bm_f1"] = float(m1.group(1))
                            continue

                rows.append(
                    {
                        "Run ID": run_id,
                        "Architecture": arch,
                        "Dataset Setup": exp,
                        "Loss Function": loss_friendly,
                        "Augmentation": augmentation,
                        "Test Skin Acc": metrics["skin_acc"],
                        "Test Skin F1": metrics["skin_f1"],
                        "Test Lesion Acc": metrics["lesion_acc"],
                        "Test Lesion F1": metrics["lesion_f1"],
                        "Test BM Acc": metrics["bm_acc"],
                        "Test BM F1": metrics["bm_f1"],
                        "Notes": "",
                    }
                )
                break
        
        if not matched:
            print(f"Skipping: {fname}")

    # Add the successful retry run manually
    retry_run = {
        "Run ID": "exp1_parallel_8classes_lmf_balanced_mixup",
        "Architecture": "parallel",
        "Dataset Setup": "exp1",
        "Loss Function": "LMF (alpha=0.5, beta=0.5, gamma=2.0)",
        "Augmentation": "Balanced MixUp (alpha=0.2)",
        "Test Skin Acc": 0.9841,
        "Test Skin F1": 0.9762,
        "Test Lesion Acc": 0.5130,
        "Test Lesion F1": 0.5550,
        "Test BM Acc": 0.6863,
        "Test BM F1": 0.5587,
        "Notes": "Retry run - successful",
    }
    rows.append(retry_run)

    return rows


def main() -> None:
    rows = collect_rows()
    out_path = os.path.join(os.path.dirname(__file__), "experiments_summary.csv")
    fields = [
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
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Expected: 18 runs (6 with MixUp + 6 without MixUp + 6 base)")
    if len(rows) != 18:
        print(f"Warning: Got {len(rows)} rows instead of 18")
    
    # Show unique run IDs for verification
    run_ids = set(r["Run ID"] for r in rows)
    print(f"\nUnique run IDs found ({len(run_ids)}):")
    for rid in sorted(run_ids):
        print(f"  - {rid}")


if __name__ == "__main__":
    main()
