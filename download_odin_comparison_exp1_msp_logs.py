#!/usr/bin/env python3
"""
Download full training logs with epochs from WandB odin_comparison_exp1_msp project
"""

import wandb
import pandas as pd
import os
from datetime import datetime

def download_odin_comparison_exp1_msp_logs():
    """Download all training logs from odin_comparison_exp1_msp project"""
    print("Connecting to WandB...")
    try:
        from wandb import Api
        api = Api()
    except ImportError:
        try:
            api = wandb.Api()
        except AttributeError:
            print("❌ WandB API not available. Please install wandb: pip install wandb")
            return None

    project_name = "taisiyaparkhomenko-technische-universit-t-graz/odin_comparison_exp1_msp"

    try:
        runs = api.runs(project_name)
        print(f"Found {len(runs)} runs in project: {project_name}")

        all_logs = []

        for i, run in enumerate(runs):
            print(f"\nProcessing run {i+1}/{len(runs)}: {run.name}")
            print(f"Run ID: {run.id}")
            print(f"Status: {run.state}")

            config = run.config
            print(f"Architecture: {config.get('architecture', 'N/A')}")
            print(f"Dataset: {config.get('dataset', config.get('experiment', 'N/A'))}")
            print(f"Loss: {config.get('loss_function', config.get('lesion_loss_fn', 'N/A'))}")
            print(f"Mixup: {config.get('mixup', config.get('balanced_mixup', 'N/A'))}")

            try:
                history = run.history()
                print(f"Training epochs: {len(history)}")

                run_log = f"""
================================================================================
RUN {i+1}: {run.name}
================================================================================
Run ID: {run.id}
Status: {run.state}
Created: {run.created_at}
Architecture: {config.get('architecture', 'N/A')}
Dataset: {config.get('dataset', config.get('experiment', 'N/A'))}
Loss Function: {config.get('loss_function', config.get('lesion_loss_fn', 'N/A'))}
Mixup: {config.get('mixup', config.get('balanced_mixup', 'N/A'))}
Learning Rate: {config.get('learning_rate', 'N/A')}
Batch Size: {config.get('batch_size', 'N/A')}
Epochs: {config.get('epochs', 'N/A')}

TRAINING HISTORY (All Epochs):
================================================================================
"""

                for epoch_idx, row in history.iterrows():
                    epoch_log = f"Epoch {epoch_idx+1}"
                    for col in row.index:
                        if pd.notna(row[col]):
                            try:
                                if isinstance(row[col], (int, float)):
                                    epoch_log += f" | {col} {row[col]:.5f}"
                                else:
                                    epoch_log += f" | {col} {str(row[col])}"
                            except Exception:
                                epoch_log += f" | {col} {str(row[col])}"
                    run_log += epoch_log + "\n"

                if len(history) > 0:
                    final_metrics = history.iloc[-1]
                    run_log += f"\nFINAL METRICS:\n"
                    for col in final_metrics.index:
                        if pd.notna(final_metrics[col]):
                            try:
                                if isinstance(final_metrics[col], (int, float)):
                                    run_log += f"{col}: {final_metrics[col]:.5f}\n"
                                else:
                                    run_log += f"{col}: {str(final_metrics[col])}\n"
                            except Exception:
                                run_log += f"{col}: {str(final_metrics[col])}\n"

                all_logs.append(run_log)

            except Exception as e:
                print(f"Error getting history for run {run.id}: {e}")
                run_log = f"""
================================================================================
RUN {i+1}: {run.name}
================================================================================
Run ID: {run.id}
Status: {run.state}
Created: {run.created_at}
Architecture: {config.get('architecture', 'N/A')}
Dataset: {config.get('dataset', config.get('experiment', 'N/A'))}
Loss Function: {config.get('loss_function', config.get('lesion_loss_fn', 'N/A'))}
Mixup: {config.get('mixup', config.get('balanced_mixup', 'N/A'))}

ERROR: Could not retrieve training history
Error: {str(e)}
================================================================================
"""
                all_logs.append(run_log)

        combined_logs = f"""
================================================================================
                    ODIN COMPARISON EXP1 MSP - FULL WANDB TRAINING LOGS
================================================================================

Downloaded from WandB project: {project_name}
Download timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total runs found: {len(runs)}

This file contains the complete training logs with all epochs from the WandB project.
Each run includes detailed metrics for every training epoch.

================================================================================
"""
        combined_logs += "\n".join(all_logs)

        output_file = "odin_comparison_exp1_msp_wandb_logs.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_logs)

        print(f"\n✅ Successfully downloaded logs to: {output_file}")
        print(f"Total runs processed: {len(runs)}")
        return combined_logs

    except Exception as e:
        print(f"Error accessing WandB project: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists(os.path.expanduser("~/.netrc")) and not os.environ.get("WANDB_API_KEY"):
        print("⚠️  WandB not configured. Please run 'wandb login' first or set WANDB_API_KEY.")
    else:
        download_odin_comparison_exp1_msp_logs()


