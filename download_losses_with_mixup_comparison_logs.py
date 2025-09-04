#!/usr/bin/env python3
"""
Download full training logs with epochs from WandB losses_with_mixup_comparison project
"""

import wandb
import pandas as pd
import os
from datetime import datetime

def download_losses_with_mixup_comparison_logs():
    """Download all training logs from losses_with_mixup_comparison project"""
    
    print("Connecting to WandB...")
    
    # Try different import approaches
    try:
        from wandb import Api
        api = Api()
    except ImportError:
        try:
            api = wandb.Api()
        except AttributeError:
            print("❌ WandB API not available. Please install wandb: pip install wandb")
            return None
    
    # Get the project
    project_name = "taisiyaparkhomenko-technische-universit-t-graz/losses_with_mixup_comparison"
    
    try:
        runs = api.runs(project_name)
        print(f"Found {len(runs)} runs in project: {project_name}")
        
        all_logs = []
        
        for i, run in enumerate(runs):
            print(f"\nProcessing run {i+1}/{len(runs)}: {run.name}")
            print(f"Run ID: {run.id}")
            print(f"Status: {run.state}")
            
            # Get run configuration
            config = run.config
            print(f"Architecture: {config.get('architecture', 'N/A')}")
            print(f"Dataset: {config.get('dataset', 'N/A')}")
            print(f"Loss: {config.get('loss_function', 'N/A')}")
            print(f"Mixup: {config.get('mixup', 'N/A')}")
            
            # Get training history (all epochs)
            try:
                history = run.history()
                print(f"Training epochs: {len(history)}")
                
                # Format the logs
                run_log = f"""
================================================================================
RUN {i+1}: {run.name}
================================================================================
Run ID: {run.id}
Status: {run.state}
Created: {run.created_at}
Architecture: {config.get('architecture', 'N/A')}
Dataset: {config.get('dataset', 'N/A')}
Loss Function: {config.get('loss_function', 'N/A')}
Mixup: {config.get('mixup', 'N/A')}
Learning Rate: {config.get('learning_rate', 'N/A')}
Batch Size: {config.get('batch_size', 'N/A')}
Epochs: {config.get('epochs', 'N/A')}

TRAINING HISTORY (All Epochs):
================================================================================
"""
                
                # Add each epoch's metrics
                for epoch_idx, row in history.iterrows():
                    epoch_log = f"Epoch {epoch_idx+1}"
                    
                    # Add all available metrics
                    for col in row.index:
                        if pd.notna(row[col]):
                            try:
                                # Handle different data types safely
                                if isinstance(row[col], (int, float)):
                                    epoch_log += f" | {col} {row[col]:.5f}"
                                else:
                                    epoch_log += f" | {col} {str(row[col])}"
                            except:
                                epoch_log += f" | {col} {str(row[col])}"
                    
                    run_log += epoch_log + "\n"
                
                # Add final metrics
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
                            except:
                                run_log += f"{col}: {str(final_metrics[col])}\n"
                
                all_logs.append(run_log)
                
            except Exception as e:
                print(f"Error getting history for run {run.id}: {e}")
                # Add basic run info even if history fails
                run_log = f"""
================================================================================
RUN {i+1}: {run.name}
================================================================================
Run ID: {run.id}
Status: {run.state}
Created: {run.created_at}
Architecture: {config.get('architecture', 'N/A')}
Dataset: {config.get('dataset', 'N/A')}
Loss Function: {config.get('loss_function', 'N/A')}
Mixup: {config.get('mixup', 'N/A')}

ERROR: Could not retrieve training history
Error: {str(e)}
================================================================================
"""
                all_logs.append(run_log)
        
        # Combine all logs
        combined_logs = f"""
================================================================================
                    LOSSES WITH MIXUP COMPARISON - FULL WANDB TRAINING LOGS
================================================================================

Downloaded from WandB project: {project_name}
Download timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total runs found: {len(runs)}

This file contains the complete training logs with all epochs from the WandB project.
Each run includes detailed metrics for every training epoch.

================================================================================
"""
        
        combined_logs += "\n".join(all_logs)
        
        # Save to file
        output_file = "losses_with_mixup_comparison_all_logs.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_logs)
        
        print(f"\n✅ Successfully downloaded logs to: {output_file}")
        print(f"Total runs processed: {len(runs)}")
        
        return combined_logs, len(runs)
        
    except Exception as e:
        print(f"Error accessing WandB project: {e}")
        return None, 0

if __name__ == "__main__":
    # Check if WandB is configured
    if not os.path.exists(os.path.expanduser("~/.netrc")):
        print("⚠️  WandB not configured. Please run 'wandb login' first.")
        print("Or set WANDB_API_KEY environment variable.")
    else:
        download_losses_with_mixup_comparison_logs()





