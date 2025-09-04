#!/usr/bin/env python3
"""
Create CSV file for losses_with_mixup_comparison models based on WandB logs
"""

import csv
import re
from datetime import datetime

def extract_model_info_from_logs():
    """Extract model information from the log file and create CSV"""
    
    log_file = "losses_with_mixup_comparison_all_logs.txt"
    
    # CSV headers based on template
    headers = [
        "Run ID", "Architecture", "Dataset Setup", "Loss Function", 
        "Augmentation", "Test Skin Acc", "Test Skin F1", "Test Lesion Acc", 
        "Test Lesion F1", "Test BM Acc", "Test BM F1", "Notes"
    ]
    
    models_data = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into runs
        run_sections = content.split("RUN ")
        
        for section in run_sections[1:]:  # Skip first empty section
            try:
                # Extract run information
                lines = section.split('\n')
                
                # Get run name and ID
                run_name = ""
                run_id = ""
                
                # Extract run name from the first line (format: "1: exp1_multi_8classes_weighted_ce_balanced_mixup")
                if lines:
                    first_line = lines[0]
                    if ":" in first_line:
                        run_name = first_line.split(":", 1)[1].strip()
                
                # Extract run ID from the section
                for line in lines[:10]:
                    if ":" in line and "RUN" not in line:
                        if "Run ID:" in line:
                            run_id = line.split("Run ID:")[1].strip()
                
                # Parse run name to extract information
                run_name_lower = run_name.lower()
                
                # Determine architecture type from run name
                if "multi" in run_name_lower:
                    arch_type = "multi"
                elif "parallel" in run_name_lower:
                    arch_type = "parallel"
                else:
                    arch_type = "unknown"
                
                # Determine dataset type
                if "8classes" in run_name_lower:
                    dataset_type = "exp1_8classes"
                elif "full" in run_name_lower:
                    dataset_type = "exp1_full"
                elif "balanced" in run_name_lower:
                    dataset_type = "exp1_balanced"
                else:
                    dataset_type = "exp1"
                
                # Determine loss function
                if "weighted_ce" in run_name_lower:
                    loss_type = "weighted CE"
                elif "focal" in run_name_lower:
                    loss_type = "focal"
                elif "lmf" in run_name_lower:
                    loss_type = "LMF"
                else:
                    loss_type = "unknown"
                
                # Determine augmentation
                if "mixup" in run_name_lower:
                    aug_type = "MixUp"
                else:
                    aug_type = "None"
                
                # Extract test metrics from the end of the run
                test_skin_acc = "N/A"
                test_skin_f1 = "N/A"
                test_lesion_acc = "N/A"
                test_lesion_f1 = "N/A"
                test_bm_acc = "N/A"
                test_bm_f1 = "N/A"
                
                # Look for test metrics in the last part of the section
                test_patterns = {
                    'test_skin_acc': r'test_skin_acc:\s*([\d.]+)',
                    'test_skin_f1_macro': r'test_skin_f1_macro:\s*([\d.]+)',
                    'test_lesion_acc': r'test_lesion_acc:\s*([\d.]+)',
                    'test_lesion_f1_macro': r'test_lesion_f1_macro:\s*([\d.]+)',
                    'test_bm_acc': r'test_bm_acc:\s*([\d.]+)',
                    'test_bm_f1_macro': r'test_bm_f1_macro:\s*([\d.]+)'
                }
                
                for metric, pattern in test_patterns.items():
                    match = re.search(pattern, section)
                    if match:
                        value = float(match.group(1))
                        if metric == 'test_skin_acc':
                            test_skin_acc = f"{value:.5f}"
                        elif metric == 'test_skin_f1_macro':
                            test_skin_f1 = f"{value:.5f}"
                        elif metric == 'test_lesion_acc':
                            test_lesion_acc = f"{value:.5f}"
                        elif metric == 'test_lesion_f1_macro':
                            test_lesion_f1 = f"{value:.5f}"
                        elif metric == 'test_bm_acc':
                            test_bm_acc = f"{value:.5f}"
                        elif metric == 'test_bm_f1_macro':
                            test_bm_f1 = f"{value:.5f}"
                
                # Create model data row
                model_data = [
                    run_name,
                    arch_type,
                    dataset_type,
                    loss_type,
                    aug_type,
                    test_skin_acc,
                    test_skin_f1,
                    test_lesion_acc,
                    test_lesion_f1,
                    test_bm_acc,
                    test_bm_f1,
                    f"Run ID: {run_id}"
                ]
                
                models_data.append(model_data)
                print(f"Processed: {run_name} -> {arch_type}, {dataset_type}, {loss_type}, {aug_type}")
                
            except Exception as e:
                print(f"Error processing section: {e}")
                continue
        
        # Sort models by name for better organization
        models_data.sort(key=lambda x: x[0])
        
        # Write to CSV
        output_file = "losses_with_mixup_comparison_summary.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(headers)
            writer.writerows(models_data)
        
        print(f"\n✅ Successfully created CSV: {output_file}")
        print(f"Total models processed: {len(models_data)}")
        
        return models_data
        
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None

if __name__ == "__main__":
    extract_model_info_from_logs()





