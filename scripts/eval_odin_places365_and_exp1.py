#!/usr/bin/env python3
"""
Evaluate ODIN performance on both Exp1 test set and Places365 OOD dataset.
Computes AUROC/AUPR for ODIN scores and standard classification metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from tqdm import tqdm
import pickle

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from datasets.parallel_unified_dataset import ParallelUnifiedDataset
from models.multitask_model import MultiTaskHead
from utils.odin import compute_odin_scores, compute_msp_scores
import yaml

def load_model(model_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model = MultiTaskHead(
        input_dim=256,
        hidden_dims=(512, 256),
        dropout=0.3,
        num_classes_skin=2,
        num_classes_lesion=8,
        num_classes_bm=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate_model_on_dataset(model, dataset, device='cuda', use_odin=True, temperature=1000, epsilon=0.0014, ood_only=False):
    """Evaluate model on dataset and compute MSP always and ODIN optionally.

    If ood_only=True (e.g., Places365), only compute ODIN metrics and skip
    lesion/bm/skin classification metrics.
    """
    model.eval()
    
    all_skin_preds = []
    all_lesion_preds = []
    all_bm_preds = []
    all_skin_labels = []
    all_lesion_labels = []
    all_bm_labels = []
    all_odin_scores = []
    all_msp_scores = []
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Handle both tuple and dict returns
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                features, labels, masks, image_names = batch
                features = features.to(device)
                skin_labels = labels['skin'].to(device)
                lesion_labels = labels['lesion'].to(device)
                bm_labels = labels['bm'].to(device)
            else:
                features = batch['features'].to(device)
                skin_labels = batch['skin_labels'].to(device)
                lesion_labels = batch['lesion_labels'].to(device)
                bm_labels = batch['bm_labels'].to(device)
            
            # Forward pass
            outputs = model(features)
            skin_logits = outputs['skin']
            lesion_logits = outputs['lesion']
            bm_logits = outputs['bm']
            
            # Get predictions
            skin_preds = F.softmax(skin_logits, dim=1)
            lesion_preds = F.softmax(lesion_logits, dim=1)
            bm_preds = F.softmax(bm_logits, dim=1)
            
            all_skin_preds.append(skin_preds.cpu())
            all_lesion_preds.append(lesion_preds.cpu())
            all_bm_preds.append(bm_preds.cpu())
            all_skin_labels.append(skin_labels.cpu())
            all_lesion_labels.append(lesion_labels.cpu())
            all_bm_labels.append(bm_labels.cpu())
            
            # Compute MSP always; ODIN optionally
            msp_scores, _ = compute_msp_scores(model, features, device, head='skin')
            all_msp_scores.append(msp_scores.cpu())
            if use_odin:
                # Ensure features are float32 and require gradients for ODIN computation
                features_float = features.float().detach().clone()
                features_float.requires_grad_(True)
                
                # Enable gradients for ODIN computation
                with torch.enable_grad():
                    odin_scores, max_probs = compute_odin_scores(
                        model, features_float, device, 
                        temperature=temperature, epsilon=epsilon
                    )
                all_odin_scores.append(odin_scores.cpu())
    
    # Concatenate labels; preds only when not OOD-only
    skin_labels = torch.cat(all_skin_labels, dim=0) if len(all_skin_labels) > 0 else torch.empty(0)
    lesion_labels = torch.cat(all_lesion_labels, dim=0) if len(all_lesion_labels) > 0 else torch.empty(0)
    bm_labels = torch.cat(all_bm_labels, dim=0) if len(all_bm_labels) > 0 else torch.empty(0)

    results = {}
    if not ood_only:
        skin_preds = torch.cat(all_skin_preds, dim=0)
        lesion_preds = torch.cat(all_lesion_preds, dim=0)
        bm_preds = torch.cat(all_bm_preds, dim=0)

        # Compute accuracy and F1 scores
        skin_acc = (skin_preds.argmax(dim=1) == skin_labels).float().mean().item()
        lesion_acc = (lesion_preds.argmax(dim=1) == lesion_labels).float().mean().item()
        bm_acc = (bm_preds.argmax(dim=1) == bm_labels).float().mean().item()

        # Compute F1 scores (macro)
        from sklearn.metrics import f1_score
        skin_f1 = f1_score(skin_labels.numpy(), skin_preds.argmax(dim=1).numpy(), average='macro')
        lesion_f1 = f1_score(lesion_labels.numpy(), lesion_preds.argmax(dim=1).numpy(), average='macro')
        bm_f1 = f1_score(bm_labels.numpy(), bm_preds.argmax(dim=1).numpy(), average='macro')

        results.update({
            'skin_acc': skin_acc,
            'skin_f1': skin_f1,
            'lesion_acc': lesion_acc,
            'lesion_f1': lesion_f1,
            'bm_acc': bm_acc,
            'bm_f1': bm_f1
        })
    
    # Compute MSP metrics always
    if all_msp_scores:
        msp_scores = torch.cat(all_msp_scores, dim=0)
        in_dist_labels = (skin_labels == 1).float()
        try:
            msp_auroc = roc_auc_score(in_dist_labels.numpy(), msp_scores.numpy())
            msp_aupr = average_precision_score(in_dist_labels.numpy(), msp_scores.numpy())
            results['msp_auroc'] = msp_auroc
            results['msp_aupr'] = msp_aupr
        except ValueError as e:
            print(f"Warning: Could not compute MSP metrics: {e}")
            results['msp_auroc'] = 0.0
            results['msp_aupr'] = 0.0
    else:
        results['msp_auroc'] = 0.0
        results['msp_aupr'] = 0.0

    # Compute ODIN metrics if enabled
    if use_odin and all_odin_scores:
        odin_scores = torch.cat(all_odin_scores, dim=0)
        
        # For ODIN, we use skin head predictions as in-distribution vs out-of-distribution
        # In-distribution: skin=1 (skin images), Out-of-distribution: skin=0 (non-skin images)
        in_dist_labels = (skin_labels == 1).float()
        
        # Compute AUROC and AUPR for ODIN scores
        try:
            odin_auroc = roc_auc_score(in_dist_labels.numpy(), odin_scores.numpy())
            odin_aupr = average_precision_score(in_dist_labels.numpy(), odin_scores.numpy())
            results['odin_auroc'] = odin_auroc
            results['odin_aupr'] = odin_aupr
        except ValueError as e:
            print(f"Warning: Could not compute ODIN metrics: {e}")
            results['odin_auroc'] = 0.0
            results['odin_aupr'] = 0.0
    else:
        results['odin_auroc'] = 0.0
        results['odin_aupr'] = 0.0
    
    if not ood_only:
        return results, skin_preds, lesion_preds, bm_preds, skin_labels, lesion_labels, bm_labels
    else:
        return results, torch.empty(0), torch.empty(0), torch.empty(0), skin_labels, lesion_labels, bm_labels

def save_confusion_matrix(preds, labels, class_names, save_path):
    """Save confusion matrix as CSV."""
    cm = confusion_matrix(labels.numpy(), preds.argmax(dim=1).numpy())
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(save_path)
    print(f"Saved confusion matrix to {save_path}")

def save_roc_pr_curves(odin_scores, labels, save_dir):
    """Save ROC and PR curves as PNG files."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    # ROC curve
    fpr, tpr, _ = roc_curve(labels.numpy(), odin_scores.numpy())
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(labels.numpy(), odin_scores.numpy()):.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ODIN ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'odin_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PR curve
    precision, recall, _ = precision_recall_curve(labels.numpy(), odin_scores.numpy())
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR (AP = {average_precision_score(labels.numpy(), odin_scores.numpy()):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('ODIN Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'odin_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main evaluation function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load config
    config_path = root / 'config.yaml'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Default ODIN config
    odin_config = config.get('odin', {})
    temperature = odin_config.get('temperature', 1000.0)
    epsilon = odin_config.get('epsilon', 0.001)
    
    # Model paths - use the latest timestamps
    no_odin_model_path = root / "backend" / "models" / "parallel" / "exp1_odin_comparison" / "no_odin" / "20250912_135028" / "best.pt"
    with_odin_model_path = root / "backend" / "models" / "parallel" / "exp1_odin_comparison" / "with_odin" / "20250912_135744" / "best.pt"
    
    # Check if models exist
    if not no_odin_model_path.exists():
        print(f"Error: No-ODIN model not found at {no_odin_model_path}")
        return
    
    if not with_odin_model_path.exists():
        print(f"Error: With-ODIN model not found at {with_odin_model_path}")
        return
    
    # Load datasets
    print("Loading datasets...")
    
    # Exp1 test set
    exp1_test_dataset = ParallelUnifiedDataset(
        metadata_csv=str(root / "data" / "metadata" / "metadata.csv"),
        features_dir=str(root / "data" / "processed" / "features"),
        experiment="exp1",
        split="test"
    )
    
    # For Places365, we need to create a custom dataset that loads only places365 entries
    # Let's create a simple dataset class for this
    class Places365Dataset(torch.utils.data.Dataset):
        def __init__(self, metadata_csv, features_pkl):
            self.df = pd.read_csv(metadata_csv)
            # Filter for places365 entries
            self.df = self.df[self.df['csv_source'] == 'places365'].reset_index(drop=True)
            
            # Load features
            with open(features_pkl, 'rb') as f:
                self.features = pickle.load(f)
            
            # Filter to only include images that have features
            available_images = set(self.features.keys())
            self.df = self.df[self.df['image_name'].isin(available_images)].reset_index(drop=True)
            
            print(f"Places365 samples with features: {len(self.df)}")
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_name = row['image_name']
            
            # Get features - the features are stored with the full image name
            features = torch.tensor(self.features[image_name], dtype=torch.float32)
            
            # For places365, all are NOT_SKIN, so skin=0, lesion=0, bm=0
            return {
                'features': features,
                'skin_labels': torch.tensor(0, dtype=torch.long),
                'lesion_labels': torch.tensor(0, dtype=torch.long),
                'bm_labels': torch.tensor(0, dtype=torch.long)
            }
    
    places365_test_dataset = Places365Dataset(
        metadata_csv=str(root / "data" / "metadata" / "metadata.csv"),
        features_pkl=str(root / "data" / "processed" / "features" / "sam_features_places365_test.pkl")
    )
    
    print(f"Exp1 test samples: {len(exp1_test_dataset)}")
    print(f"Places365 test samples: {len(places365_test_dataset)}")
    
    # Results storage
    results = []
    
    # Evaluate both models
    for model_name, model_path, use_odin in [
        ("no_odin", no_odin_model_path, False),
        ("with_odin", with_odin_model_path, True)
    ]:
        print(f"\nEvaluating {model_name} model...")
        
        # Load model
        model = load_model(model_path, device)
        
        # Evaluate on Exp1 test set
        print("Evaluating on Exp1 test set...")
        exp1_results, exp1_skin_preds, exp1_lesion_preds, exp1_bm_preds, exp1_skin_labels, exp1_lesion_labels, exp1_bm_labels = evaluate_model_on_dataset(
            model, exp1_test_dataset, device, use_odin=use_odin,
            temperature=temperature,
            epsilon=epsilon
        )
        
        # Evaluate on Places365 test set (OOD)
        print("Evaluating on Places365 test set (OOD)...")
        places365_results, places365_skin_preds, places365_lesion_preds, places365_bm_preds, places365_skin_labels, places365_lesion_labels, places365_bm_labels = evaluate_model_on_dataset(
            model, places365_test_dataset, device, use_odin=use_odin,
            temperature=temperature,
            epsilon=epsilon,
            ood_only=True # Pass ood_only=True for Places365 evaluation
        )
        
        # Store results (Exp1 full + ODIN; Places365 only ODIN)
        result_row = {
            'run_name': f'exp1_parallel_lmf_{model_name}',
            'experiment': 'exp1',
            'architecture': 'parallel',
            'lesion_loss': 'lmf',
            'odin_enabled': use_odin,
            'exp1_test_skin_acc': exp1_results['skin_acc'],
            'exp1_test_skin_f1': exp1_results['skin_f1'],
            'exp1_test_lesion_acc': exp1_results['lesion_acc'],
            'exp1_test_lesion_f1': exp1_results['lesion_f1'],
            'exp1_test_bm_acc': exp1_results['bm_acc'],
            'exp1_test_bm_f1': exp1_results['bm_f1'],
            'exp1_test_msp_auroc': exp1_results.get('msp_auroc', 0.0),
            'exp1_test_msp_aupr': exp1_results.get('msp_aupr', 0.0),
            'exp1_test_odin_auroc': exp1_results['odin_auroc'],
            'exp1_test_odin_aupr': exp1_results['odin_aupr'],
            'places365_test_msp_auroc': places365_results.get('msp_auroc', 0.0),
            'places365_test_msp_aupr': places365_results.get('msp_aupr', 0.0),
            'places365_test_odin_auroc': places365_results.get('odin_auroc', 0.0),
            'places365_test_odin_aupr': places365_results.get('odin_aupr', 0.0)
        }
        results.append(result_row)
        
        # Save confusion matrices and curves for the with_odin model
        if use_odin:
            output_dir = root / "backend" / "models" / "parallel" / "exp1_odin_comparison" / "with_odin" / "20250912_135744"
            
            # Save lesion confusion matrix for Exp1 test
            lesion_class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC']
            save_confusion_matrix(
                exp1_lesion_preds, exp1_lesion_labels, lesion_class_names,
                output_dir / "confusion_matrix_lesion_exp1_test.csv"
            )
            
            # Save ODIN curves for Exp1 test
            if exp1_results['odin_auroc'] > 0:
                in_dist_labels = (exp1_skin_labels == 1).float()
                save_roc_pr_curves(
                    torch.cat([torch.tensor([exp1_results['odin_auroc']])] * len(in_dist_labels)),  # Placeholder
                    in_dist_labels, output_dir
                )
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder/select columns per spec and write standard summary CSV
    cols = [
        'run_name','experiment','architecture','lesion_loss','odin_enabled',
        'exp1_test_skin_acc','exp1_test_skin_f1',
        'exp1_test_lesion_acc','exp1_test_lesion_f1',
        'exp1_test_bm_acc','exp1_test_bm_f1',
        'exp1_test_msp_auroc','exp1_test_msp_aupr',
        'exp1_test_odin_auroc','exp1_test_odin_aupr',
        'places365_test_msp_auroc','places365_test_msp_aupr',
        'places365_test_odin_auroc','places365_test_odin_aupr'
    ]
    results_df = results_df[cols]
    summary_path = root / "backend" / "experiments_summary.csv"
    combined_df = results_df
    
    # Save updated summary
    combined_df.to_csv(summary_path, index=False)
    print(f"\nSaved results to {summary_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Log to WandB
    print("\nLogging to WandB...")
    wandb.init(project="odin_comparison_exp1", name="evaluation_summary")
    
    for _, row in results_df.iterrows():
        # Console per spec
        print(
            f"Exp1: skin acc/F1={row['exp1_test_skin_acc']:.4f}/{row['exp1_test_skin_f1']:.4f}, "
            f"lesion acc/F1={row['exp1_test_lesion_acc']:.4f}/{row['exp1_test_lesion_f1']:.4f}, "
            f"bm acc/F1={row['exp1_test_bm_acc']:.4f}/{row['exp1_test_bm_f1']:.4f}, "
            f"MSP AUROC={row['exp1_test_msp_auroc']:.4f}, AUPR={row['exp1_test_msp_aupr']:.4f}, "
            f"ODIN AUROC={row['exp1_test_odin_auroc']:.4f}, AUPR={row['exp1_test_odin_aupr']:.4f}"
        )
        print(
            f"Places365 MSP: AUROC={row['places365_test_msp_auroc']:.4f}, AUPR={row['places365_test_msp_aupr']:.4f} | "
            f"ODIN: AUROC={row['places365_test_odin_auroc']:.4f}, AUPR={row['places365_test_odin_aupr']:.4f}"
        )
        # W&B (Exp1 classification + ODIN; Places365 only ODIN)
        wandb.log({
            "exp1_test_skin_acc": row['exp1_test_skin_acc'],
            "exp1_test_skin_f1": row['exp1_test_skin_f1'],
            "exp1_test_lesion_acc": row['exp1_test_lesion_acc'],
            "exp1_test_lesion_f1": row['exp1_test_lesion_f1'],
            "exp1_test_bm_acc": row['exp1_test_bm_acc'],
            "exp1_test_bm_f1": row['exp1_test_bm_f1'],
            "exp1_test_msp_auroc": row['exp1_test_msp_auroc'],
            "exp1_test_msp_aupr": row['exp1_test_msp_aupr'],
            "exp1_test_odin_auroc": row['exp1_test_odin_auroc'],
            "exp1_test_odin_aupr": row['exp1_test_odin_aupr'],
            "places365_msp_auroc": row['places365_test_msp_auroc'],
            "places365_msp_aupr": row['places365_test_msp_aupr'],
            "places365_odin_auroc": row['places365_test_odin_auroc'],
            "places365_odin_aupr": row['places365_test_odin_aupr']
        })
    
    wandb.finish()
    print("Evaluation completed!")

if __name__ == "__main__":
    main()