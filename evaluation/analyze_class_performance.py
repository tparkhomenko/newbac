import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.lesion_type_head import LesionTypeHead
from datasets.skin_dataset import SkinLesionDataset
from sam.sam_encoder import SAMFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract class mapping information
    class_mapping = checkpoint.get('class_mapping', None)
    if class_mapping is None:
        logger.warning("No class mapping found in checkpoint, using original model output")
        original_to_new = None
        new_to_original = None
        class_names = ['melanocytic', 'non-melanocytic carcinoma', 'keratosis', 'fibrous', 'vascular']
    else:
        original_to_new = class_mapping['original_to_new']
        new_to_original = class_mapping['new_to_original']
        class_names = class_mapping['class_names']
        logger.info(f"Found class mapping: {original_to_new}")
        logger.info(f"Active classes: {class_names}")
    
    # Determine model output dim from class mapping
    output_dim = len(class_names)
    
    # Create model
    model = LesionTypeHead(
        input_dim=256,
        hidden_dims=[512, 256],
        output_dim=output_dim,
        dropout=0.3
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, original_to_new, new_to_original, class_names

def evaluate_model(model, dataset, feature_extractor, class_names, original_to_new, device):
    """Evaluate model on dataset and analyze per-class performance."""
    logger.info("Starting model evaluation...")
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            features = features.to(device)
            
            # Remap labels if using class mapping
            if original_to_new is not None:
                remapped_labels = torch.tensor([original_to_new.get(t.item(), -1) for t in labels], 
                                              device=device)
                # Skip samples with unknown label mapping
                valid_mask = remapped_labels >= 0
                if not torch.all(valid_mask):
                    logger.warning(f"Skipping {torch.sum(~valid_mask).item()} samples with unknown label mapping")
                    features = features[valid_mask]
                    remapped_labels = remapped_labels[valid_mask]
            else:
                remapped_labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(remapped_labels.cpu().numpy())
            
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Generate classification report
    report = classification_report(all_labels, all_predictions, 
                                  target_names=class_names, 
                                  output_dict=True)
    
    # Calculate per-class accuracy
    class_acc = {}
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        if np.sum(mask) > 0:
            class_acc[class_name] = np.mean(all_predictions[mask] == i) * 100
        else:
            class_acc[class_name] = 0
    
    # Create results dictionary
    results = {
        'confusion_matrix': cm,
        'classification_report': report,
        'per_class_accuracy': class_acc,
        'class_names': class_names,
    }
    
    return results

def plot_confusion_matrix(confusion_matrix, class_names, normalize=True):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    if normalize:
        # Normalize by row (true labels)
        row_sums = confusion_matrix.sum(axis=1)
        confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    sns.heatmap(confusion_matrix, annot=True, fmt=fmt, cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(project_root, 'results', 'confusion_matrices', 
                           f"{'normalized_' if normalize else ''}confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_per_class_metrics(results):
    """Plot per-class metrics."""
    # Extract metrics for each class
    metrics = {}
    for class_name in results['class_names']:
        metrics[class_name] = {
            'Precision': results['classification_report'][class_name]['precision'] * 100,
            'Recall': results['classification_report'][class_name]['recall'] * 100,
            'F1-Score': results['classification_report'][class_name]['f1-score'] * 100,
            'Accuracy': results['per_class_accuracy'][class_name]
        }
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics).T
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
    
    # Plot bar chart for each metric
    bar_width = 0.2
    x = np.arange(len(results['class_names']))
    
    for i, metric in enumerate(['Precision', 'Recall', 'F1-Score', 'Accuracy']):
        plt.bar(x + i*bar_width, df[metric], width=bar_width, label=metric)
    
    plt.xlabel('Class')
    plt.ylabel('Score (%)')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x + bar_width*1.5, df.index, rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.ylim(0, 100)
    
    # Add data labels on bars
    for i, metric in enumerate(['Precision', 'Recall', 'F1-Score', 'Accuracy']):
        for j, val in enumerate(df[metric]):
            plt.text(j + i*bar_width, val + 1, f'{val:.1f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(project_root, 'results', 'confusion_matrices', 'per_class_metrics.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def analyze_class_distribution(dataset, class_names, original_to_new):
    """Analyze class distribution in the dataset."""
    # Count samples per class
    class_counts = {}
    for class_name in class_names:
        class_counts[class_name] = 0
    
    for i in range(len(dataset)):
        sample = dataset.metadata.iloc[i]
        class_idx = dataset._get_label_index(sample['lesion_group'])
        
        # Remap if needed
        if original_to_new is not None:
            if class_idx in original_to_new:
                remapped_idx = original_to_new[class_idx]
                if 0 <= remapped_idx < len(class_names):
                    class_counts[class_names[remapped_idx]] += 1
        else:
            if 0 <= class_idx < len(class_names):
                class_counts[class_names[class_idx]] += 1
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_counts)))
    
    plt.bar(class_counts.keys(), class_counts.values(), color=colors)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Dataset')
    plt.xticks(rotation=45, ha='right')
    
    # Add counts on top of bars
    for i, (class_name, count) in enumerate(class_counts.items()):
        plt.text(i, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(project_root, 'results', 'confusion_matrices', 'class_distribution.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return class_counts

def main():
    # Load config
    config = load_config()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Make sure directories exist
    os.makedirs(os.path.join(project_root, 'results', 'confusion_matrices'), exist_ok=True)
    
    # Load validation dataset
    val_dataset = SkinLesionDataset(
        metadata_path=config['training'].get('val_metadata_path', config['training']['metadata_path']),
        feature_cache_dir=config['training']['val_features_dir'],
        skin_only=True,
        original_only=True,
        subset_fraction=None
    )
    
    # Load model and class mappings
    checkpoint_path = os.path.join(
        config['training']['model_save_dir'], 
        'lesion_type_best.pth'
    )
    model, original_to_new, new_to_original, class_names = load_model(checkpoint_path, device)
    
    # Initialize feature extractor
    feature_extractor = SAMFeatureExtractor(device=device)
    
    # Analyze class distribution
    logger.info("Analyzing class distribution...")
    class_counts = analyze_class_distribution(val_dataset, class_names, original_to_new)
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(model, val_dataset, feature_extractor, class_names, original_to_new, device)
    
    # Plot confusion matrices
    logger.info("Plotting confusion matrices...")
    plot_confusion_matrix(results['confusion_matrix'], class_names, normalize=False)
    plot_confusion_matrix(results['confusion_matrix'], class_names, normalize=True)
    
    # Plot per-class metrics
    logger.info("Plotting per-class metrics...")
    plot_per_class_metrics(results)
    
    # Print classification report
    logger.info("Classification Report:")
    report_df = pd.DataFrame(results['classification_report']).transpose()
    logger.info("\n" + report_df.to_string())
    
    # Print per-class accuracy
    logger.info("Per-Class Accuracy:")
    for class_name, acc in results['per_class_accuracy'].items():
        logger.info(f"{class_name}: {acc:.2f}%")
    
    logger.info(f"Analysis completed! Results saved to {os.path.join(project_root, 'results', 'confusion_matrices')}")

if __name__ == "__main__":
    main() 