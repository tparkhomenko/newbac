import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import sys
import random
from PIL import Image
import torch.nn.functional as F
import logging
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.lesion_type_head import LesionTypeHead
from datasets.skin_dataset import SkinLesionDataset
from sam.sam_encoder import SAMFeatureExtractor
from utils.augmentation import create_saliency_map

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
        dropout=0.5  # Higher dropout for improved model
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, original_to_new, new_to_original, class_names

def predict_sample(model, feature_extractor, image_path, device, original_to_new=None):
    """Predict on a single image with saliency map."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Extract SAM features
        with torch.no_grad():
            features = feature_extractor.extract_features([image])[0]
            features = features.to(device)
            
            # Forward pass
            outputs = model(features.unsqueeze(0))
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item() * 100
        
        # Generate saliency map
        saliency = create_saliency_map(model, features, predicted_class, device)
        
        return {
            'image': image,
            'features': features,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'saliency': saliency
        }
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None

def visualize_predictions(model, dataset, feature_extractor, class_names, original_to_new, device, num_samples=8, save_dir=None):
    """Visualize predictions on random samples."""
    # Create results directory if needed
    if save_dir is None:
        save_dir = os.path.join(project_root, 'results', 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get random indices stratified by class to ensure all classes are represented
    indices_by_class = {}
    for idx, sample in enumerate(dataset.metadata.itertuples()):
        class_name = sample.lesion_group
        if class_name not in indices_by_class:
            indices_by_class[class_name] = []
        indices_by_class[class_name].append(idx)
    
    # Select random indices for each class
    selected_indices = []
    for class_name, indices in indices_by_class.items():
        num_per_class = max(1, num_samples // len(indices_by_class))
        if indices:
            selected_indices.extend(random.sample(indices, min(num_per_class, len(indices))))
    
    # If we don't have enough samples, add more
    if len(selected_indices) < num_samples:
        remaining = num_samples - len(selected_indices)
        all_indices = list(range(len(dataset)))
        extra_indices = random.sample([i for i in all_indices if i not in selected_indices], remaining)
        selected_indices.extend(extra_indices)
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))  # Increased size for more samples
    axes = axes.flatten()
    
    # Collect features for TSNE visualization
    all_features = []
    all_labels = []
    all_predictions = []
    
    for i, idx in enumerate(selected_indices):
        if i >= len(axes):
            break
            
        sample = dataset.metadata.iloc[idx]
        image_path = os.path.join(project_root, 'datasets', sample['image'])
        
        # Get original label
        original_label_idx = dataset._get_label_index(sample['lesion_group'])
        
        # Remap label if using class mapping
        if original_to_new is not None:
            if original_label_idx in original_to_new:
                remapped_label = original_to_new[original_label_idx]
            else:
                logger.warning(f"Label {original_label_idx} not in mapping, skipping sample")
                continue
        else:
            remapped_label = original_label_idx
        
        # Make prediction
        prediction = predict_sample(model, feature_extractor, image_path, device, original_to_new)
        
        if prediction is None:
            continue
            
        # Store features and labels for t-SNE
        all_features.append(prediction['features'].cpu().numpy())
        all_labels.append(remapped_label)
        all_predictions.append(prediction['predicted_class'])
        
        # Get class name for both true and predicted labels
        true_class_name = sample['lesion_group']
        pred_class_name = class_names[prediction['predicted_class']]
        
        # Plot image with true and predicted labels
        axes[i].imshow(prediction['image'])
        
        # Add correct/incorrect indication with color-coding
        if true_class_name == pred_class_name:
            title_color = 'green'
            result_marker = '✓'
        else:
            title_color = 'red'
            result_marker = '✗'
        
        # Set title with color to indicate correct/incorrect
        axes[i].set_title(f"{result_marker} True: {true_class_name}\nPred: {pred_class_name} ({prediction['confidence']:.1f}%)", 
                         color=title_color, fontsize=10)
        
        # Add probability bars as subplot
        sub_ax = fig.add_axes([axes[i].get_position().x0 + 0.01, 
                             axes[i].get_position().y0 + 0.01, 
                             axes[i].get_position().width * 0.98,
                             0.05])
        
        # Plot probabilities for all classes as horizontal bars
        colors = ['C0' if c != pred_class_name else 'C3' for c in class_names]
        sub_ax.barh(range(len(class_names)), prediction['probabilities'], color=colors)
        sub_ax.set_yticks(range(len(class_names)))
        sub_ax.set_yticklabels([])  # Hide labels as they're in the title
        sub_ax.set_xlim(0, 1)
        sub_ax.set_xticks([])
        
        # Add a small saliency visualization to see what features are important
        sal_ax = fig.add_axes([axes[i].get_position().x0 + 0.75, 
                             axes[i].get_position().y0 + 0.75, 
                             axes[i].get_position().width * 0.2,
                             axes[i].get_position().height * 0.2])
        
        # Create heatmap for saliency
        sal_ax.bar(range(min(20, len(prediction['saliency']))), 
                  prediction['saliency'][:20], 
                  color='red', alpha=0.7)
        sal_ax.set_title('Top 20 Features', fontsize=6)
        sal_ax.set_xticks([])
        sal_ax.set_yticks([])
        
        # Remove axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions_visualization.png'), dpi=300)
    
    # Create t-SNE visualization of feature space
    if all_features:
        plt.figure(figsize=(12, 10))
        
        # Convert features to numpy array
        features_np = np.vstack(all_features)
        labels_np = np.array(all_labels)
        preds_np = np.array(all_predictions)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features_np)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot by true class
        for i, class_name in enumerate(class_names):
            mask = labels_np == i
            if np.any(mask):
                ax1.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=class_name, alpha=0.7)
        
        ax1.set_title('Feature Space by True Class')
        ax1.legend()
        
        # Plot by predicted class
        for i, class_name in enumerate(class_names):
            mask = preds_np == i
            if np.any(mask):
                ax2.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=class_name, alpha=0.7)
        
        ax2.set_title('Feature Space by Predicted Class')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tsne_visualization.png'), dpi=300)
    
    # Create confusion matrix
    if all_labels and all_predictions:
        plt.figure(figsize=(10, 8))
        
        # Compute confusion matrix
        cm = np.zeros((len(class_names), len(class_names)))
        for t, p in zip(all_labels, all_predictions):
            cm[t][p] += 1
        
        # Normalize
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        
        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                  xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Normalized Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_confusion_matrix.png'), dpi=300)
    
    logger.info(f"Visualizations saved to {save_dir}")
    return os.path.join(save_dir, 'predictions_visualization.png')
    
def main():
    # Load config
    config = load_config()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create save directory
    save_dir = os.path.join(project_root, 'results', 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load validation dataset
    val_dataset = SkinLesionDataset(
        metadata_path=config['training'].get('val_metadata_path', config['training']['metadata_path']),
        feature_cache_dir=None,  # Don't use feature cache as we need images
        skin_only=True,
        original_only=True,
        subset_fraction=None
    )
    
    # Load model and class mappings
    checkpoint_path = os.path.join(
        config['training']['model_save_dir'], 
        'lesion_type_balanced_best.pth'  # Use new balanced model
    )
    
    # If the balanced model doesn't exist, try the original model
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(
            config['training']['model_save_dir'], 
            'lesion_type_best.pth'
        )
        logger.warning(f"Balanced model not found, using original model: {checkpoint_path}")
    
    model, original_to_new, new_to_original, class_names = load_model(checkpoint_path, device)
    
    # Initialize feature extractor
    feature_extractor = SAMFeatureExtractor(device=device)
    
    # Visualize predictions
    result_path = visualize_predictions(
        model, 
        val_dataset, 
        feature_extractor, 
        class_names, 
        original_to_new, 
        device, 
        num_samples=16,  # Increased sample count
        save_dir=save_dir
    )
    
    logger.info(f"Visualization completed! Saved to {result_path}")

if __name__ == "__main__":
    main() 