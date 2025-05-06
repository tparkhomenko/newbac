# 🔬 Skin Lesion Classification Pipeline

A deep learning pipeline for skin lesion classification using SAM2 encoder features.

## 📋 Project Overview

This project implements a multi-head classification pipeline for skin lesion analysis:

1. **Skin/Not-Skin Classification (MLP1)**: Binary classification to identify skin images
2. **Lesion Type Classification (MLP2)**: Multi-class classification for 5 lesion groups
3. **Benign/Malignant Classification (MLP3)**: Binary classification of lesion malignancy

All classifiers use features extracted from a frozen SAM2 encoder (ViT architecture).

## 🧠 Architecture

- **Feature Extractor**: SAM2 ViT-H (frozen)
- **Classifiers**:
  - MLP1: Skin vs. Not-Skin (2 classes)
  - MLP2: Lesion Type (5 classes: melanocytic, non-melanocytic carcinoma, keratosis, fibrous, vascular)
  - MLP3: Benign vs. Malignant (2 classes)

## 📊 Datasets

- **ISIC 2019**: 33,569 dermatology images with lesion classifications
- **DTD (Describable Textures Dataset)**: 5,640 texture images used as non-skin samples
- **Augmented Total**: Over 196,000 images after augmentation

## 🔧 Technical Implementation Notes

### Class Index Mapping Fix

We implemented a robust class index mapping solution for handling class imbalance and absent classes:

1. **Problem**: When training with only a subset of class samples, class indices in the model output (0 to N-1) 
   might not match the original dataset class indices, leading to incorrect loss calculation and training failure.

2. **Solution**:
   - Added `create_class_index_mapping()` function to create bidirectional mappings between original and new consecutive indices
   - Modified model output layer to match only active classes
   - Applied proper class weight filtering to match new indices
   - Implemented target index remapping in training/validation loops
   - Saved class mapping information with checkpoints for inference

This ensures consistent class index handling throughout the pipeline, even when some classes have zero samples.

## 🚀 Getting Started

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

### Training

```bash
# Train the Skin/Not-Skin classifier
python training/train_skin_not_skin.py

# Train the Lesion Type classifier
python training/train_lesion_type.py

# Train the Benign/Malignant classifier
python training/train_benign_malignant.py
```

### Evaluation

```bash
# Evaluate all classifiers
python evaluation/evaluate_heads.py
```

## 📄 Documentation

For comprehensive documentation, see:
- [Pipeline Protocol](pipeline_protocol.md) - Detailed implementation plan and progress tracking
- [Common Errors](common_errors.md) - Solutions to common issues

## 📊 Results Visualization

Classification results and performance metrics are stored in:
- `/results/confusion_matrices/`
- `/results/auc_curves/`
- `/results/tsne_umap_embeddings/`

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
