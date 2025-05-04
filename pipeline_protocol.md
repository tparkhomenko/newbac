# 🧠 Skin Lesion Classification Pipeline Protocol

Rules: **Keep `.gitignore`, `requirements.txt`, `pipeline_protocol.md`, and `common_errors.md` updated**

For common errors and their solutions, see [Common Errors](common_errors.md)

## 📝 Notes and Future Explorations

1. **Alternative Training Approaches**:
   - Compare balanced sampling (current: 2k images/class) vs class weight balancing
   - Advantages of sampling: consistent batch sizes, predictable memory usage
   - Advantages of weighting: uses all available data, might capture more variance
   
2. **Architecture Alternatives**:
   - Current: Multi-head approach (MLP1 → MLP2/MLP3)
   - Explore: Single flat classifier comparing all classes at once
   - Potential benefits: simpler architecture, direct optimization
   - Potential drawbacks: might lose hierarchical information (skin/not-skin → lesion type)

3. **Memory Management Optimizations**:
   - Using mixed precision training (torch.cuda.amp)
   - Reduced batch size (16) for better memory efficiency
   - Periodic GPU memory cleanup during training
   - Feature extraction with proper cleanup
   - Progress monitoring with tqdm
   - Error handling with fallback to zero features

## 🚀 Project TODOs and Progress Tracker

- [x] **Prepare Folder Structure**: Create all necessary folders as per protocol
- [x] **Prepare Datasets**: Organize ISIC 2019 and DTD under `/data/`
- [x] **Process ISIC Metadata**: Create unified metadata with lesion groups and malignancy labels
  - Combined train/test ground truth
  - Mapped lesions to broader groups:
    * melanocytic (MEL, NV)
    * non-melanocytic carcinoma (BCC, SCC)
    * keratosis (AK, BKL)
    * fibrous (DF)
    * vascular (VASC)
  - Added malignancy labels:
    * malignant (MEL, BCC, SCC, AK)
    * benign (NV, BKL, DF, VASC)
- [x] **Preprocess ISIC 2019 Dataset**:
  - Filtered out unknown lesions
  - Resized all images to 1024x1024 (SAM format)
  - Applied augmentations:
    * Original image
    * Horizontal flip
    * Vertical flip
    * 90° rotation
    * 270° rotation
  - Created augmented metadata matching processed images
  - Final dataset statistics:
    * Original images: 33,569
    * Augmented total: 167,845 (5x original)
    * All images stored in: `datasets/isic2019_aug/`
    * Metadata format: `ISIC_XXXXXXX_[augmentation].jpg`
- [x] **Preprocess DTD Dataset**:
  - Resized all images to 1024x1024 (SAM format)
  - Applied same augmentations as ISIC:
    * Original image
    * Horizontal flip
    * Vertical flip
    * 90° rotation
    * 270° rotation
  - Final dataset statistics:
    * Original images: 5,640
    * Augmented total: 28,200 (5x original)
    * All images stored in: `datasets/dtd_aug/`
    * Metadata format: `[texture]_XXXX_[augmentation].jpg`
- [x] **Create Unified Metadata**:
  - Created unified metadata combining both datasets
  - Saved as: `datasets/metadata/unified_augmented.csv`
  - Standardized columns:
    * image: relative path to augmented image
    * lesion_group: original for ISIC, 'not_skin_texture' for DTD
    * lesion: original for ISIC, 'DTD' for DTD
    * malignancy: original for ISIC, 'not_applicable' for DTD
    * skin: 1 for ISIC, 0 for DTD
  - Final statistics:
    * Total images: 196,045
    * ISIC (skin) images: 167,845
    * DTD (not-skin) images: 28,200
    * Balanced skin/not-skin ratio: ~86%/14%
- [x] **Build SAM Feature Extractor**: Use SAM2 encoder (ViT, frozen)
  - Created `sam/sam_encoder.py` with SAMFeatureExtractor class
  - Implemented automatic model download and caching
  - Added image preprocessing (resize to 1024x1024, normalize)
  - Feature extraction returns [B, 256] dimensional embeddings
  - Model checkpoint path: `saved_models/sam/sam_vit_h.pth`
  - ✅ Verified functionality:
    * Successfully loads on GPU (CUDA)
    * Weights properly frozen
    * Correct preprocessing pipeline
    * Feature extraction tested with batch processing
- [x] **Implement MLP1 Architecture**: Skin/Not Skin Classifier
  - Created `models/skin_not_skin_head.py`
  - Architecture:
    * Input: 256-dim SAM features
    * Hidden layers: [512, 256] with ReLU & Dropout(0.3)
    * Output: 2 classes (skin/not-skin)
  - ✅ Verified functionality:
    * Config loading works
    * Forward pass tested with batch input
    * Correct output shape [B, 2]
- [x] **Implement Dataset Pipeline**:
  - Created `datasets/skin_dataset.py` with PyTorch Dataset class
  - Features:
    * Loads from unified metadata CSV
    * Handles train/val/test splits
    * Extracts SAM features on-the-fly
    * Optional feature caching to disk
    * Proper error handling for missing files
  - Split statistics from metadata:
    * Train: 137,122 images (117,490 skin / 19,632 non-skin)
    * Val: 29,399 images (25,175 skin / 4,224 non-skin)
    * Test: 29,524 images (25,180 skin / 4,344 non-skin)
- [x] **Train Skin/Not Skin Classifier (MLP1)**
  - Training with balanced subset:
    * 2000 samples per class (4000 total)
    * 400 validation samples per class
    * Using SAM features with mixed precision
    * Class weights: [1.0, 5.93]
  - Training Results:
    * Best validation accuracy: 99.38% (Epoch 37)
    * Best validation loss: 0.0144
    * Final training accuracy: 99.08%
    * Final training loss: 0.0124
    * Best model saved: `saved_models/skin_not_skin/skin_not_skin_10k_best.pth`
    * Training converged successfully with no overfitting
- [ ] **Train Lesion Type Classifier (MLP2)**
  - Architecture:
    * Input: 256-dim SAM features (same as MLP1)
    * Output: Multi-class classification for lesion groups
  - Training Strategy:
    * Train only on skin images (filtered by MLP1)
    * Using 30% of available data for faster iteration
    * Memory optimizations:
      - Mixed precision training
      - Batch size: 16
      - Periodic GPU memory cleanup
      - Feature caching with error handling
    * Class distribution in full dataset:
      - Melanocytic: 106,095 samples (use 30%)
      - Non-melanocytic carcinoma: 25,455 samples (use 30%)
      - Keratosis: 22,625 samples (use 30%)
      - Fibrous: 1,650 samples (use all)
      - Vascular: 1,785 samples (use all)
    * Class imbalance: ~64x difference between largest and smallest classes (recalculated after unknown removal)
  - ✅ Dataset Implementation:
    * Added skin-only filtering
    * Added lesion type target support
    * Added dynamic class weight computation
    * Implemented proper feature caching for different configurations
    * Verified correct feature shape [256] and label mapping
    * Confirmed working data loading and preprocessing pipeline
    * Added robust error handling with zero feature fallback
    * Implemented memory-efficient feature extraction
    * 🆕 Now supports training on only original images (no augmentations) via `original_only=True` in `SkinLesionDataset`. This is now used in `train_lesion_type.py`.
    * 🆕 Weighted loss approach implemented: CrossEntropyLoss uses class weights computed from the training set to address class imbalance. Tested with 5% and 10% of all original, skin-only images. Validation accuracy improved to ~55% with 10% data, showing benefit of weighting and more data.
- [ ] **Train Benign/Malignant Classifier (MLP3)**
  - Optional: Apply class weight correction for benign (96,705) vs malignant (60,905)
- [ ] **Evaluate all classifiers** (accuracy, F1, AUROC)
- [ ] **Generate Segmentation Masks (for Visualization Only)**
- [ ] **Generate Prompt Points**: Find and save extremum grayscale points
- [ ] **Plot Heuristic Statistics** for SAM mask filtering
- [ ] **Write Final Results and Analysis**

## 🎯 Project Goal

Train a two-stage classification pipeline using frozen SAM2 encoder features:
1. Classify image as **Skin vs Not Skin** (MLP1)
2. If Skin → classify **Lesion Group** (MLP2) and **Benign vs Malignant** (MLP3)

Segmentation masks are only for visualization, not training, therefore are applied at the end to an original image.

## 📂 Datasets Used

- ✅ ISIC 2019 – Skin lesions
  - Original dataset:
    * Total images: 33,569
    * Lesion groups:
      - melanocytic: 21,219
      - non-melanocytic carcinoma: 5,091
      - keratosis: 4,525
      - vascular: 357
      - fibrous: 330
    * Malignancy:
      - benign: 19,341
      - malignant: 12,181
  - Preprocessed dataset (512x512, augmented):
    * Total images: 157,610
    * Lesion groups:
      - melanocytic: 106,095
      - non-melanocytic carcinoma: 25,455
      - keratosis: 22,625
      - vascular: 1,785
      - fibrous: 1,650
    * Malignancy:
      - benign: 96,705
      - malignant: 60,905
    * All labeled as skin (skin=1)
- ✅ DTD – Describable Textures Dataset (non-skin images)
  - Original dataset:
    * Total images: 5,640
    * 47 texture categories
    * 120 images per category
  - Preprocessed dataset (512x512, augmented):
    * Total images: 28,200
    * 600 images per texture category
    * Augmentations per image:
      - Original
      - Horizontal flip
      - Vertical flip
      - 90° rotation
      - 270° rotation
    * All labeled as non-skin (skin=0)
  - Used for skin/not-skin classification training

## 📊 Dataset Splits

- Total dataset size: 196,045 images
- Split distribution:
  * Train: 137,122 images (70%)
  * Validation: 29,399 images (15%)
  * Test: 29,524 images (15%)

- Skin vs Non-Skin distribution per split:
  * Train: 117,490 skin / 19,632 non-skin
  * Validation: 25,175 skin / 4,224 non-skin
  * Test: 25,180 skin / 4,344 non-skin

- Lesion group distribution:
  * Train set:
    - Melanocytic: 74,530
    - Non-melanocytic carcinoma: 17,670
    - Keratosis: 15,845
    - Fibrous: 1,220
    - Vascular: 1,205
    - Not skin texture: 19,632
  * Validation set:
    - Melanocytic: 15,850
    - Non-melanocytic carcinoma: 3,835
    - Keratosis: 3,405
    - Fibrous: 245
    - Vascular: 280
    - Not skin texture: 4,224
  * Test set:
    - Melanocytic: 15,715
    - Non-melanocytic carcinoma: 3,950
    - Keratosis: 3,375
    - Fibrous: 185
    - Vascular: 300
    - Not skin texture: 4,344

## 🏗️ Encoder/Decoder Structure

| Module     | Description                            |
|------------|----------------------------------------|
| Encoder    | SAM2 Encoder (ViT, frozen)             |
| Decoder    | 3 MLP heads:                           |
|            | - 🧠 MLP1: Skin/Not Skin Classifier    |
|            | - 🧠 MLP2: Lesion Type Classifier      |
|            | - 🧠 MLP3: Benign/Malignant Classifier |

## 🔥 Pipeline Flow

Input Image  
→ SAM2 Encoder (frozen)  
→ MLP1: Skin/Not Skin  
  ├─ If Not Skin → STOP  
  └─ If Skin →  
    → MLP2: Lesion Type (ISIC only)  
    → MLP3: Benign/Malignant (ISIC only)

After classification:  
→ Generate 3 SAM2 masks 
→ Apply heuristics:  
 - Remove masks covering >80% of image  
 - Prefer masks with fewer blobs  
 - Discard overlaps >95%  
→ Keep a many as passed the heuristics, do an overview (graph) on masks left after heuristics

## 🏛️ Project Folder Structure

skin_classification_pipeline/  
├── data/  
│   ├── isic2019/  
│   ├── dtd/  
│   ├── processed/  
│   └── metadata/  
├── sam/  
│   ├── sam_encoder.py  
│   ├── sam_prompting.py  
│   └── sam_mask_utils.py  
├── datasets/  
│   ├── skin_dataset.py  
│   └── mask_dataset.py  
├── models/  
│   ├── skin_not_skin_head.py  
│   ├── lesion_type_head.py  
│   └── benign_malignant_head.py  
├── training/  
│   ├── train_skin_not_skin.py  
│   └── train_lesion_malignancy.py  
├── evaluation/  
│   ├── evaluate_heads.py  
│   ├── visualize_masks.py  
│   └── plot_metrics.py  
├── configs/  
│   └── config.yaml  
├── saved_models/  
├── results/  
│   ├── confusion_matrices/  
│   ├── auc_curves/  
│   ├── mask_visualizations/  
│   └── tsne_umap_embeddings/  
├── utils/  
│   ├── metrics.py  
│   └── feature_extraction.py  
├── run_pipeline.py  
├── requirements.txt  
├── .gitignore  
├── README.md  
└── pipeline_protocol.md  

## ⚙️ Development Rules

- Update `config.yaml` with all new paths, hyperparameters, and flags.
- Save all intermediate data under `/data/processed/`.
- Keep SAM-specific code inside `/sam/` only.
- Store evaluation results in `/results/`.
- Commit all code changes with clear messages (e.g., `"feat: train MLP1"`, `"fix: mask filtering"`).
- Keep `README.md` and `pipeline_protocol.md` aligned with current code and structure.
- Exclude unnecessary files in `.gitignore` (e.g., datasets, checkpoints, logs, `.ipynb_checkpoints/`)

## ⚡ Implementation Notes

- SAM2 encoder is frozen and never fine-tuned.
- Only MLP heads (classifiers) are trained.
- SAM masks are for visualization only, not used in model training.
- Mask heuristics (area, blob count, overlap) are applied post-classification.
- ISIC 2019 dataset provides official train/test split
- Evaluation must separately report:
  - Skin/Not Skin (MLP1)
  - Lesion Type (MLP2, 5 classes)
  - Benign/Malignant (MLP3)

✅ End of Protocol
