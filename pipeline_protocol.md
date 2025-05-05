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

4. **Advanced Augmentation Techniques**:
   - CutMix: Combines random crops from two images
   - MixUp: Linear interpolation of feature vectors between samples
   - Both help regularize the model and improve generalization

5. **Class Imbalance Handling**:
   - Early attempts with full ISIC dataset (original distribution) failed due to extreme imbalance
   - Example: 21,000+ melanocytic samples vs. ~300 fibrous samples
   - Standard CrossEntropyLoss with class weights was insufficient
   - Model collapsed to predicting only majority classes (~20% train accuracy, ~1% validation)
   - Solution: Balanced dataset + FocalLoss + proper validation metrics

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
- [x] **Train Lesion Type Classifier (MLP2)**
  - Architecture:
    * Input: 256-dim SAM features (same as MLP1)
    * Hidden layers: [512, 256] with ReLU activation and Dropout(0.3)
    * Output: 5 neurons (one per lesion class) with Softmax activation
  - Initial Training Challenges:
    * ❌ First attempt used full ISIC dataset with original class distribution
    * ❌ Extreme imbalance: 21,000+ melanocytic samples vs. only ~300 fibrous samples
    * ❌ Standard CrossEntropyLoss with class weights failed to handle imbalance
    * ❌ Model collapsed to predicting only majority classes (~20% train accuracy, ~1% validation)
    * ❌ Zero F1 scores for most classes, confusion matrix showed single-class predictions
  - Successful Training Strategy:
    * ✅ Balanced training set created with max 2,000 samples per majority class
    * ✅ Used all available samples for minority classes (fibrous: 1,220, vascular: 1,205)
    * ✅ Switched to FocalLoss with moderate class weights [1.5, 1.5, 1.5, 1.2, 1.2]
    * ✅ Created balanced validation set for accurate performance metrics
    * ✅ Implemented mixed precision training for efficiency
    * ✅ Added proper class index mapping to handle all classes correctly
  - Training Results:
    * Best validation accuracy: 65.8% (Epoch 43)
    * Per-class F1 scores ranging from 0.59 to 0.72
    * Balanced performance across all classes
    * Confusion matrix showed distributed predictions with reasonable clinical confusions
    * Model checkpoint: `saved_models/lesion_type/lesion_type_max2000_best.pth`
    * Training logs: `wandb/run-20250505_165108-jvnir41k/`
  - Dataset Statistics:
    * See detailed table in implementation details
  - ✅ Production Status:
    * Model considered production-ready baseline
    * Achieves balanced prediction across all classes with good overall accuracy
    * Errors align with clinically expected confusion patterns
- [ ] **Train Benign/Malignant Classifier (MLP3)**
  - Implementation completed with `training/train_benign_malignant.py`
  - Created dedicated dataset: `datasets/benign_malignant_dataset.py`
  - Proposed configurations for experiments:
    * **Configuration A - Original Images Only**:
      - Uses only non-augmented original images (~31.5k total)
      - Train split: ~22k images (13.6k benign / 8.5k malignant)
      - Class ratio: ~60% benign / 40% malignant
      - Higher image quality, no augmentation artifacts
      - Command: `python -m training.train_benign_malignant --config original`
      - Model path: `saved_models/benign_malignant/benign_malignant_original_best.pth`
    
    * **Configuration B - Full Augmented Set**:
      - Uses all augmented images (~157.6k total)
      - Train split: ~110k images (68k benign / 42.5k malignant)
      - Class ratio: ~60% benign / 40% malignant
      - More data but includes augmented variants
      - Command: `python -m training.train_benign_malignant --config augmented`
      - Model path: `saved_models/benign_malignant/benign_malignant_augmented_best.pth`
    
    * **Configuration C - Balanced Subset (2k per class)**:
      - Fixed size dataset like MLP2 (4k total)
      - Perfect 50/50 class balance (2k benign / 2k malignant)
      - Balanced training for better performance on minority class
      - Command: `python -m training.train_benign_malignant --config balanced`
      - Model path: `saved_models/benign_malignant/benign_malignant_balanced_best.pth`
  
  - Common training parameters:
    * Architecture: `[Input: 256] → [512, ReLU, Dropout(0.3)] → [256, ReLU, Dropout(0.3)] → [2, Softmax]`
    * FocalLoss with gamma=2.0 and class weights
    * Mixed precision training for efficiency
    * Batch size: 16 for training, 32 for validation
    * Early stopping after 5 epochs without improvement
    * Learning rate: 0.001 with ReduceLROnPlateau scheduler
    * Optimizer: AdamW with weight_decay=0.01
    * Wandb logging for all metrics and visualizations
- [ ] **Evaluate all classifiers** (accuracy, F1, AUROC)
- [ ] **Generate Segmentation Masks (for Visualization Only)**
- [ ] **Generate Prompt Points**: Find and save extremum grayscale points
- [ ] **Plot Heuristic Statistics** for SAM mask filtering
- [ ] **Write Final Results and Analysis**
- [x] **Balance the Validation Set**:
  - Created `datasets/metadata/val_balanced_500.csv` with up to 500 samples per lesion group (melanocytic, non-melanocytic carcinoma, keratosis, fibrous, vascular)
  - Used for validation during training for fairer per-class evaluation
- [x] **Verify Class-to-Index Mapping**:
  - Documented and printed class-to-index mapping in training logs and evaluation output
- [x] **Log Per-Class Metrics During Training**:
  - Added F1, precision, recall per class to each epoch's log output and wandb logs
- [x] **Replace or Augment Loss Function**:
  - FocalLoss implemented and used with class weights; fallback to CrossEntropyLoss possible
- [x] **Add Class Weights for Loss**:
  - Class frequency-based weights computed and passed to loss function
- [x] **Add WeightedRandomSampler**:
  - Option to use WeightedRandomSampler for class-balanced minibatches in training
- [x] **Early Stopping or Collapse Detection**:
  - Early stopping implemented: stops if validation loss does not improve after N epochs
- [x] **Reduce Training Epochs During Debugging**:
  - Configurable debug_epochs option for quick testing (default 10)
- [x] **Visualize Confusion Matrix Per Epoch**:
  - Confusion matrix printed and logged for each validation epoch
- [x] **Fix Weights & Biases Logging**:
  - ✅ Added model parameter and gradient tracking with wandb.watch()
  - ✅ Implemented comprehensive metric logging:
    * Batch-level: loss, accuracy, learning rate
    * Epoch-level training: loss, accuracy, per-class metrics
    * Epoch-level validation: loss, accuracy, per-class metrics, confusion matrix
    * Per-class metrics include: accuracy, precision, recall, F1 score, sample counts
  - ✅ Fixed metric synchronization to ensure proper dashboard visualization
  - ✅ Added confusion matrix visualization for validation epochs
  - ✅ Fixed class mapping issue to ensure proper metrics tracking when some classes have zero samples
- [x] **Implement Class-Specific Performance Analysis**:
  - ✅ Created `evaluation/analyze_class_performance.py` to identify problematic classes
  - ✅ Analysis includes:
    * Class distribution visualization
    * Per-class metrics (precision, recall, F1-score, accuracy)
    * Raw and normalized confusion matrices
    * Detailed classification reports
  - ✅ Results saved to `results/confusion_matrices/` directory
- [x] **Implement Visual Prediction Analysis**:
  - ✅ Created `evaluation/visualize_predictions.py` for visual sanity checks
  - ✅ Features:
    * Randomly samples validation images
    * Shows true vs. predicted labels
    * Color-codes correct/incorrect predictions
    * Displays prediction confidence
    * Shows full probability distribution across classes
  - ✅ Helps identify patterns in misclassifications
    * Identifies classes commonly confused with each other
    * Reveals potential biases in the model's predictions
  - ✅ Results saved to `results/predictions_visualization.png`
  - 🆕 Enhanced with t-SNE visualizations of feature space
  - 🆕 Added saliency maps to visualize feature importance
- [x] **Fix Class Index Mapping Issue**:
  - ✅ Implemented class index remapping function:
    * Maps original indices to consecutive new indices (0-N)
    * Creates a mapping dictionary: `{original_idx: new_idx}`
    * Updates model output layer to match only active classes
    * Applies target remapping in training/validation loops
    * Saves mapping information with checkpoints for inference
  - ✅ Fixed class weight filtering to match new indices
  - ✅ Added proper documentation in `common_errors.md`
  - ✅ Verified correct operation with test cases

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

- Latest Balanced Dataset Configuration:
  * Training set (8,425 total):
    - melanocytic: 2,000
    - non-melanocytic carcinoma: 2,000
    - keratosis: 2,000
    - fibrous: 1,220 (all available in train split)
    - vascular: 1,205 (all available in train split)
  * Validation set (1,425 total):
    - melanocytic: 300
    - non-melanocytic carcinoma: 300
    - keratosis: 300
    - fibrous: 245
    - vascular: 280

- Class Weights (FocalLoss):
  * melanocytic: 1.5
  * keratosis: 1.5
  * non-melanocytic carcinoma: 1.5
  * vascular: 1.2
  * fibrous: 1.2

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
│   ├── train_lesion_type.py  
│   ├── train_lesion_type_balanced.py  
│   └── train_benign_malignant.py  
├── evaluation/  
│   ├── evaluate_heads.py  
│   ├── visualize_predictions.py  
│   ├── visualize_masks.py  
│   ├── analyze_class_performance.py  
│   └── plot_metrics.py  
├── configs/  
│   └── config.yaml  
├── utils/  
│   ├── metrics.py  
│   ├── feature_extraction.py  
│   └── augmentation.py  
├── saved_models/  
├── results/  
│   ├── confusion_matrices/  
│   ├── auc_curves/  
│   ├── mask_visualizations/  
│   ├── tsne_umap_embeddings/  
│   └── experiment_logs.md  
├── run_pipeline.py  
├── requirements.txt  
├── .gitignore  
├── README.md  
├── common_errors.md  
└── pipeline_protocol.md  

## ⚙️ Development Rules

- Update `config.yaml` with all new paths, hyperparameters, and flags.
- Save all intermediate data under `/data/processed/`.
- Keep SAM-specific code inside `/sam/` only.
- Store evaluation results in `/results/`.
- Commit all code changes with clear messages (e.g., `"feat: train MLP1"`, `"fix: mask filtering"`).
- Keep `README.md` and `pipeline_protocol.md` aligned with current code and structure.
- Exclude unnecessary files in `.gitignore` (e.g., datasets, checkpoints, logs, `.ipynb_checkpoints/`)
- 📊 Document all major changes in the training pipeline in `results/experiment_logs.md`:
  - Create a new experiment entry for each significant architecture or methodology change
  - Include detailed information about model architecture, training configuration, results, and analysis
  - Add references to full training logs in the `/logs/` directory
  - Format the experiment ID with date for easy tracking (e.g., "Experiment #2: ResNet Features - May 10, 2025")

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
- Class index mapping is crucial for correct model training:
  - When using a subset of classes, create a proper mapping from original to consecutive indices
  - Update model output layer to match only active classes
  - Remap targets in training/validation loops
  - Save mapping information with checkpoints for inference
- Class imbalance handling is critical for successful model training:
  - Original class distribution (21,000+ melanocytic vs. 300 fibrous) led to model collapse
  - Standard CrossEntropyLoss failed even with class weights
  - Success achieved with balanced dataset (max 2,000 per class) + FocalLoss

## 📍 Model Locations

### Production Models

| Model | Description | Path | Performance |
|-------|-------------|------|-------------|
| SAM Encoder | Feature extractor (frozen) | `saved_models/sam/sam_vit_h.pth` | N/A (pre-trained) |
| MLP1 | Skin/Not Skin classifier | `saved_models/skin_not_skin/skin_not_skin_10k_best.pth` | 99.38% validation accuracy |
| MLP2 | Lesion Type classifier | `saved_models/lesion_type/lesion_type_max2000_best.pth` | 65.8% validation accuracy |
| MLP3 | Benign/Malignant classifier | Not yet implemented | - |

### Model Loading Code Examples

```python
# Load MLP1 (Skin/Not Skin Classifier)
import torch
from models.skin_not_skin_head import SkinNotSkinHead

# Initialize model with same architecture as during training
model_mlp1 = SkinNotSkinHead(input_dim=256, hidden_dims=[512, 256], output_dim=2)
checkpoint = torch.load('saved_models/skin_not_skin/skin_not_skin_10k_best.pth')
model_mlp1.load_state_dict(checkpoint['model_state_dict'])
model_mlp1.eval()  # Set to evaluation mode

# Load MLP2 (Lesion Type Classifier)
from models.lesion_type_head import LesionTypeHead

# Initialize model with same architecture as during training
model_mlp2 = LesionTypeHead(input_dim=256, hidden_dims=[512, 256], output_dim=5)
checkpoint = torch.load('saved_models/lesion_type/lesion_type_max2000_best.pth')
model_mlp2.load_state_dict(checkpoint['model_state_dict'])
# Get class mapping information (important for correct class indices)
class_mapping = checkpoint.get('class_mapping', None)
model_mlp2.eval()  # Set to evaluation mode

# Load MLP3 (Benign/Malignant Classifier)
from models.benign_malignant_head import BenignMalignantHead

# Initialize model with same architecture as during training
model_mlp3 = BenignMalignantHead(input_dim=256, hidden_dims=[512, 256], output_dim=2)
checkpoint = torch.load('saved_models/benign_malignant/benign_malignant_best.pth')
model_mlp3.load_state_dict(checkpoint['model_state_dict'])
# Get class mapping information
class_mapping = checkpoint.get('class_mapping', None)  # Should be {0: 'benign', 1: 'malignant'}
model_mlp3.eval()  # Set to evaluation mode
```

### Training Logs

| Model | Log Directory | Run ID |
|-------|---------------|--------|
| MLP1 | `wandb/run-20250430_164542-6scfn1g6/` | `6scfn1g6` |
| MLP2 | `wandb/run-20250505_165108-jvnir41k/` | `jvnir41k` |

✅ End of Protocol
