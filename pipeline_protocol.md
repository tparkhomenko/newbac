# 🧠 Skin Lesion Classification Pipeline Protocol

Rules: **Keep `.gitignore`, `requirements.txt`, `pipeline_protocol.md`, and `common_errors.md` updated**

For common errors and their solutions, see [Common Errors](common_errors.md)

## 📊 Complete Model Summary Table

| MLP # | Model Path | Log Path | Script Path | Purpose | Datasets Used | Dataset Split | Architecture | Val Accuracy | Confusion Matrix | Training Time | Comments | Other Details |
|-------|------------|----------|-------------|---------|---------------|--------------|--------------|--------------|------------------|---------------|----------|--------------|
| MLP1 | `saved_models/skin_not_skin/skin_not_skin_10k_best.pth` | `wandb/run-20250430_164542-6scfn1g6/` | `training/train_skin_not_skin.py` | Binary classification of images as skin lesion or non-skin texture | ISIC 2019 + DTD (both augmented) | 2000 samples per class (4000 total) with 400 validation per class | Input: 256 → Hidden: [512, 256] with ReLU & Dropout(0.3) → Output: 2 classes | 99.38% | Not specified | Not specified | Production model with excellent performance; successfully distinguishes skin from non-skin with high accuracy | Uses class weights [1.0, 5.93] to account for imbalance; trained with balanced subset for efficiency |
| MLP1 | `saved_models/skin_not_skin/skin_not_skin_subset_best.pth` | Not specified | `training/train_skin_not_skin.py` | Binary classification of images as skin lesion or non-skin texture | ISIC 2019 + DTD (both augmented) | Smaller subset than production model | Same as production model | Not specified | Not specified | Not specified | Initial test model with smaller dataset | Used for quick testing and debugging |
| MLP2 | `saved_models/lesion_type/lesion_type_max2000_best.pth` | `wandb/run-20250505_165108-jvnir41k/` | `training/train_lesion_type_max2k.py` | Multi-class classification of skin lesion type into 5 categories | ISIC 2019 only (augmented) | Max 2000 samples per majority class; 8,425 total training samples | Input: 256 → Hidden: [512, 256] with ReLU & Dropout(0.3) → Output: 5 classes | 65.8% | `results/confusion_matrices/lesion_type_max2000_cm.png` | 3.2 hours on NVIDIA RTX 3090 | Production model for lesion type classification; achieved balanced performance across all classes | Used FocalLoss with class weights [1.5, 1.5, 1.5, 1.2, 1.2]; F1 scores: 0.59-0.72 across classes |
| MLP2 | `saved_models/lesion_type/lesion_type_best.pth` | Various experimental runs | `training/train_lesion_type.py` | Multi-class classification of skin lesion type into 5 categories | ISIC 2019 only (augmented) | Original class distribution; highly imbalanced | Same as production model | ~1% (failed) | Not available | Not specified | Failed experiment with original class distribution; model collapsed to predicting only majority classes | Demonstrated that class balancing is essential for highly imbalanced datasets |
| MLP3 | `saved_models/benign_malignant/benign_malignant_balanced_best.pth` | `wandb/run-20250505_193729-sxlaexj3/` | `training/train_benign_malignant.py` | Binary classification of skin lesions as benign or malignant | ISIC 2019 only (augmented) | 2000 samples per class (4000 total); perfect 50/50 class balance | Input: 256 → Hidden: [512, 256] with ReLU & Dropout(0.3) → Output: 2 classes | 72.55% | `results/confusion_matrices/benign_malignant/` | ~1.5 hours | Balanced dataset approach; achieved slightly better metrics than the original-only model despite using much fewer samples | Best epoch: 49; val loss: 0.1426; F1 score: 0.7255 |
| MLP3 | `saved_models/benign_malignant/benign_malignant_original_best.pth` | `wandb/run-20250505_210604-y1pt2a99/` | `training/train_benign_malignant.py` | Binary classification of skin lesions as benign or malignant | ISIC 2019 only (original, non-augmented) | 22,094 images (13,598 benign / 8,496 malignant); ~61.5% benign / 38.5% malignant | Same as balanced model | 71.29% | `results/confusion_matrices/benign_malignant/` | ~4.5 hours | Used only original non-augmented images; slightly lower accuracy but similar loss to balanced model | Best epoch: 26; val loss: 0.1373; F1 score: 0.7045 |
| MLP3 | Not confirmed (expected: `benign_malignant_augmented_best.pth`) | `wandb/run-20250506_013422-qklqtlj8/` (in progress) | `training/train_benign_malignant.py` | Binary classification of skin lesions as benign or malignant | ISIC 2019 only (fully augmented) | 110,470 images (67,990 benign / 42,480 malignant); ~61.5% benign / 38.5% malignant | Same as other MLP3 models | Not available (training was in progress) | Not available | 10+ hours (in progress when documented) | Uses complete augmented dataset; results were pending when documented | This configuration tests if large data volume with augmentations improves performance |

## 🔥 Pipeline Flow

Input Image  
→ SAM2 Encoder (frozen) - Outputs 256-dim features  
→ MLP1: Skin/Not Skin  
  ├─ If Not Skin → STOP  
  └─ If Skin →  
    → MLP2: Lesion Type (ISIC only)  
    → MLP3: Benign/Malignant (ISIC only)

After classification:  
→ Generate 3 SAM2 masks 
→ Apply heuristics for visualization

## 🏗️ Overall Architecture

| Module     | Description                            |
|------------|----------------------------------------|
| Encoder    | SAM2 Encoder (ViT, frozen)             |
| Decoder    | 3 MLP heads:                           |
|            | - 🧠 MLP1: Skin/Not Skin Classifier    |
|            | - 🧠 MLP2: Lesion Type Classifier      |
|            | - 🧠 MLP3: Benign/Malignant Classifier |

## 🧠 MLP1: Skin/Not Skin Classifier

### Architecture Details
```python
# From models/skin_not_skin_head.py
class SkinNotSkinHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[512, 256], output_dim=2):
        super(SkinNotSkinHead, self).__init__()
        
        # First hidden layer
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        ]
        
        # Additional hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.classifier = nn.Sequential(*layers)
```

### Production Model
- **Model Path**: `saved_models/skin_not_skin/skin_not_skin_10k_best.pth`
- **Log Path**: `wandb/run-20250430_164542-6scfn1g6/`
- **Training Script**: `training/train_skin_not_skin.py`
- **Performance**: 99.38% validation accuracy (Epoch 37)
- **Training Configuration**:
  - **Dataset**: 2000 samples per class (balanced)
  - **Loss Function**: FocalLoss with class weights [1.0, 5.93]
  - **Optimizer**: Adam (lr=0.001, weight_decay=0.01)
  - **Batch Size**: 16
  - **Early Stopping**: After 5 epochs without improvement

### Layer Architecture Breakdown
1. **Input Layer**: 256-dimensional SAM features
2. **Hidden Layer 1**: Linear(256 → 512) → ReLU → Dropout(0.3)
3. **Hidden Layer 2**: Linear(512 → 256) → ReLU → Dropout(0.3)
4. **Output Layer**: Linear(256 → 2) 

### Architecture Rationale
- **Two Hidden Layers**: Sufficient complexity for binary classification
- **Moderate Dropout (0.3)**: Prevents overfitting without underfitting
- **ReLU Activation**: Standard non-linearity with good gradient properties
- **Progressive Layer Narrowing**: Gradually reduces feature dimensions

## 🧠 MLP2: Lesion Type Classifier

### Architecture Details
```python
# From models/lesion_type_head.py
class LesionTypeHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[512, 256], output_dim=5):
        super(LesionTypeHead, self).__init__()
        
        # First hidden layer
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        ]
        
        # Additional hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.classifier = nn.Sequential(*layers)
```

### Production Model
- **Model Path**: `saved_models/lesion_type/lesion_type_max2000_best.pth`
- **Log Path**: `wandb/run-20250505_165108-jvnir41k/`
- **Training Script**: `training/train_lesion_type_max2k.py`
- **Performance**: 65.8% validation accuracy (Epoch 43)
- **Training Time**: 3.2 hours on NVIDIA RTX 3090
- **Training Configuration**:
  - **Dataset**: Max 2000 samples per majority class, all available for minority classes
  - **Loss Function**: FocalLoss with class weights [1.5, 1.5, 1.5, 1.2, 1.2]
  - **Optimizer**: Adam (lr=0.001, weight_decay=0.01)
  - **Batch Size**: 16
  - **Early Stopping**: After 5 epochs without improvement
  - **F1 Scores**: Range from 0.59 to 0.72 across classes

### Layer Architecture Breakdown
1. **Input Layer**: 256-dimensional SAM features
2. **Hidden Layer 1**: Linear(256 → 512) → ReLU → Dropout(0.3)
3. **Hidden Layer 2**: Linear(512 → 256) → ReLU → Dropout(0.3)
4. **Output Layer**: Linear(256 → 5) 

### Architecture Rationale
- **Same Architecture as MLP1**: Consistent design pattern across models
- **FocalLoss Implementation**: Critical for handling class imbalance
- **Class Weighting**: Essential for balanced performance across all 5 classes
- **Previous Failed Attempts**: Showed that class balance was more important than architecture design

## 🧠 MLP3: Benign/Malignant Classifier

### Architecture Details
```python
# From models/benign_malignant_head.py
class BenignMalignantHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[512, 256], output_dim=2):
        super(BenignMalignantHead, self).__init__()
        
        # First hidden layer
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        ]
        
        # Additional hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.classifier = nn.Sequential(*layers)
```

### Production Model
- **Model Path**: `saved_models/benign_malignant/benign_malignant_balanced_best.pth`
- **Log Path**: `wandb/run-20250505_193729-sxlaexj3/`
- **Training Script**: `training/train_benign_malignant.py`
- **Performance**: 72.55% validation accuracy (Epoch 49)
- **Training Time**: ~1.5 hours
- **Training Configuration**:
  - **Dataset**: 2000 samples per class (balanced)
  - **Loss Function**: FocalLoss with gamma=2.0
  - **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)
  - **Batch Size**: 16 for training, 32 for validation
  - **Early Stopping**: After 5 epochs without improvement
  - **F1 Score**: 0.7255

### Alternative Model (Original Images Only)
- **Model Path**: `saved_models/benign_malignant/benign_malignant_original_best.pth`
- **Log Path**: `wandb/run-20250505_210604-y1pt2a99/`
- **Performance**: 71.29% validation accuracy (Epoch 26)
- **Training Time**: ~4.5 hours
- **Dataset**: Original ISIC images only (22,094 training images)
- **F1 Score**: 0.7045

### Layer Architecture Breakdown
1. **Input Layer**: 256-dimensional SAM features
2. **Hidden Layer 1**: Linear(256 → 512) → ReLU → Dropout(0.3)
3. **Hidden Layer 2**: Linear(512 → 256) → ReLU → Dropout(0.3)
4. **Output Layer**: Linear(256 → 2) 

### Architecture Rationale
- **Same Architecture as Other MLPs**: Consistent design pattern
- **Key Experiment Findings**: Balanced dataset (4k samples) outperformed larger imbalanced dataset (22k samples)
- **Performance/Time Tradeoff**: Balanced model trained 3x faster with slightly better metrics

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

## 📍 Model Locations

### Production Models

| Model | Description | Path | Performance |
|-------|-------------|------|-------------|
| SAM Encoder | Feature extractor (frozen) | `saved_models/sam/sam_vit_h.pth` | N/A (pre-trained) |
| MLP1 | Skin/Not Skin classifier | `saved_models/skin_not_skin/skin_not_skin_10k_best.pth` | 99.38% validation accuracy |
| MLP2 | Lesion Type classifier | `saved_models/lesion_type/lesion_type_max2000_best.pth` | 65.8% validation accuracy |
| MLP3 | Benign/Malignant classifier | `saved_models/benign_malignant/benign_malignant_balanced_best.pth` | 72.55% validation accuracy |

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
checkpoint = torch.load('saved_models/benign_malignant/benign_malignant_balanced_best.pth')
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
| MLP3 (Balanced) | `wandb/run-20250505_193729-sxlaexj3/` | `sxlaexj3` |
| MLP3 (Original) | `wandb/run-20250505_210604-y1pt2a99/` | `y1pt2a99` |
| MLP3 (Augmented) | `wandb/run-20250506_013422-qklqtlj8/` | `qklqtlj8` (in progress) |

## ⚙️ Development Rules

- Update `config.yaml` with all new paths, hyperparameters, and flags.
- Save all intermediate data under `/data/processed/`.
- Keep SAM-specific code inside `/sam/` only.
- Store evaluation results in `/results/`.
- Commit all code changes with clear messages (e.g., `"feat: train MLP1"`, `"fix: mask filtering"`).
- Keep `README.md` and `pipeline_protocol.md` aligned with current code and structure.
- Exclude unnecessary files in `.gitignore` (e.g., datasets, checkpoints, logs, `.ipynb_checkpoints/`)
- 📊 Document all major changes in the training pipeline in `results/experiment_logs.md`

## 🧹 Cleanup Tasks

- [x] **Remove Data Preprocessing Scripts**:
  - Deleted `utils/preprocessing.py` - Main preprocessing implementation
- [x] **Remove Temporary Test Scripts**:
  - Deleted `training/test_single_epoch.py` - Single epoch test script
- [x] **Remove Test Logs**:
  - Deleted `mlp3_test.log` - Test execution log
- [x] **Remove Temporary/Duplicate Utility Scripts**:
  - Deleted `temp_stats.py` - Temporary statistics script
  - Deleted `modified_generate_stats.py` - Modified duplicate of generate_stats.py
- [ ] **Review and Clean Duplicate Training Directories**:
  - Identified potential duplicate: `training/data/processed/` - Contains cached features
  