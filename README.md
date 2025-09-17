# 🔬 Skin Lesion Classification Pipeline

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tparkhomenko/newbac)

A deep learning pipeline for skin lesion classification using SAM2 encoder features.

## 📋 Project Overview

This project implements a multi-head classification pipeline for skin lesion analysis:

1. **Skin/Not-Skin Classification (MLP1)**: Binary classification to identify skin images
2. **Lesion Type Classification (MLP2)**: Multi-class classification for 8 lesion classes (Exp1)
3. **Benign/Malignant Classification (MLP3)**: Binary classification of lesion malignancy
4. NEW: **Direct Final Multi-Class Head (optional)**: Single-head prediction over 11 final classes (combined lesion + benign/malignant + NOT_SKIN)

All classifiers use features extracted from a frozen SAM2 encoder (ViT architecture).

## 🧠 Architecture

- **Feature Extractor**: SAM2 ViT-H (frozen)
- **Classifiers**:
  - MLP1: Skin vs. Not-Skin (2 classes)
  - MLP2: Lesion Type (8 classes in Exp1)
  - MLP3: Benign vs. Malignant (2 classes)
  - Optional Final Head: 11-class direct final label

## 📊 Datasets

- ISIC 2019 (primary)
- DTD (textures) as non-skin
- Metadata: `data/metadata/metadata.csv` with experiment columns (`exp1`, ...). For `exp_finalmulticlass`, data loading maps to `exp1`.

## 🔧 Technical Implementation Notes

### Latest changes
- Added `exp_finalmulticlass` experiment with `final` label/mask in `datasets/parallel_unified_dataset.py` (`num_final_classes = 11`).
- Added optional `head_final` in `models/multitask_model.py`.
- Updated `training/train_parallel.py`:
  - Dynamic loss over available heads, `final`-only training for `--experiment exp_finalmulticlass`.
  - Metrics use weighted F1/precision/recall.
  - Added `--oversample_lesion` using `WeightedRandomSampler` for `lesion` (Exp1) or `final` (exp_finalmulticlass).
  - W&B logging and `backend/experiments_summary.csv` append.
- Added `scripts/plot_confusions.py` to save confusion matrices (CSV + PNG) for lesion and pipeline/final modes.

### Background removal
- No dedicated background-removal step exists. SAM is used for features; we do not mask/zero the background in preprocessing.

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training (examples)

```bash
# 1) Baseline (no oversampling)
python training/train_parallel.py \
  --experiment exp1_parallel_lmf \
  --epochs 30 --batch_size 64 \
  --wandb_project lesion_ablation \
  --run_name baseline_no_oversample

# 2) Oversampling (B)
python training/train_parallel.py \
  --experiment exp1_parallel_lmf \
  --epochs 30 --batch_size 64 \
  --oversample_lesion \
  --wandb_project lesion_ablation \
  --run_name oversample_lesion

# 3) Final multi-class (A)
python training/train_parallel.py \
  --experiment exp_finalmulticlass \
  --epochs 30 --batch_size 64 \
  --wandb_project lesion_ablation \
  --run_name final_multiclass

# 4) Final multi-class + oversampling (A+B)
python training/train_parallel.py \
  --experiment exp_finalmulticlass \
  --epochs 30 --batch_size 64 \
  --oversample_lesion \
  --wandb_project lesion_ablation \
  --run_name final_multiclass_oversample
```

### Evaluation and Confusions

```bash
# Plot/save confusion matrices for a checkpoint
python scripts/plot_confusions.py \
  --checkpoint backend/models/parallel/lesion_ablation/<run>/<timestamp>/final.pt \
  --wandb_project lesion_ablation_eval
```

- Confusion matrices and CSVs are saved alongside checkpoints under `backend/models/parallel/<project>/<run>/<timestamp>/`.

## 📄 Documentation

- Results summary recorded in `backend/experiments_summary.csv`.
- Metrics use accuracy and weighted F1/precision/recall.
- See `protocol.md` for up-to-date pipeline and experiment details.

## 📊 Results (recent)

- Baseline vs Oversampling: oversampling degraded lesion performance in our runs.
- Direct final multi-class head performed comparably to multi-stage pipeline on Exp1 data.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
