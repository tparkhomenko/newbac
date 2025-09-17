# Protocol: Skin Lesion Classifier (Backend + Frontend)

This document captures the complete setup, configuration, and operational steps for the demo.

## 1) Project Structure (key paths)
- Backend API: `backend/`
  - Entry: `backend/main.py`
  - Training scripts: `training/train_parallel.py`, `training/train_multihead.py`
  - Models (checkpoints): `backend/models/parallel/<project>/<run>/<timestamp>/final.pt`
  - Logs and summaries: `backend/experiments_summary.csv`
  - Pipeline: `backend/pipeline/`
- Frontend (React + Vite): `frontend/`
- Data:
  - Metadata CSV: `data/metadata/metadata.csv`
  - Cleaned images: `data/cleaned_resized/isic2018_512/`
  - QuickTest images: `data/testing_only/` (auto-filled up to 100 exp1 test images)

## 2) Requirements
- Python 3.12
- Node.js 18
- CUDA GPU optional (SAM + Torch accelerate on GPU if available)

### Python deps
- Install: `pip install -r requirements.txt`
- Important packages: `fastapi`, `uvicorn`, `torch`, `torchvision`, `pillow`, `numpy`, `pydantic`, `python-multipart`, `pyyaml`, `scikit-learn`, `seaborn`, `pandas`, `wandb`

### Node deps
- `cd frontend && npm install`

## 3) Environment variables
- `MODEL_DIR` (optional): base models dir (default: `backend/models/`)
- `MODEL_ARCH` (optional): initial architecture (`multi` | `parallel`; default: `multi`)
- `TORCH_DEVICE` (optional): `cuda` or `cpu` (auto-detects if unset)
- `SAM_MODEL_TYPE` (optional): default `vit_h`

## 4) Model files and logs
- Parallel checkpoints and artifacts:
  - `backend/models/parallel/<project>/<run>/<timestamp>/final.pt`
  - confusion matrices: `lesion_confusion_{csv,png}`, `pipeline_confusion_{csv,png}` if generated
- Legacy TorchScript files (if using the older serving path):
  - `backend/models/{parallel|multi}/mlp{1,2,3}.pt`

## 5) Data
- `data/metadata/metadata.csv` must exist; key columns include: `image_name`, `unified_diagnosis`, `exp1 .. exp5`.
- For `exp_finalmulticlass`, dataset loading maps to `exp1` column.
- QuickTest uses `data/testing_only/` and will auto-populate from `data/cleaned_resized/isic2018_512/` using `exp1=test` (up to 100 images).

## 6) Backend: run and endpoints
- Start: `uvicorn backend.main:app --host 127.0.0.1 --port 8000`
- Health: `GET /healthz`
- Predict: `POST /predict` (multipart form, field `file`)
- Model info: `GET /model`
- Switch model: `POST /model/switch` with JSON `{ "architecture": "multi" | "parallel" }`
- Quick test: `POST /quicktest` → accuracy, weighted-F1, per-class accuracy, confusion matrix

## 7) Frontend: run and configuration
- Start dev: `cd frontend && npm run dev` → `http://localhost:3000`
- Backend URL: default `http://127.0.0.1:8000` in components (search/replace if needed)
- UI:
  - Model panel: toggle `Parallel | Multi`, stats per `mlp1..3`
  - Quick Test: runs `/quicktest`, displays accuracy/F1/confusion matrix
  - Predictions: Skin/LesionType/Malignancy with probabilities

## 8) Label mapping (GT abbreviations → full)
- `NV`: `nevus`
- `MEL`: `melanoma`
- `BCC`: `basal_cell_carcinoma`
- `BKL`: `seborrheic_keratosis`
- `AKIEC`: `actinic_keratosis`
- `SCC`: `squamous_cell_carcinoma`
- `VASC`: `vascular_lesion`
- `DF`: `dermatofibroma`
- `UNKNOWN`: `unknown`

## 9) Training recipes

### Parallel trainer (shared trunk + heads)
- Script: `training/train_parallel.py`
- Key flags:
  - `--experiment {exp1_parallel_lmf | exp_finalmulticlass}`
  - `--epochs`, `--batch_size`, `--learning_rate`
  - `--wandb_project`, `--run_name`
  - `--oversample_lesion` (enables `WeightedRandomSampler` on `lesion` for `exp1_parallel_lmf` or `final` for `exp_finalmulticlass`)
- Behavior:
  - Loss computed over available heads; when `exp_finalmulticlass`, trains/evaluates only `final` head
  - Metrics: accuracy and weighted F1/precision/recall
  - Appends results to `backend/experiments_summary.csv`

### Multihead trainer (legacy separate heads)
- Script: `training/train_multihead.py`
- Metrics now use weighted averaging for F1/precision/recall

## 10) Evaluation
- Confusion matrices script: `scripts/plot_confusions.py`
  - Args: `--checkpoint`, `--wandb_project`, `--run_name`(opt), `--eval_mode {lesion_headwise | pipeline_taskaware | both}`
  - Saves CSV and PNG in checkpoint dir
- Summary CSV: `backend/experiments_summary.csv`

## 11) Recent experiments
- A) Direct final multi-class head (`exp_finalmulticlass`): comparable to multi-stage pipeline on Exp1
- B) Oversampling (`--oversample_lesion`): degraded lesion performance in our runs

## 12) Preprocessing notes
- Image preprocessing does resize/normalize; no explicit background-removal mask is applied.
- SAM is used for feature extraction; no background zeroing is performed.

## 13) Troubleshooting
- Port busy: `pkill -f "uvicorn"` / `pkill -f "vite"`
- Missing models: backend falls back for MLPs; SAM may still load
- QuickTest 404s: ensure `data/cleaned_resized/isic2018_512/` exists and CSV has `exp1=test`

## 14) Version & hygiene
- Use `.gitignore`; consider LFS for large `.pt`
- Track `metadata.csv` (small); avoid committing dataset binaries
