# Protocol: Skin Lesion Classifier (Backend + Frontend)

This document captures the complete setup, configuration, and operational steps for the demo.

## 1) Project Structure (key paths)
- Backend API: `backend/`
  - Entry: `backend/main.py`
  - Models: `backend/models/{parallel|multi}/mlp{1,2,3}.pt`
  - Logs for stats: `backend/models/{parallel|multi}/logs_mlp{1,2,3}.txt`
  - Pipeline: `backend/pipeline/`
- Frontend (React + Vite): `frontend/`
- Data:
  - Metadata CSV: `data/metadata/metadata.csv`
  - Cleaned images: `data/cleaned_resized/isic2018_512/`
  - QuickTest images: `data/testing_only/` (auto-filled up to 100 exp1 test images)

## 2) Requirements
- Python 3.12
- Node.js 9 18
- CUDA GPU optional (SAM + Torch accelerate on GPU if available)

### Python deps
- Install: `pip install -r backend/requirements.txt`
- Important packages: `fastapi`, `uvicorn`, `torch`, `torchvision`, `pillow`, `numpy`, `pydantic`, `python-multipart`, `segment-anything`, `pyyaml`

### Node deps
- `cd frontend && npm install`

## 3) Environment variables
- `MODEL_DIR` (optional): override base models dir (default: `backend/models/`)
- `MODEL_ARCH` (optional): initial architecture (`multi` | `parallel`; default: `multi`)
- `TORCH_DEVICE` (optional): `cuda` or `cpu` (auto-detects if unset)
- `SAM_MODEL_TYPE` (optional): default `vit_h`

## 4) Model files and logs
- Place TorchScript weights as:
  - `backend/models/multi/mlp1.pt`
  - `backend/models/multi/mlp2.pt`
  - `backend/models/multi/mlp3.pt`
  - `backend/models/parallel/mlp1.pt`
  - `backend/models/parallel/mlp2.pt`
  - `backend/models/parallel/mlp3.pt`
- Optional stats logs (parsed for frontend Model panel):
  - `backend/models/multi/logs_mlp1.txt` (and 2,3)
  - `backend/models/parallel/logs_mlp1.txt` (and 2,3)
  - Expected lines in final ~50 lines:
    - `wandb: test_skin_acc 0.99881`
    - `wandb: test_lesion_acc 0.51674`
    - `wandb: test_bm_acc 0.74648`
    - `wandb: test_skin_f1_macro 0.98162`
    - `wandb: test_lesion_f1_macro 0.37602`
    - `wandb: test_bm_f1_macro 0.72808`

## 5) Data
- `data/metadata/metadata.csv` must exist; columns used: `image_name`, `unified_diagnosis`, `exp1`  `exp5`
- QuickTest pulls images from `data/testing_only/` and will auto-populate from `data/cleaned_resized/isic2018_512/` using exp1=test in CSV (up to 100 images)

## 6) Backend: run and endpoints
- Start: `uvicorn backend.main:app --host 127.0.0.1 --port 8000`
- Health: `GET /healthz`
- Predict: `POST /predict` (multipart form, field `file`)
- Model info: `GET /model`
  - Returns active architecture and per-MLP stats parsed from logs
- Switch model: `POST /model/switch` with JSON `{ "architecture": "multi" | "parallel" }`
  - Reloads pipeline using `backend/models/{arch}/`
- Quick test: `POST /quicktest`
  - Evaluates up to 100 images in `data/testing_only/`
  - Returns accuracy, macro-F1, per-class accuracy, confusion matrix

## 7) Frontend: run and configuration
- Start dev: `cd frontend && npm run dev`  `http://localhost:3000`
- Backend URL: hardcoded to `http://127.0.0.1:8000` in components.
  - Change in: `frontend/src/components/*` if needed (search for `http://127.0.0.1:8000`)
- UI:
  - Model panel: toggle `Parallel | Multi`, stats per `mlp1..3`
  - Quick Test: runs `/quicktest`, displays accuracy/F1/confusion matrix
  - Predictions: shows Skin/LesionType/Malignancy with probabilities
  - Ground truth: abbreviations expanded (`NV→nevus`, `MEL→melanoma`, etc.)

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

## 9) Troubleshooting
- Port busy: `pkill -f "uvicorn"` / `pkill -f "vite"`
- Missing models: backend falls back for MLPs; SAM still loads
- `/quicktest` returns 404 images: ensure `data/cleaned_resized/isic2018_512/` exists and CSV has exp1=test
- Stats missing: ensure `logs_mlp{1,2,3}.txt` exist under selected architecture

## 10) Git hygiene
- Use `.gitignore` (see root) to avoid committing large binaries/data
- Consider Git LFS for `.pt` weights if you must track them
- Keep `metadata.csv` tracked; keep small `testing_only` README tracked (images optional)

## 11) Pipeline Protocol (merged summary)
This section consolidates the full contents of `pipeline_protocol.md`.

### Rules & Guidelines
- Keep key files updated: `.gitignore`, `requirements.txt`, `protocol.md`, `common_errors.md`.
- Archive previous protocol versions after major changes.
- Document all pipeline stages: Raw → Cleaned → Processed → Features.
- Track dataset statistics and maintain reproducible splits (fixed seeds, stratification).

### Data Pipeline Rules
- Raw data validation (≥100 images per dataset), CSV verification, 512×512 size checks.
- Feature extraction logging (SAM2), experiment 70/15/15 integrity.

### Model Training Rules
- Consistent MLP design for MLP1/2/3; handle imbalance; save checkpoints with metadata; WandB/CSV logging; GPU mem cleanup.

### Evaluation Rules
- Accuracy, macro-F1, precision/recall, AUC; confusion matrices and visualizations (UMAP/T-SNE/ROC); error analysis.

### Current Metadata CSV Stats (live)
- File: `data/metadata/metadata.csv`, Total: 67,332; columns include `image_name`, `unified_diagnosis`, `exp1..exp5`.
- Unified diagnosis counts and experiment coverage summarized in the original file.

### Training Architectures
- Separate (independent heads) vs Parallel (shared trunk + heads). Pros/cons as described above.

### Project Structure v2 (condensed)
- Data folders (`raw`, `cleaned_resized`, `processed`, `metadata`), models, training, evaluation, utils, configs, saved_models, results, logs, wandb.

### Experiment System v2
- Five experiment modes (exp1 full, exp2 balanced subset, exp3 rare-first, exp4 prioritize minorities, exp5 small classes only) with dataset sizes and splits as in source document.

### Technical Implementation
- Live metadata at `data/metadata/metadata.csv` with `exp1..exp5` columns; usage examples for filtering provided in source.

### Pipeline Flow v2
- Input → SAM2 features → MLP1 (stop if not-skin) → MLP2 (5-class) → MLP3 (benign/malignant) → masks/visualization.

### Development Workflow v2
- Stages: validation → preprocessing → feature extraction (batching, resume, logging) → splitting → training → evaluation.
- Quality assurance: validation, fixed seeds, logging, graceful error handling.

### Version History & Status
- v1 achievements and v2 goals summarized (see source for detailed counts and plots).

Archived originals: `pipeline_protocol.md`, `pipeline_protocol_v1.md`.
