# Skin Lesion Classifier Backend

FastAPI service exposing a staged ML pipeline:

1. MLP1: Skin vs Not-Skin
2. MLP2: Lesion Type (if skin)
3. MLP3: Benign vs Malignant (if skin)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Health check:

```bash
curl -s http://localhost:8000/healthz | jq
```

Prediction:

```bash
curl -s -X POST http://localhost:8000/predict \
  -F "file=@/path/to/image.jpg" | jq
```

## Models

Place TorchScript files in `backend/models`:

- `mlp1.pt`: binary classifier (index 0 -> skin, 1 -> not_skin)
- `mlp2.pt`: multiclass lesion type
- `mlp3.pt`: binary classifier (index 0 -> benign, 1 -> malignant)

Environment variables:

- `MODEL_DIR` path to models (default `backend/models`)
- `TORCH_DEVICE` e.g., `cuda`, `cpu`
- `PREPROCESS_SIZE` image resize (default 224)
- `PREPROCESS_MEAN` e.g., `0.485,0.456,0.406`
- `PREPROCESS_STD` e.g., `0.229,0.224,0.225`


