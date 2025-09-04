## Quick Test (multi-task, configurable max) – How to run and what to expect

This document explains how to run the backend Quick Test locally on configurable numbers of images in `data/testing_only/` and how to interpret the output and diagnostics across all three tasks: skin, lesion, and malignancy (bm).

Note: The `data/testing_only/` folder may contain a 500‑image dataset composed of 50 images for each of 10 unified classes. The Quick Test endpoint evaluates up to `max` images from that folder (default: 100) and computes metrics for:
- Skin (2 classes): `skin`, `not_skin`
- Lesion (8 classes): `melanoma`, `nevus`, `basal_cell_carcinoma`, `squamous_cell_carcinoma`, `seborrheic_keratosis`, `actinic_keratosis`, `dermatofibroma`, `vascular`
- Malignancy (2 classes): `benign`, `malignant`

Samples with missing ground truth (e.g. `lesion_label: -`) are skipped for that task.

### 1) Prepare a test set (optional)

From the project root:

```bash
source .venv/bin/activate
python create_balanced_testing_dataset.py
```

This creates or refreshes `data/testing_only/`. Your setup may already contain 500 images (50 per class across 10 unified labels). The Quick Test endpoint will evaluate up to `max` images from this folder.

Structured testing metadata (recommended): create `data/testing_only/metadata.csv` with columns:
- `filename` (without extension)
- `skin_label` in {`skin`, `not_skin`}
- `lesion_label` in {`MEL`, `NV`, `BCC`, `SCC`, `BKL`, `AKIEC`, `DF`, `VASC`, `-`}
- `bm_label` in {`benign`, `malignant`, `-`}
- `unified_label` (10-class, optional for compatibility)

### 2) Start the backend

```bash
source .venv/bin/activate
HOST=127.0.0.1 PORT=8000 \
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info
```

Verify it is up:

```bash
curl -s http://127.0.0.1:8000/healthz | jq .
```

### 3) Run Quick Test (configurable max images)

**Default (100 images for speed):**
```bash
curl -s -X POST http://127.0.0.1:8000/quicktest | jq .
```

**Custom max (e.g., 500 images):**
```bash
curl -s -X POST "http://127.0.0.1:8000/quicktest?max=500" | jq .
```

**Any number of images:**
```bash
curl -s -X POST "http://127.0.0.1:8000/quicktest?max=1000" | jq .
```

Response JSON includes:
- `total_images`
- `model_used`
- `skin` → `{ num_samples, accuracy, f1_macro, confusion_matrix, class_names }`
- `lesion` → `{ num_samples, accuracy, f1_macro, confusion_matrix, class_names }`
- `bm` → `{ num_samples, accuracy, f1_macro, confusion_matrix, class_names }`
- `test_time`

### 4) Per‑image diagnostics (backend console)

Keep the backend terminal open while running Quick Test. For the first 5 images it prints:
- File name
- Raw MLP2 lesion logits shape (should be `[1, 8]`)
- Softmaxed lesion probabilities (rounded) and Top‑3 classes
- Predicted index / label / confidence
- Ground Truth labels for skin, lesion, and bm

This confirms:
- Lesion classification uses MLP2 (8‑class head) with softmax before argmax
- The probabilities are distributed correctly
- GT labels align with metadata

### 5) Label space used by Quick Test

Quick Test evaluates lesion accuracy over the 8 lesion classes (lesion head):

```text
["melanoma","nevus","basal_cell_carcinoma","squamous_cell_carcinoma",
 "seborrheic_keratosis","actinic_keratosis","dermatofibroma","vascular"]
```

Additional notes:
- The dataset may include 10 unified labels overall, but lesion metrics are computed on the 8 lesion classes listed above. Images with `unknown` or `not_skin` labels are ignored for lesion evaluation (and `not_skin` is gated by MLP1).
- MLP1 (skin/not_skin) is only used as a gate; if `not_skin`, lesion evaluation is skipped.
- Lesion predictions are strictly from MLP2 (8‑class head), with `softmax(dim=1)` applied to logits before argmax.
- Backward compatibility: if only `unified_label` is present, the endpoint auto-inferrs `skin_label`, `lesion_label` (abbr), and `bm_label` as best-effort.

### 6) Common checks

- If metrics seem off, verify logits shape and class_names in the console output.
- Ensure the testing subset is representative (the supplied script attempts a balanced sample).
- Training logs (W&B/printed) are the reference for expected validation/test performance.
- Use `?max=500` to test all images in the folder for comprehensive evaluation.


