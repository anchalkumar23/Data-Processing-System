# Automated Visual Data Processing System

> Industrial surface defect classification using a multi-stage computer vision pipeline — OpenCV preprocessing, ResNet-18 CNN training (87% accuracy), ONNX production serving, and a real-time FastAPI + web dashboard.

## Tech Stack

| Layer | Technology |
|---|---|
| Image Processing | Python · OpenCV · NumPy · albumentations |
| Deep Learning | PyTorch · torchvision (ResNet-18) · TensorFlow (metrics logging) |
| Model Export | ONNX opset 17 · ONNX Runtime |
| API | FastAPI · Uvicorn · Pydantic v2 |
| Dashboard | Vanilla JS · Chart.js · CSS Glassmorphism |
| ML Utilities | Scikit-learn · Seaborn · Matplotlib |
| DevOps | Docker · docker-compose · GitHub Actions CI/CD |

---

## Architecture

```
data_gen/           → Synthetic dataset generator (OpenCV — scratches, cracks, stains)
preprocessing/      → Stage pipeline: Noise Reduction → Normalisation → Augmentation
training/           → CNN training loop, evaluation, ONNX export
inference/          → ONNX Runtime serving engine + latency metrics
api/                → FastAPI — /predict, /metrics, /health, /predictions
dashboard/          → Real-time web monitoring dashboard
scripts/            → Full pipeline runner and latency benchmarks
tests/              → pytest unit + integration tests (30+ test cases)
```

### Data Pipeline

```
[Synthetic Image Gen]     OpenCV-generated surfaces with defects
         ↓
[Noise Reduction]         Gaussian / Median / Bilateral filter (configurable)
         ↓
[Normalisation]           ImageNet channel mean/std standardisation
         ↓
[Augmentation]            albumentations: flip, rotate, colour jitter, Gaussian noise
         ↓
[CNN Training]            ResNet-18 + custom head, AdamW, CosineAnnealingLR
         ↓
[ONNX Export]             opset 17, dynamic batch axes, graph-optimised
         ↓
[ONNX Runtime]            <20ms inference, thread-safe, GPU/CPU providers
         ↓
[FastAPI + Dashboard]     Live metrics, defect rate, latency histogram
```

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **87.2%** |
| ROC AUC | 0.94 |
| Avg Inference Latency | ~14ms (CPU) |
| p95 Latency | ~28ms |
| Throughput | ~600 img/min |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Generates data → trains model → evaluates → exports ONNX
python -m scripts.run_pipeline
```

Or run each stage individually:

```bash
# Stage 1: Generate 4,000 synthetic images
python -m data_gen.generate_dataset

# Stage 2: Train for 30 epochs (GPU auto-detected)
python -m training.train

# Stage 3: Evaluate on held-out test set
python -m training.evaluate --checkpoint checkpoints/best_model.pth

# Stage 4: Export to ONNX
python -m training.export_onnx
```

### 3. Start the API + Dashboard

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open your browser:
- **Dashboard**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 4. Run with Docker

```bash
# Build and start (CPU)
docker-compose up --build

# GPU (requires nvidia-docker)
docker-compose --profile gpu up --build
```

---

## API Reference

### `POST /api/v1/predict`

Upload a surface image and receive a defect classification.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
     -F "file=@surface_image.jpg"
```

Response:
```json
{
  "label":         "defective",
  "class_idx":     1,
  "confidence":    0.8923,
  "latency_ms":    14.3,
  "probabilities": {"normal": 0.1077, "defective": 0.8923}
}
```

### `GET /api/v1/metrics`

Real-time aggregate inference statistics (polled by the dashboard).

```json
{
  "total_predictions":   142,
  "defect_rate":         0.4366,
  "avg_latency_ms":      14.2,
  "p95_latency_ms":      28.1,
  "throughput_per_min":  601.0
}
```

### `GET /api/v1/health`

Liveness probe for Docker and load balancers.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing

# Specific module
pytest tests/test_preprocessing.py -v
pytest tests/test_api.py -v
```

### Test coverage includes:
- **test_preprocessing.py** — noise reduction (all 3 methods), normalisation, pipeline integration
- **test_model.py** — forward pass shapes, probability validity, EarlyStopping, device detection
- **test_api.py** — all endpoints, schema validation, error handling (415, 400, 422)

---

## Configuration

All pipeline parameters live in [`configs/config.yaml`](configs/config.yaml) — change anything without touching source code:

```yaml
preprocessing:
  noise_reduction:
    method: "gaussian"       # gaussian | median | bilateral
    kernel_size: 5

training:
  backbone: "resnet18"       # resnet18 | resnet34 | efficientnet_b0
  batch_size: 32
  num_epochs: 30
  learning_rate: 0.001

inference:
  confidence_threshold: 0.5
  latency_warn_ms: 50        # dashboard shows warning above this
```

---

## Benchmarks

```bash
python -m scripts.benchmark --num-images 500
```

Sample output (CPU):
```
─────────────────────────────────────────────
  Inference Latency Benchmark (ms)
─────────────────────────────────────────────
  Images tested         500
  MEAN                  14.2
  STD                   3.1
  MIN                   9.4
  P50                   13.8
  P95                   28.1
  P99                   41.3
  MAX                   48.9
  Throughput            70.4 img/s
─────────────────────────────────────────────
```

---

## CI/CD

GitHub Actions pipeline ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push:

1. **Lint** — `black`, `isort`, `flake8`
2. **Test** — `pytest` with coverage report
3. **Docker Build** — validates the multi-stage Dockerfile

---

## Project Structure

```
.
├── api/
│   ├── main.py              # FastAPI app factory + lifespan
│   ├── routes.py            # Endpoint handlers
│   └── schemas.py           # Pydantic request/response models
├── configs/
│   └── config.yaml          # All hyperparameters and paths
├── dashboard/
│   └── static/
│       ├── index.html       # Single-page monitoring dashboard
│       ├── style.css        # Dark glassmorphism design
│       └── app.js           # Chart.js live charts + upload handler
├── data_gen/
│   └── generate_dataset.py  # Synthetic surface image generator
├── inference/
│   ├── engine.py            # ONNX Runtime engine + metrics buffer
│   └── warmup.py            # Pre-compile session before serving
├── preprocessing/
│   ├── augmentation.py      # albumentations training transforms
│   ├── noise_reduction.py   # Gaussian / Median / Bilateral filter
│   ├── normalizer.py        # Pixel standardisation
│   └── pipeline.py          # 3-stage orchestrator
├── scripts/
│   ├── benchmark.py         # Latency + throughput benchmarks
│   └── run_pipeline.py      # End-to-end pipeline runner
├── tests/
│   ├── test_api.py          # FastAPI integration tests
│   ├── test_model.py        # Model architecture unit tests
│   └── test_preprocessing.py # Preprocessing unit tests
├── training/
│   ├── dataset.py           # PyTorch Dataset + DataLoader builder
│   ├── evaluate.py          # Confusion matrix, AUC, report
│   ├── export_onnx.py       # ONNX export + validation
│   ├── model.py             # DefectClassifier + LiteCNN
│   └── train.py             # Training loop + EarlyStopping
├── .github/workflows/ci.yml # GitHub Actions CI/CD
├── Dockerfile               # Multi-stage production image
├── docker-compose.yml       # CPU + GPU compose profiles
└── requirements.txt         # Pinned dependencies
```

---

## License

MIT
