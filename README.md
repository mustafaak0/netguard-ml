# netguard-ml

**Network anomaly detection with an Isolation Forest + Autoencoder ensemble.**

[![CI](https://github.com/mustafaak0/netguard-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/mustafaak0/netguard-ml/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`netguard-ml` scores network flows for anomaly risk and explains *which features* drove each
score. It combines two unsupervised detectors that catch different attack shapes:

- **Isolation Forest** — fast global outlier detection (low overhead, strong on point outliers)
- **Dense Autoencoder** — reconstruction-error detection (strong on subtle multivariate patterns)

Both detectors are calibrated to `[0, 1]`, blended into a **0–100 risk score** with a severity
level (`normal` / `suspicious` / `critical`) and a per-flow feature explanation derived from the
autoencoder's reconstruction errors.

## How it works

```
                       ┌──────────────────────┐
  network flows ──►    │ preprocessor         │  one-hot (categorical) + scale (numeric)
  (CSV / JSON / API)   └──────────┬───────────┘
                                  │
                 ┌────────────────┴───────────────┐
                 ▼                                ▼
      ┌─────────────────────┐          ┌─────────────────────┐
      │ Isolation Forest    │          │ Autoencoder         │
      │ (sklearn)           │          │ (PyTorch)           │
      └──────────┬──────────┘          └──────────┬──────────┘
                 │ score 0..1                     │ score 0..1 + per-feature errors
                 └────────────────┬───────────────┘
                                  ▼
                      ┌───────────────────────┐
                      │ RiskScorer (ensemble) │
                      │ risk 0..100 + level   │
                      │ + top-k features      │
                      └───────────────────────┘
```

## Quickstart

```bash
# 1. install (CPU torch is enough)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .

# 2. generate a synthetic KDD-style dataset (or bring your own CSV)
netguard generate-data --samples 20000 --out data/sample.csv

# 3. train
netguard train --data data/sample.csv --out artifacts

# 4. score flows
netguard predict --data data/sample.csv --model artifacts

# 5. serve the REST API
netguard serve --model artifacts --port 8000
```

### Example output

```
[8] risk=100 level=critical   top=srv_count
[9] risk= 12 level=normal     top=src_bytes
```

```json
{
  "risk_score": 100,
  "level": "critical",
  "detector_scores": {"isolation_forest": 1.0, "autoencoder": 1.0},
  "top_features": [
    {"feature": "srv_count", "contribution": 0.316},
    {"feature": "count", "contribution": 0.265},
    {"feature": "src_bytes", "contribution": 0.105}
  ]
}
```

## REST API

`netguard serve` (or `uvicorn netguard.api:app`) exposes:

| Endpoint           | Method | Description                         |
| ------------------ | ------ | ----------------------------------- |
| `/health`          | GET    | Liveness + model status             |
| `/model/info`      | GET    | Detectors, weights, training metrics|
| `/predict`         | POST   | Score a single flow                 |
| `/predict/batch`   | POST   | Score up to 10,000 flows            |

Interactive docs live at `/docs` (Swagger UI).

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"duration": 2.0, "src_bytes": 500000, "dst_bytes": 0.0,
       "count": 150, "srv_count": 140, "serror_rate": 1.0, "same_srv_rate": 0.05,
       "protocol_type": "tcp", "service": "http", "flag": "S0",
       "...": "remaining numeric features as floats"}}'
```

## Docker

```bash
docker build -t netguard-ml .
docker run --rm -p 8000:8000 netguard-ml
```

The image trains on a bundled synthetic dataset at build time, so the container
is immediately usable; mount your own artifacts directory to serve a custom model:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/artifacts:/artifacts" netguard-ml
```

## Using real datasets

The pipeline is dataset-agnostic: any CSV with numeric/categorical feature columns works.
For the classic **NSL-KDD** benchmark, keep the default categorical columns
(`protocol_type`, `service`, `flag`) and a `label` column — when labels are present,
training automatically:

- fits detectors on **normal traffic only**,
- computes ROC AUC per detector and for the ensemble on a stratified holdout,
- reports attack recall and false-alarm rate at the critical threshold.

Other schemas just need `categorical_columns` adjusted in `configs/default.yaml`.

## Configuration

All knobs live in [`configs/default.yaml`](configs/default.yaml):

| Section           | Key                           | Default     | Meaning                              |
| ----------------- | ----------------------------- | ----------- | ------------------------------------ |
| `seed`            |                               | `42`        | Global reproducibility seed          |
| `isolation_forest`| `n_estimators` / `contamination` | `200` / `0.05` | Forest size / expected anomaly rate |
| `autoencoder`     | `encoder_dims`                | `[64, 32]`  | Encoder layers; last = latent dim    |
| `autoencoder`     | `epochs` / `batch_size` / `learning_rate` | `40` / `256` / `1e-3` | Optimization |
| `ensemble`        | `*_weight`                    | `0.5/0.5`   | Detector blend weights               |
| `ensemble`        | `critical_threshold` / `suspicious_threshold` | `70` / `50` | Risk level cut-offs |
| `preprocess`      | `categorical_columns`         | KDD columns | Which columns to one-hot encode      |

## Project structure

```
netguard/
├── config.py               # YAML-backed configuration dataclasses
├── data.py                 # loading, validation, synthetic traffic generator
├── preprocess.py           # one-hot + scaling pipeline (persistable)
├── models/
│   ├── base.py             # Detector interface (fit / score_samples)
│   ├── isolation_forest.py # sklearn Isolation Forest wrapper
│   └── autoencoder.py      # PyTorch autoencoder (optional dependency)
├── ensemble.py             # weighted blending, risk levels, explanations
├── train.py                # training pipeline + artifact persistence
├── predict.py              # Predictor: load artifacts, score flows
├── api.py                  # FastAPI service
└── cli.py                  # `netguard` command line tool
tests/                      # pytest suite (data, models, pipeline, API)
configs/default.yaml        # training configuration
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v      # full suite
ruff check netguard tests
```

> PyTorch is optional: without it the ensemble degrades gracefully to the
> Isolation Forest alone (autoencoder tests are skipped automatically).

## Notes for defenders

This tool is built for **defensive** monitoring — SOC triage, lab research and
education. It is an anomaly *detector*, not an intrusion-prevention system:
treat scores as leads for investigation, not verdicts. Always validate on your
own traffic before acting on alerts.

## License

[MIT](LICENSE) © Mustafa Ak
