"""FastAPI prediction service.

Run locally:
    netguard serve --model artifacts            # or
    uvicorn netguard.api:app --model-dir artifacts

The model directory is resolved from --model-dir (uvicorn factory arg),
the NETGUARD_MODEL_DIR environment variable, or ``artifacts`` by default.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from netguard.data import LABEL_COLUMN
from netguard.predict import Predictor

_DEFAULT_MODEL_DIR = "artifacts"
_state: dict[str, Predictor] = {}


def _resolve_model_dir() -> Path:
    import os

    return Path(os.environ.get("NETGUARD_MODEL_DIR", _DEFAULT_MODEL_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = _resolve_model_dir()
    try:
        _state["predictor"] = Predictor.load(model_dir)
    except Exception as exc:  # pragma: no cover - startup failure path
        raise RuntimeError(
            f"could not load model from {model_dir}: {exc}. "
            "Train first with `netguard train` or set NETGUARD_MODEL_DIR."
        ) from exc
    yield
    _state.clear()


app = FastAPI(
    title="netguard-ml",
    version="0.1.0",
    description="Network anomaly detection with an Isolation Forest + Autoencoder ensemble",
    lifespan=lifespan,
)


class FlowRecord(BaseModel):
    """A single network flow; fields are validated against the training schema."""

    features: dict[str, Any] = Field(
        ...,
        description="Feature name -> value mapping (e.g. src_bytes, protocol_type)",
        examples=[
            {
                "duration": 2.0,
                "src_bytes": 500000,
                "dst_bytes": 0.0,
                "wrong_fragment": 0,
                "urgent": 0,
                "count": 150,
                "srv_count": 140,
                "serror_rate": 1.0,
                "srv_serror_rate": 1.0,
                "same_srv_rate": 0.05,
                "dst_host_count": 255,
                "dst_host_srv_count": 255,
                "dst_host_same_srv_rate": 0.05,
                "protocol_type": "tcp",
                "service": "http",
                "flag": "S0",
            }
        ],
    )


class BatchRequest(BaseModel):
    records: list[FlowRecord] = Field(..., min_length=1, max_length=10_000)


def _predict_records(records: list[FlowRecord]) -> dict:
    predictor: Predictor = _state["predictor"]
    df = pd.DataFrame([r.features for r in records])
    if LABEL_COLUMN in df.columns:
        df = df.drop(columns=[LABEL_COLUMN])
    try:
        results = predictor.predict(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "count": len(results),
        "predictions": [r.to_dict() for r in results],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "predictor" in _state}


@app.get("/model/info")
def model_info() -> dict:
    predictor: Predictor = _state["predictor"]
    return {
        "detectors": predictor.detectors,
        "ensemble_weights": predictor.scorer.weights,
        "n_encoded_features": predictor.preprocessor.n_features_,
        "trained_at": predictor.metadata.get("trained_at"),
        "metrics": predictor.metadata.get("metrics"),
    }


@app.post("/predict")
def predict(record: FlowRecord) -> dict:
    """Score a single network flow."""
    return _predict_records([record])


@app.post("/predict/batch")
def predict_batch(batch: BatchRequest) -> dict:
    """Score up to 10,000 flows at once."""
    return _predict_records(batch.records)
