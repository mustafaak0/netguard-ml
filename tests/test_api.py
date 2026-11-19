"""FastAPI service tests (TestClient)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from netguard.data import generate_synthetic
from netguard.train import train


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    artifacts = tmp_path_factory.mktemp("artifacts")
    df = generate_synthetic(n_samples=1_500, attack_ratio=0.15, seed=11)

    import netguard.config as config_mod

    cfg = config_mod.NetGuardConfig()
    cfg.isolation_forest.n_estimators = 50
    cfg.autoencoder.encoder_dims = [16, 8]
    cfg.autoencoder.epochs = 3
    cfg.train.test_size = 0.25
    train(df, cfg, artifacts)

    import os

    os.environ["NETGUARD_MODEL_DIR"] = str(artifacts)
    from netguard.api import app

    with TestClient(app) as client:
        yield client
    os.environ.pop("NETGUARD_MODEL_DIR", None)


def _normal_flow():
    return {
        "duration": 30.0,
        "src_bytes": 1500.0,
        "dst_bytes": 8000.0,
        "wrong_fragment": 0,
        "urgent": 0,
        "count": 5,
        "srv_count": 4,
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "same_srv_rate": 0.95,
        "dst_host_count": 40,
        "dst_host_srv_count": 35,
        "dst_host_same_srv_rate": 0.9,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
    }


def _dos_flow():
    flow = _normal_flow()
    flow.update(
        src_bytes=400_000.0,
        dst_bytes=0.0,
        count=300,
        srv_count=290,
        serror_rate=1.0,
        same_srv_rate=0.02,
        flag="S0",
    )
    return flow


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["detectors"]) == {"isolation_forest", "autoencoder"}
    assert body["n_encoded_features"] > 0


def test_predict_single(client):
    resp = client.post("/predict", json={"features": _normal_flow()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    pred = body["predictions"][0]
    assert 0 <= pred["risk_score"] <= 100
    assert pred["level"] in {"normal", "suspicious", "critical"}


def test_predict_flags_dos(client):
    normal = client.post("/predict", json={"features": _normal_flow()}).json()
    attack = client.post("/predict", json={"features": _dos_flow()}).json()
    assert (
        attack["predictions"][0]["risk_score"]
        > normal["predictions"][0]["risk_score"]
    )
    assert attack["predictions"][0]["level"] in {"suspicious", "critical"}


def test_predict_batch(client):
    batch = {
        "records": [
            {"features": _normal_flow()},
            {"features": _dos_flow()},
        ]
    }
    resp = client.post("/predict/batch", json=batch)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_unknown_field_rejected(client):
    resp = client.post("/predict", json={"features": {"bogus": 1}})
    assert resp.status_code == 422
