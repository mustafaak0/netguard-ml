"""End-to-end training and inference tests."""

from __future__ import annotations

import json

import pytest

from netguard.data import LABEL_COLUMN, split_features_labels
from netguard.ensemble import RiskLevel
from netguard.predict import Predictor
from netguard.train import train


def _assert_reasonable(metadata, expect_metrics: bool):
    assert metadata["n_train_samples"] > 0
    assert metadata["detectors"] == ["isolation_forest", "autoencoder"]
    if expect_metrics:
        assert metadata["metrics"]["roc_auc"]["ensemble"] > 0.9
        assert metadata["metrics"]["attack_recall_at_critical"] > 0.6


def test_train_and_predict_labeled(tmp_path, sample_df, small_config):
    artifacts = tmp_path / "artifacts"
    metadata = train(sample_df, small_config, artifacts)
    _assert_reasonable(metadata, expect_metrics=True)
    assert (artifacts / "autoencoder.pt").exists()

    predictor = Predictor.load(artifacts)
    features, labels = split_features_labels(sample_df.tail(200))
    results = predictor.predict(features)

    assert len(results) == len(features)
    for r in results:
        assert 0 <= r.risk_score <= 100
        assert isinstance(r.level, RiskLevel)
        assert set(r.detector_scores) == {"isolation_forest", "autoencoder"}

    is_attack = (labels != "normal").to_numpy()
    attack_risk = max(
        (
            r.risk_score
            for r, a in zip(results, is_attack, strict=True)
            if a and r.top_features
        ),
        default=0,
    )
    normal_mask = ~is_attack
    avg_normal = (
        sum(r.risk_score for r, nm in zip(results, normal_mask, strict=True) if nm)
        / normal_mask.sum()
    )
    assert attack_risk > avg_normal

    explained = next(r for r in results if r.top_features)
    assert all(c.feature for c in explained.top_features)


def test_train_unlabeled(tmp_path, sample_df, small_config):
    unlabeled = sample_df.drop(columns=[LABEL_COLUMN])
    metadata = train(unlabeled, small_config, tmp_path / "artifacts")
    assert metadata["labeled"] is False
    assert metadata["metrics"] == {}
    predictor = Predictor.load(tmp_path / "artifacts")
    results = predictor.predict(unlabeled.head(5))
    assert len(results) == 5


def test_metadata_json_serializable(tmp_path, sample_df, small_config):
    artifacts = tmp_path / "artifacts"
    train(sample_df.head(300), small_config, artifacts)
    payload = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    assert payload["config"]["ensemble"]["critical_threshold"] == 70


def test_predict_rejects_missing_columns(tmp_path, sample_df, small_config):
    artifacts = tmp_path / "artifacts"
    train(sample_df.head(300), small_config, artifacts)
    predictor = Predictor.load(artifacts)
    with pytest.raises(ValueError, match="missing expected columns"):
        predictor.predict(sample_df.head(3).drop(columns=["src_bytes"]))
