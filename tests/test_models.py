"""Tests for the individual detectors and the ensemble scorer."""

from __future__ import annotations

import numpy as np
import pytest

from netguard.ensemble import RiskLevel, RiskScorer
from netguard.models.autoencoder import AutoencoderDetector, is_torch_available
from netguard.models.isolation_forest import IsolationForestDetector

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def test_if_scores_bounded_and_discriminative(numeric_matrix):
    detector = IsolationForestDetector()
    detector.fit(numeric_matrix[:400])
    scores = detector.score_samples(numeric_matrix)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert scores[400:].mean() > scores[:400].mean()


def test_if_save_load_roundtrip(tmp_path, numeric_matrix):
    detector = IsolationForestDetector()
    detector.fit(numeric_matrix[:400])
    path = tmp_path / "if.joblib"
    detector.save(path)
    restored = IsolationForestDetector.load(path)
    np.testing.assert_allclose(
        detector.score_samples(numeric_matrix), restored.score_samples(numeric_matrix)
    )


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch not installed")
def test_ae_scores_bounded_and_discriminative(numeric_matrix):
    detector = AutoencoderDetector()
    detector.config.epochs = 10
    detector.fit(numeric_matrix[:400])
    scores = detector.score_samples(numeric_matrix)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert scores[400:].mean() > scores[:400].mean()


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch not installed")
def test_ae_save_load_roundtrip(tmp_path, numeric_matrix):
    detector = AutoencoderDetector()
    detector.fit(numeric_matrix[:400])
    path = tmp_path / "ae.pt"
    detector.save(path)
    restored = AutoencoderDetector.load(path)
    np.testing.assert_allclose(
        detector.score_samples(numeric_matrix),
        restored.score_samples(numeric_matrix),
        atol=1e-6,
    )


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch not installed")
def test_ae_feature_errors_shape(numeric_matrix):
    detector = AutoencoderDetector()
    detector.fit(numeric_matrix[:400])
    errors = detector.feature_errors(numeric_matrix)
    assert errors.shape == numeric_matrix.shape
    assert (errors[400:].mean() > errors[:400].mean())


def test_risk_scorer_combine_and_levels():
    scorer = RiskScorer(
        weights={"isolation_forest": 0.5, "autoencoder": 0.5},
        critical_threshold=70,
        suspicious_threshold=50,
    )
    scores = {
        "isolation_forest": np.array([0.10, 0.55, 0.90]),
        "autoencoder": np.array([0.30, 0.45, 0.80]),
    }
    risk = scorer.combine(scores)
    np.testing.assert_allclose(risk, [20.0, 50.0, 85.0])
    assert scorer.level(risk[0]) is RiskLevel.NORMAL
    assert scorer.level(risk[1]) is RiskLevel.SUSPICIOUS
    assert scorer.level(risk[2]) is RiskLevel.CRITICAL


def test_risk_scorer_single_detector():
    scorer = RiskScorer(weights={"isolation_forest": 1.0})
    risk = scorer.combine({"isolation_forest": np.array([0.6])})
    np.testing.assert_allclose(risk, [60.0])


def test_risk_scorer_explanation():
    scorer = RiskScorer(weights={"a": 1.0}, top_k_features=2)
    errors = np.array([0.1, 5.0, 2.5, 0.0])
    contribs = scorer.explain(errors, ["f0", "f1", "f2", "f3"])
    assert [c.feature for c in contribs] == ["f1", "f2"]
    assert contribs[0].contribution > contribs[1].contribution


def test_risk_scorer_explanation_empty():
    scorer = RiskScorer(weights={"a": 1.0})
    assert scorer.explain(None, ["f0"]) == []
