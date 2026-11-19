"""Ensemble risk scoring: combine detector scores into an explainable risk score."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RiskLevel(str, Enum):
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    CRITICAL = "critical"


@dataclass
class FeatureContribution:
    feature: str
    contribution: float  # 0..1, relative share of the anomaly signal


@dataclass
class PredictionResult:
    risk_score: int  # 0..100
    level: RiskLevel
    detector_scores: dict[str, float] = field(default_factory=dict)
    top_features: list[FeatureContribution] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "level": self.level.value,
            "detector_scores": self.detector_scores,
            "top_features": [
                {"feature": c.feature, "contribution": round(c.contribution, 4)}
                for c in self.top_features
            ],
        }


class RiskScorer:
    """Weighted ensemble over calibrated detector scores.

    Explanations come from per-feature reconstruction errors (autoencoder)
    or, when only the isolation forest is available, from absolute z-scores
    of the scaled features.
    """

    def __init__(
        self,
        weights: dict[str, float],
        critical_threshold: int = 70,
        suspicious_threshold: int = 50,
        top_k_features: int = 5,
    ):
        total = sum(weights.values()) or 1.0
        self.weights = {name: w / total for name, w in weights.items()}
        self.critical_threshold = critical_threshold
        self.suspicious_threshold = suspicious_threshold
        self.top_k_features = top_k_features

    def combine(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Combine ``{detector_name: calibrated scores}`` into risk scores in [0, 100]."""
        names = [n for n in self.weights if n in scores and scores[n] is not None]
        if not names:
            raise ValueError("no detector scores supplied")
        stacked = np.vstack([scores[n] for n in names])
        weights = np.array([self.weights[n] for n in names]).reshape(-1, 1)
        blended = (stacked * weights).sum(axis=0)
        return np.clip(blended * 100.0, 0.0, 100.0)

    def level(self, risk_score: float) -> RiskLevel:
        if risk_score >= self.critical_threshold:
            return RiskLevel.CRITICAL
        if risk_score >= self.suspicious_threshold:
            return RiskLevel.SUSPICIOUS
        return RiskLevel.NORMAL

    def explain(
        self,
        feature_errors: np.ndarray | None,
        feature_names: list[str],
    ) -> list[FeatureContribution]:
        """Rank features by their share of the anomaly signal for a single row.

        ``feature_errors`` is a 1-D array (one value per encoded feature).
        """
        if feature_errors is None or not feature_names:
            return []
        total = feature_errors.sum()
        if total <= 0:
            return []
        k = min(self.top_k_features, len(feature_names))
        top_idx = np.argsort(feature_errors)[::-1][:k]
        return [
            FeatureContribution(
                feature=feature_names[i],
                contribution=float(feature_errors[i] / total),
            )
            for i in top_idx
        ]
