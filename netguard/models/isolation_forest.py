"""Isolation Forest detector (scikit-learn backed)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from netguard.config import IsolationForestConfig
from netguard.models.base import Detector


class IsolationForestDetector(Detector):
    """Wraps :class:`sklearn.ensemble.IsolationForest`.

    ``decision_function`` returns higher values for *normal* samples, so the
    raw score is negated and min-max calibrated on the training scores to
    land in ``[0, 1]`` where 1 means most anomalous.
    """

    name = "isolation_forest"

    def __init__(self, config: IsolationForestConfig | None = None):
        self.config = config or IsolationForestConfig()
        self.model: IsolationForest | None = None
        self._cal_min: float = 0.0
        self._cal_max: float = 1.0

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X)
        raw = self._raw_scores(X)
        self._cal_min = float(raw.min())
        self._cal_max = float(raw.max())
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self._calibrate(self._raw_scores(X))

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("detector is not fitted")
        return -self.model.decision_function(X)

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        span = self._cal_max - self._cal_min
        if span <= 0:
            return np.zeros_like(raw)
        return np.clip((raw - self._cal_min) / span, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "config": self.config,
                "model": self.model,
                "cal_min": self._cal_min,
                "cal_max": self._cal_max,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> IsolationForestDetector:
        state = joblib.load(path)
        obj = cls(config=state["config"])
        obj.model = state["model"]
        obj._cal_min = state["cal_min"]
        obj._cal_max = state["cal_max"]
        return obj
