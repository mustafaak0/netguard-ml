"""Common interface for all anomaly detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    """A detector maps feature vectors to anomaly scores in ``[0, 1]``.

    Higher scores mean *more anomalous*. Implementations must calibrate
    their raw scores onto ``[0, 1]`` during :meth:`fit` so the ensemble can
    combine them with simple weighted averaging.
    """

    name: str = "detector"

    @abstractmethod
    def fit(self, X: np.ndarray) -> Detector:
        """Fit on (mostly normal) training features and calibrate scores."""

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly scores in ``[0, 1]``."""
