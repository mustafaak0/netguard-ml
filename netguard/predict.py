"""Inference: load artifacts and score new flows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from netguard.data import LABEL_COLUMN
from netguard.ensemble import PredictionResult, RiskScorer
from netguard.models.autoencoder import AutoencoderDetector, is_torch_available
from netguard.models.isolation_forest import IsolationForestDetector
from netguard.preprocess import Preprocessor
from netguard.train import AE_FILE, IF_FILE, METADATA_FILE, PREPROCESSOR_FILE


class Predictor:
    """Loads a trained model directory and turns feature rows into risk reports."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        if_detector: IsolationForestDetector,
        ae_detector: AutoencoderDetector | None,
        scorer: RiskScorer,
        metadata: dict,
    ):
        self.preprocessor = preprocessor
        self.if_detector = if_detector
        self.ae_detector = ae_detector
        self.scorer = scorer
        self.metadata = metadata

    @classmethod
    def load(cls, model_dir: str | Path) -> Predictor:
        model_dir = Path(model_dir)
        preprocessor = Preprocessor.load(model_dir / PREPROCESSOR_FILE)
        if_detector = IsolationForestDetector.load(model_dir / IF_FILE)
        ae_path = model_dir / AE_FILE
        ae_detector = AutoencoderDetector.load(ae_path) if ae_path.exists() else None
        if ae_detector is None and not is_torch_available():
            pass  # ensemble falls back to the isolation forest alone

        with open(model_dir / METADATA_FILE, encoding="utf-8") as fh:
            metadata = json.load(fh)
        scorer = RiskScorer(
            weights=metadata["ensemble_weights"],
            critical_threshold=metadata["config"]["ensemble"]["critical_threshold"],
            suspicious_threshold=metadata["config"]["ensemble"]["suspicious_threshold"],
            top_k_features=metadata["config"]["ensemble"]["top_k_features"],
        )
        return cls(preprocessor, if_detector, ae_detector, scorer, metadata)

    @property
    def detectors(self) -> list[str]:
        return ["isolation_forest"] + (["autoencoder"] if self.ae_detector else [])

    def predict(self, df: pd.DataFrame) -> list[PredictionResult]:
        features = df.drop(columns=[LABEL_COLUMN], errors="ignore")
        X = self.preprocessor.transform(features)

        scores = {"isolation_forest": self.if_detector.score_samples(X)}
        row_errors = None
        if self.ae_detector is not None:
            feature_errors = self.ae_detector.feature_errors(X)
            span = max(self.ae_detector._cal_max - self.ae_detector._cal_min, 1e-12)
            scores["autoencoder"] = np.clip(
                (feature_errors.mean(axis=1) - self.ae_detector._cal_min) / span, 0.0, 1.0
            )
            row_errors = feature_errors

        risk = self.scorer.combine(scores)
        results: list[PredictionResult] = []
        for i in range(len(df)):
            top_features = self.scorer.explain(
                row_errors[i] if row_errors is not None else None,
                self.preprocessor.feature_names_,
            )
            results.append(
                PredictionResult(
                    risk_score=int(round(float(risk[i]))),
                    level=self.scorer.level(risk[i]),
                    detector_scores={k: round(float(v[i]), 4) for k, v in scores.items()},
                    top_features=top_features,
                )
            )
        return results
