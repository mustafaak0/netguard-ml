"""Training pipeline: preprocess -> fit detectors -> calibrate -> persist artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from netguard.config import NetGuardConfig
from netguard.data import NORMAL_LABEL, split_features_labels, validate_schema
from netguard.models.autoencoder import AutoencoderDetector, is_torch_available
from netguard.models.isolation_forest import IsolationForestDetector
from netguard.preprocess import Preprocessor

PREPROCESSOR_FILE = "preprocessor.joblib"
IF_FILE = "isolation_forest.joblib"
AE_FILE = "autoencoder.pt"
METADATA_FILE = "metadata.json"


def train(
    dataset: pd.DataFrame,
    config: NetGuardConfig,
    output_dir: str | Path,
) -> dict:
    """Train the ensemble on ``dataset`` and save all artifacts to ``output_dir``.

    Returns the metadata dict that is also written to ``metadata.json``.
    """
    config.validate()
    validate_schema(dataset)
    features, labels = split_features_labels(dataset)

    n = len(features)
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * config.train.test_size))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    # Stratify the holdout by label when labels exist so metrics stay honest.
    if labels is not None:
        normal_mask = (labels.iloc[train_idx] == NORMAL_LABEL).to_numpy()
        train_idx = train_idx[normal_mask]  # fit only on mostly-normal traffic
    X_train_df = features.iloc[train_idx]
    X_test_df = features.iloc[test_idx]
    y_test = (
        (labels.iloc[test_idx] != NORMAL_LABEL).astype(int).to_numpy()
        if labels is not None
        else None
    )

    started = time.perf_counter()
    preprocessor = Preprocessor(config.preprocess)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    if_detector = IsolationForestDetector(config.isolation_forest)
    if_detector.fit(X_train)
    if_scores = if_detector.score_samples(X_test)

    ae_detector = None
    ae_scores = None
    if is_torch_available():
        ae_detector = AutoencoderDetector(config.autoencoder)
        ae_detector.fit(X_train)
        ae_scores = ae_detector.score_samples(X_test)

    weights = {"isolation_forest": config.ensemble.isolation_forest_weight}
    if ae_detector is not None:
        weights["autoencoder"] = config.ensemble.autoencoder_weight
    from netguard.ensemble import RiskScorer  # local import avoids a cycle at import time

    scorer = RiskScorer(
        weights=weights,
        critical_threshold=config.ensemble.critical_threshold,
        suspicious_threshold=config.ensemble.suspicious_threshold,
        top_k_features=config.ensemble.top_k_features,
    )
    risk = scorer.combine({"isolation_forest": if_scores, "autoencoder": ae_scores})

    metrics: dict = {}
    if y_test is not None and 0 < y_test.sum() < len(y_test):
        metrics["roc_auc"] = {
            "isolation_forest": round(float(roc_auc_score(y_test, if_scores)), 4),
            "ensemble": round(float(roc_auc_score(y_test, risk / 100.0)), 4),
        }
        if ae_scores is not None:
            metrics["roc_auc"]["autoencoder"] = round(float(roc_auc_score(y_test, ae_scores)), 4)
        metrics["attack_recall_at_critical"] = round(
            float((risk[y_test == 1] >= config.ensemble.critical_threshold).mean()), 4
        )
        metrics["false_alarm_rate_at_critical"] = round(
            float((risk[y_test == 0] >= config.ensemble.critical_threshold).mean()), 4
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preprocessor.save(output / PREPROCESSOR_FILE)
    if_detector.save(output / IF_FILE)
    if ae_detector is not None:
        ae_detector.save(output / AE_FILE)

    metadata = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training_seconds": round(time.perf_counter() - started, 2),
        "n_train_samples": int(len(X_train)),
        "n_holdout_samples": int(len(X_test)),
        "n_encoded_features": int(X_train.shape[1]),
        "detectors": list(weights.keys()),
        "ensemble_weights": weights,
        "labeled": labels is not None,
        "config": config.to_dict(),
        "metrics": metrics,
    }
    with open(output / METADATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


def train_from_csv(
    csv_path: str | Path,
    config: NetGuardConfig,
    output_dir: str | Path,
) -> dict:
    from netguard.data import load_dataset

    return train(load_dataset(csv_path), config, output_dir)
