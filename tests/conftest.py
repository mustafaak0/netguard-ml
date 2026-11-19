"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netguard.config import NetGuardConfig
from netguard.data import generate_synthetic


@pytest.fixture(scope="session")
def small_config() -> NetGuardConfig:
    cfg = NetGuardConfig()
    cfg.isolation_forest.n_estimators = 50
    cfg.autoencoder.encoder_dims = [16, 8]
    cfg.autoencoder.epochs = 3
    cfg.train.test_size = 0.25
    return cfg


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return generate_synthetic(n_samples=2_000, attack_ratio=0.10, seed=7)


@pytest.fixture()
def numeric_matrix() -> np.ndarray:
    """Two well-separated Gaussian blobs (normal / anomalous)."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=(400, 6))
    anomalous = rng.normal(6.0, 1.0, size=(60, 6))
    return np.vstack([normal, anomalous])
