"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class PreprocessConfig:
    categorical_columns: list[str] = field(
        default_factory=lambda: ["protocol_type", "service", "flag"]
    )
    max_onehot_categories: int = 50


@dataclass
class IsolationForestConfig:
    n_estimators: int = 200
    contamination: float = 0.05


@dataclass
class AutoencoderConfig:
    encoder_dims: list[int] = field(default_factory=lambda: [64, 32])
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    validation_split: float = 0.1


@dataclass
class EnsembleConfig:
    isolation_forest_weight: float = 0.5
    autoencoder_weight: float = 0.5
    critical_threshold: int = 70
    suspicious_threshold: int = 50
    top_k_features: int = 5


@dataclass
class TrainConfig:
    test_size: float = 0.2


@dataclass
class NetGuardConfig:
    seed: int = 42
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if not 0.0 < self.train.test_size < 1.0:
            raise ValueError(f"train.test_size must be in (0, 1), got {self.train.test_size}")
        if not 0.0 < self.isolation_forest.contamination < 0.5:
            raise ValueError("isolation_forest.contamination must be in (0, 0.5)")
        weights = (
            self.ensemble.isolation_forest_weight,
            self.ensemble.autoencoder_weight,
        )
        if any(w < 0 for w in weights) or math.isclose(sum(weights), 0.0):
            raise ValueError("ensemble weights must be non-negative and not all zero")
        if self.ensemble.critical_threshold < self.ensemble.suspicious_threshold:
            raise ValueError("critical_threshold must be >= suspicious_threshold")
        if not self.autoencoder.encoder_dims:
            raise ValueError("autoencoder.encoder_dims must not be empty")
        latent = self.autoencoder.encoder_dims[-1]
        if latent < 1:
            raise ValueError("last encoder dim (latent dim) must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NetGuardConfig:
        cfg = cls()
        cfg.seed = data.get("seed", cfg.seed)
        for section in ("preprocess", "isolation_forest", "autoencoder", "ensemble", "train"):
            values = data.get(section) or {}
            section_obj = getattr(cfg, section)
            for key, value in values.items():
                if not hasattr(section_obj, key):
                    raise ValueError(f"unknown config key: {section}.{key}")
                setattr(section_obj, key, value)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> NetGuardConfig:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)
