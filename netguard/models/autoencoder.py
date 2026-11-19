"""Autoencoder detector (PyTorch backed, optional dependency).

Reconstruction error is the anomaly signal: flows the bottleneck cannot
reconstruct are considered anomalous. Errors are min-max calibrated on the
training set to ``[0, 1]``. Per-feature squared errors are exposed so the
ensemble can explain *which* features drove a risk score.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from netguard.config import AutoencoderConfig
from netguard.models.base import Detector

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def is_torch_available() -> bool:
    return _TORCH_AVAILABLE


if _TORCH_AVAILABLE:
    import torch
    from torch import nn

    class _Autoencoder(nn.Module):
        def __init__(self, n_features: int, encoder_dims: list[int]):
            super().__init__()
            dims = [n_features, *encoder_dims]
            encoder_layers: list[nn.Module] = []
            for i in range(len(dims) - 2):
                encoder_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            encoder_layers.append(nn.Linear(dims[-2], dims[-1]))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_layers: list[nn.Module] = []
            rev = list(reversed(dims))
            for i in range(len(rev) - 2):
                decoder_layers += [nn.Linear(rev[i], rev[i + 1]), nn.ReLU()]
            decoder_layers.append(nn.Linear(rev[-2], rev[-1]))
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))


class AutoencoderDetector(Detector):
    """Dense autoencoder; scores are calibrated reconstruction errors."""

    name = "autoencoder"

    def __init__(self, config: AutoencoderConfig | None = None):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for the autoencoder detector. "
                "Install it with: pip install torch --index-url "
                "https://download.pytorch.org/whl/cpu"
            )
        self.config = config or AutoencoderConfig()
        self.model: _Autoencoder | None = None
        self._cal_min: float = 0.0
        self._cal_max: float = 1.0

    def fit(self, X: np.ndarray) -> AutoencoderDetector:
        cfg = self.config
        rng = torch.Generator().manual_seed(42)
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        n = X_tensor.shape[0]
        n_val = max(1, int(n * cfg.validation_split)) if n > 10 else 0

        perm = torch.randperm(n, generator=rng)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        X_train = X_tensor[train_idx] if n_val else X_tensor

        self.model = _Autoencoder(X.shape[1], list(cfg.encoder_dims))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=3
        )

        self.model.train()
        batch_size = min(cfg.batch_size, len(X_train))
        for _epoch in range(cfg.epochs):
            shuffle = torch.randperm(len(X_train), generator=rng)
            for start in range(0, len(X_train), batch_size):
                batch = X_train[shuffle[start : start + batch_size]]
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.model(batch), batch)
                loss.backward()
                optimizer.step()
            if n_val:
                self.model.eval()
                with torch.no_grad():
                    X_val = X_tensor[val_idx]
                    val_loss = nn.functional.mse_loss(self.model(X_val), X_val)
                scheduler.step(val_loss)
                self.model.train()

        # calibrate on training reconstruction errors
        self._cal_min, self._cal_max = self._error_range(X_tensor)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        errors = self.feature_errors(X)
        return self._calibrate(errors.mean(axis=1))

    def feature_errors(self, X: np.ndarray) -> np.ndarray:
        """Per-feature squared reconstruction errors, shape ``(n, d)``."""
        if self.model is None:
            raise RuntimeError("detector is not fitted")
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32)
            recon = self.model(x)
        return ((x - recon) ** 2).numpy()

    def _error_range(self, X_tensor: torch.Tensor) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            row_errors = ((X_tensor - recon) ** 2).mean(dim=1)
        return float(row_errors.min()), float(row_errors.max())

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        span = self._cal_max - self._cal_min
        if span <= 0:
            return np.zeros_like(raw)
        return np.clip((raw - self._cal_min) / span, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted detector")
        payload = {
            "state_dict": self.model.state_dict(),
            "n_features": self.model.encoder[0].in_features,
            "encoder_dims": list(self.config.encoder_dims),
            "cal_min": self._cal_min,
            "cal_max": self._cal_max,
            "config": self.config,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> AutoencoderDetector:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to load an autoencoder model. "
                "Install it with: pip install torch --index-url "
                "https://download.pytorch.org/whl/cpu"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(config=payload["config"])
        obj.model = _Autoencoder(payload["n_features"], payload["encoder_dims"])
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.eval()
        obj._cal_min = payload["cal_min"]
        obj._cal_max = payload["cal_max"]
        return obj
