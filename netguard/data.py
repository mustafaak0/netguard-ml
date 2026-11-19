"""Dataset loading, validation and synthetic traffic generation.

The expected schema is KDD-style: a mix of numeric traffic features plus
categorical columns (``protocol_type``, ``service``, ``flag``) and an
optional ``label`` column. Real datasets such as NSL-KDD or CICIDS2017
(exported to CSV) can be used as-is when their columns match; anything
else can be adapted via :class:`netguard.config.PreprocessConfig`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLUMN = "label"
NORMAL_LABEL = "normal"

NUMERIC_COLUMNS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "wrong_fragment",
    "urgent",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "same_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "https", "ftp", "ssh", "dns", "smtp", "other"]
FLAGS = ["SF", "REJ", "S0", "RSTR", "SH"]

ATTACK_KINDS = ["dos", "probe", "r2l", "u2r"]


def validate_schema(df: pd.DataFrame, require_label: bool = False) -> None:
    """Raise ``ValueError`` when the frame cannot be used for training/scoring."""
    if df.empty:
        raise ValueError("dataset is empty")
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"duplicate columns: {dupes}")
    if require_label and LABEL_COLUMN not in df.columns:
        raise ValueError(f"missing required column: {LABEL_COLUMN!r}")
    if df.drop(columns=[LABEL_COLUMN], errors="ignore").isna().any().any():
        raise ValueError("dataset contains missing values; impute or drop them first")


def load_dataset(path: str | Path, require_label: bool = False) -> pd.DataFrame:
    """Load a CSV dataset and validate its schema."""
    df = pd.read_csv(path)
    validate_schema(df, require_label=require_label)
    return df


def split_features_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return ``(features, labels)``; labels are ``None`` when no label column exists."""
    if LABEL_COLUMN in df.columns:
        return df.drop(columns=[LABEL_COLUMN]), df[LABEL_COLUMN]
    return df, None


def generate_synthetic(
    n_samples: int = 20_000,
    attack_ratio: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic KDD-style traffic dataset.

    Normal flows are drawn from per-service byte/rate profiles that overlap
    heavily with each other; each attack kind (dos, probe, r2l, u2r) then
    shifts a characteristic subset of features far outside those profiles.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not 0.0 <= attack_ratio <= 0.9:
        raise ValueError("attack_ratio must be in [0, 0.9]")

    rng = np.random.default_rng(seed)
    n_attacks = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attacks

    # ---- normal traffic -------------------------------------------------
    service_probs = [0.4, 0.25, 0.08, 0.07, 0.1, 0.05, 0.05]
    service_idx = rng.choice(len(SERVICES), size=n_normal, p=service_probs)
    protocol_by_service = {
        "http": "tcp", "https": "tcp", "ftp": "tcp", "ssh": "tcp",
        "smtp": "tcp", "dns": "udp", "other": "tcp",
    }
    # (log-mean, log-std) of src/dst bytes per service
    byte_profile = {
        "http": (6.0, 1.0, 6.5, 1.0),
        "https": (7.0, 1.0, 7.0, 1.2),
        "ftp": (6.5, 1.5, 6.0, 1.0),
        "ssh": (5.0, 0.8, 5.0, 0.8),
        "dns": (3.0, 0.5, 4.0, 0.5),
        "smtp": (7.5, 1.2, 6.5, 1.0),
        "other": (5.5, 1.5, 5.5, 1.5),
    }
    services = np.array(SERVICES)[service_idx]
    src = np.empty(n_normal)
    dst = np.empty(n_normal)
    for svc in SERVICES:
        mask = services == svc
        mu_s, sd_s, mu_d, sd_d = byte_profile[svc]
        src[mask] = rng.lognormal(mu_s, sd_s, mask.sum())
        dst[mask] = rng.lognormal(mu_d, sd_d, mask.sum())

    normal = pd.DataFrame(
        {
            "duration": rng.lognormal(3.5, 1.2, n_normal),
            "src_bytes": src,
            "dst_bytes": dst,
            "wrong_fragment": rng.choice([0, 1], size=n_normal, p=[0.98, 0.02]),
            "urgent": np.zeros(n_normal, dtype=int),
            "count": rng.poisson(5, n_normal),
            "srv_count": rng.poisson(4, n_normal),
            "serror_rate": rng.beta(1, 40, n_normal),
            "srv_serror_rate": rng.beta(1, 40, n_normal),
            "same_srv_rate": rng.beta(20, 2, n_normal),
            "dst_host_count": rng.poisson(50, n_normal),
            "dst_host_srv_count": rng.poisson(40, n_normal),
            "dst_host_same_srv_rate": rng.beta(20, 2, n_normal),
            "protocol_type": [protocol_by_service[s] for s in services],
            "service": services,
            "flag": rng.choice(FLAGS, size=n_normal, p=[0.9, 0.05, 0.03, 0.01, 0.01]),
            "label": NORMAL_LABEL,
        }
    )

    # ---- attacks ---------------------------------------------------------
    frames = [normal]
    if n_attacks > 0:
        frames.append(_generate_attacks(n_attacks, rng))
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _generate_attacks(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate ``n`` attack flows spread across the four attack kinds."""
    kinds = rng.choice(ATTACK_KINDS, size=n)
    svc = rng.choice(SERVICES, size=n)
    df = pd.DataFrame(
        {
            "duration": rng.lognormal(4.0, 1.5, n),
            "src_bytes": rng.lognormal(6.0, 1.5, n),
            "dst_bytes": rng.lognormal(6.0, 1.5, n),
            "wrong_fragment": rng.choice([0, 1, 3], size=n, p=[0.4, 0.3, 0.3]),
            "urgent": rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]),
            "count": rng.poisson(80, n),
            "srv_count": rng.poisson(70, n),
            "serror_rate": rng.beta(30, 2, n),
            "srv_serror_rate": rng.beta(30, 2, n),
            "same_srv_rate": rng.beta(2, 20, n),
            "dst_host_count": rng.poisson(200, n),
            "dst_host_srv_count": rng.poisson(180, n),
            "dst_host_same_srv_rate": rng.beta(2, 20, n),
            "protocol_type": rng.choice(PROTOCOLS, size=n, p=[0.7, 0.2, 0.1]),
            "service": svc,
            "flag": rng.choice(FLAGS, size=n, p=[0.1, 0.35, 0.4, 0.1, 0.05]),
            "label": kinds,
        }
    )

    # Each attack kind distorts a characteristic feature subset.
    dos = kinds == "dos"
    df.loc[dos, "src_bytes"] = rng.lognormal(12.5, 0.6, dos.sum())
    df.loc[dos, "duration"] = rng.lognormal(1.5, 0.5, dos.sum())

    probe = kinds == "probe"
    df.loc[probe, "count"] = rng.poisson(300, probe.sum())
    df.loc[probe, "srv_count"] = rng.poisson(280, probe.sum())
    df.loc[probe, "serror_rate"] = rng.uniform(0.9, 1.0, probe.sum())

    r2l = kinds == "r2l"
    df.loc[r2l, "dst_bytes"] = rng.lognormal(3.0, 0.4, r2l.sum())
    df.loc[r2l, "service"] = rng.choice(["ftp", "ssh", "other"], size=r2l.sum())
    df.loc[r2l, "flag"] = "S0"

    u2r = kinds == "u2r"
    df.loc[u2r, "urgent"] = rng.choice([2, 3, 4], size=u2r.sum())
    df.loc[u2r, "duration"] = rng.lognormal(0.5, 0.3, u2r.sum())
    df.loc[u2r, "service"] = "ssh"

    return df
