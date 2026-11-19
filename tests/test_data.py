"""Tests for dataset loading, validation and the synthetic generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netguard.data import (
    ATTACK_KINDS,
    CATEGORICAL_COLUMNS,
    LABEL_COLUMN,
    NUMERIC_COLUMNS,
    generate_synthetic,
    load_dataset,
    split_features_labels,
    validate_schema,
)


def test_synthetic_schema():
    df = generate_synthetic(n_samples=500, attack_ratio=0.1, seed=1)
    expected = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [LABEL_COLUMN])
    assert set(df.columns) == expected
    validate_schema(df, require_label=True)


def test_synthetic_label_ratio():
    df = generate_synthetic(n_samples=1_000, attack_ratio=0.2, seed=2)
    ratio = (df[LABEL_COLUMN] != "normal").mean()
    assert ratio == pytest.approx(0.2, abs=0.01)


def test_synthetic_attacks_use_known_kinds():
    df = generate_synthetic(n_samples=2_000, attack_ratio=0.3, seed=3)
    kinds = set(df.loc[df[LABEL_COLUMN] != "normal", LABEL_COLUMN])
    assert kinds <= set(ATTACK_KINDS)
    assert kinds  # at least one kind present


def test_synthetic_reproducible():
    a = generate_synthetic(n_samples=200, attack_ratio=0.1, seed=9)
    b = generate_synthetic(n_samples=200, attack_ratio=0.1, seed=9)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_attack_features_shifted():
    """Attack src_bytes must dominate normal src_bytes for e.g. DoS."""
    df = generate_synthetic(n_samples=5_000, attack_ratio=0.2, seed=4)
    dos = df[df[LABEL_COLUMN] == "dos"]["src_bytes"]
    normal = df[df[LABEL_COLUMN] == "normal"]["src_bytes"]
    assert dos.median() > normal.median() * 10


@pytest.mark.parametrize(
    "bad_frame,match",
    [
        (pd.DataFrame(), "empty"),
        (pd.DataFrame({"a": [1, None]}), "missing values"),
        (pd.DataFrame({"a": [1], LABEL_COLUMN: ["normal"]}).rename(columns={}), "none"),
    ],
)
def test_validate_schema_rejects_bad_frames(bad_frame, match):
    if match == "none":
        validate_schema(bad_frame, require_label=True)  # should pass
    else:
        with pytest.raises(ValueError, match=match):
            validate_schema(bad_frame)


def test_load_dataset_roundtrip(tmp_path, sample_df):
    path = tmp_path / "traffic.csv"
    sample_df.to_csv(path, index=False)
    loaded = load_dataset(path, require_label=True)
    assert len(loaded) == len(sample_df)


def test_split_features_labels(sample_df):
    features, labels = split_features_labels(sample_df)
    assert LABEL_COLUMN not in features.columns
    assert labels is not None and set(labels.unique()) >= {"normal"}


def test_split_features_labels_without_label(sample_df):
    features, labels = split_features_labels(sample_df.drop(columns=[LABEL_COLUMN]))
    assert labels is None
    assert np.isfinite(features[NUMERIC_COLUMNS].to_numpy()).all()
