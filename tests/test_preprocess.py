"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from netguard.config import PreprocessConfig
from netguard.data import CATEGORICAL_COLUMNS
from netguard.preprocess import Preprocessor


def test_fit_transform_shapes(sample_df):
    features = sample_df.drop(columns=["label"])
    pre = Preprocessor(PreprocessConfig(categorical_columns=CATEGORICAL_COLUMNS))
    X = pre.fit_transform(features)
    assert X.shape[0] == len(features)
    assert X.shape[1] == pre.n_features_
    assert pre.numeric_columns and pre.categorical_columns
    # numeric columns are standardized
    num_idx = [pre.feature_names_.index(c) for c in pre.numeric_columns]
    means = X[:, num_idx].mean(axis=0)
    assert np.allclose(means, 0.0, atol=0.1)


def test_unseen_categories_ignored(sample_df):
    features = sample_df.drop(columns=["label"])
    pre = Preprocessor(PreprocessConfig(categorical_columns=CATEGORICAL_COLUMNS))
    pre.fit(features)
    novel = features.head(3).copy()
    novel.loc[novel.index[0], "service"] = "brand-new-service"
    X = pre.transform(novel)
    assert X.shape == (3, pre.n_features_)
    assert np.isfinite(X).all()


def test_missing_column_raises(sample_df):
    features = sample_df.drop(columns=["label"])
    pre = Preprocessor(PreprocessConfig(categorical_columns=CATEGORICAL_COLUMNS))
    pre.fit(features)
    with pytest.raises(ValueError, match="missing expected columns"):
        pre.transform(features.drop(columns=["src_bytes"]))


def test_save_load_roundtrip(tmp_path, sample_df):
    features = sample_df.drop(columns=["label"])
    pre = Preprocessor(PreprocessConfig(categorical_columns=CATEGORICAL_COLUMNS))
    X1 = pre.fit_transform(features)
    path = tmp_path / "pre.joblib"
    pre.save(path)
    restored = Preprocessor.load(path)
    X2 = restored.transform(features)
    np.testing.assert_allclose(X1, X2)
    assert restored.feature_names_ == pre.feature_names_


def test_label_column_excluded_when_present(sample_df):
    pre = Preprocessor(PreprocessConfig(categorical_columns=CATEGORICAL_COLUMNS))
    pre.fit_transform(sample_df)
    assert "label" not in pre.feature_names_
