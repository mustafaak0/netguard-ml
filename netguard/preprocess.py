"""Feature preprocessing: one-hot encoding for categoricals, scaling for numerics."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from netguard.config import PreprocessConfig


class Preprocessor:
    """Fits a column transformer and keeps track of encoded feature names.

    Categorical columns are one-hot encoded (unseen categories are ignored at
    transform time); numeric columns are median-imputed and standard-scaled.
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()
        self.categorical_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.transformer: ColumnTransformer | None = None
        self.feature_names_: list[str] = []

    @property
    def n_features_(self) -> int:
        return len(self.feature_names_)

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        self.categorical_columns = [
            c for c in self._detect_categorical(df) if c in df.columns
        ]
        self.numeric_columns = [
            c for c in df.columns
            if c not in self.categorical_columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not self.numeric_columns and not self.categorical_columns:
            raise ValueError("no usable feature columns found in dataframe")

        max_categories = self.config.max_onehot_categories
        self.transformer = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        max_categories=max_categories,
                        min_frequency=0.005,
                    ),
                    self.categorical_columns,
                ),
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    self.numeric_columns,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.transformer.fit(df)
        self.feature_names_ = self.transformer.get_feature_names_out().tolist()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("preprocessor is not fitted; call fit() or load() first")
        expected = self.categorical_columns + self.numeric_columns
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"missing expected columns: {missing}")
        X = self.transformer.transform(df)
        return np.asarray(X, dtype=np.float64)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _detect_categorical(self, df: pd.DataFrame) -> list[str]:
        configured = self.config.categorical_columns
        if configured:
            return list(configured)
        detected = [
            c
            for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            or df[c].nunique() <= 20
        ]
        return detected

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.transformer is None:
            raise RuntimeError("cannot save an unfitted preprocessor")
        joblib.dump(
            {
                "config": self.config,
                "categorical_columns": self.categorical_columns,
                "numeric_columns": self.numeric_columns,
                "feature_names": self.feature_names_,
                "transformer": self.transformer,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Preprocessor:
        state = joblib.load(path)
        obj = cls(config=state["config"])
        obj.categorical_columns = state["categorical_columns"]
        obj.numeric_columns = state["numeric_columns"]
        obj.feature_names_ = state["feature_names"]
        obj.transformer = state["transformer"]
        return obj
