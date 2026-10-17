"""Cleaning, encoding and scaling for the IDS features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from . import config


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ids/duplicates, normalise categories and repair numeric values."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if config.ATTACK_CAT_COL in df.columns:
        df[config.ATTACK_CAT_COL] = (
            df[config.ATTACK_CAT_COL].astype(str).str.strip().replace({"": "Normal"})
        )

    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns], errors="ignore")

    numeric = [c for c in config.NUMERIC_COLS if c in df.columns]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)

    for col in config.CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"": "unknown", "nan": "unknown"})

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"[prep] dropped {dropped} duplicate rows")

    missing = int(df[numeric].isna().sum().sum())
    if missing:
        print(f"[prep] filling {missing} missing numeric cells via median imputation")
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Return (features, binary label, attack category)."""
    y = df[config.TARGET_COL].astype(int)
    attack = df[config.ATTACK_CAT_COL] if config.ATTACK_CAT_COL in df.columns else None
    X = df[[c for c in config.ALL_FEATURE_COLS if c in df.columns]].copy()
    return X, y, attack


def build_preprocessor() -> ColumnTransformer:
    """Column-wise imputation + scaling (numeric) and one-hot encoding (categorical)."""
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=config.CONFIG.proto_min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, config.NUMERIC_COLS),
            ("cat", categorical_pipe, config.CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select a fixed subset of already-transformed columns by index."""

    def __init__(self, indices: list[int] | None = None):
        self.indices = indices

    def fit(self, X, y=None):
        if self.indices is None:
            self.indices = list(range(X.shape[1]))
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.indices]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(self.indices)
        return np.asarray(input_features)[self.indices]
