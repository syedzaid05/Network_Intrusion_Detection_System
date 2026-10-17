"""Correlation analysis + importance-based feature selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from . import config


def _correlation_filter(matrix: np.ndarray, names: np.ndarray, threshold: float):
    """Greedy removal of highly correlated (redundant) features."""
    frame = pd.DataFrame(matrix, columns=names)
    corr = frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Keep the feature with higher mean absolute correlation to the rest when tied.
    mean_corr = corr.abs().mean().sort_values(ascending=False)
    to_drop: set[str] = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > threshold].tolist()
        for partner in partners:
            if partner in to_drop:
                continue
            weaker = partner if mean_corr[partner] <= mean_corr[col] else col
            to_drop.add(weaker)
    kept = [c for c in names if c not in to_drop]
    return kept, sorted(to_drop)


def profile_features(matrix: np.ndarray, names: np.ndarray, top_n: int = 20) -> pd.DataFrame:
    """Return pairwise correlations for the strongest relationships (for reporting)."""
    corr = pd.DataFrame(matrix, columns=names).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .sort_values(ascending=False)
        .head(top_n)
        .rename("abs_corr")
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    return pairs


def select_features(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: config.Config = config.CONFIG,
):
    """Fit encoder on a sample, then drop redundant features and rank the rest.

    Returns a dict with:
      - ``selected_names``: transformed feature names chosen
      - ``selected_indices``: their column positions in the transformed matrix
      - ``importance``: mutual-information score per candidate feature
      - ``dropped_correlated``: features removed by the correlation filter
    """
    sample_n = min(cfg.importance_sample, len(X_train))
    rng = np.random.default_rng(cfg.random_state)
    idx = rng.choice(len(X_train), size=sample_n, replace=False)
    X_sample = X_train.iloc[idx]
    y_sample = y_train.iloc[idx]

    sampler = clone(preprocessor)
    matrix = sampler.fit_transform(X_sample, y_sample)
    names = np.asarray(sampler.get_feature_names_out())

    nonzero = np.flatnonzero(np.nanstd(np.asarray(matrix), axis=0) > 1e-12)
    matrix = np.asarray(matrix)[:, nonzero]
    names = names[nonzero]

    kept, dropped = _correlation_filter(matrix, names, cfg.correction_threshold)
    keep_mask = np.isin(names, kept)
    matrix_f = matrix[:, keep_mask]
    names_f = names[keep_mask]

    mi = mutual_info_classif(matrix_f, y_sample, random_state=cfg.random_state)
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    rf.fit(matrix_f, y_sample)

    importance = pd.DataFrame(
        {"feature": names_f, "mutual_info": mi, "rf_importance": rf.feature_importances_}
    ).sort_values("mutual_info", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)

    top = importance.head(cfg.top_k_features)
    selected_names = top["feature"].tolist()
    selected_indices = [int(i) for i, n in enumerate(names) if n in set(selected_names)]

    print(f"[features] {len(names)} -> {len(selected_names)} selected "
          f"({len(dropped)} dropped as redundant)")
    return {
        "selected_names": selected_names,
        "selected_indices": selected_indices,
        "all_names": names.tolist(),
        "importance": importance,
        "dropped_correlated": dropped,
        "sample_matrix": matrix,
        "sample_names": names,
    }
