"""Unit tests for preprocessing, feature selection and controls."""
from __future__ import annotations

import numpy as np
import pytest

from src import config, feature_selection, preprocessing


def test_clean_frame_drops_duplicates_and_fixes_values(sample_frame):
    cleaned = preprocessing.clean_frame(sample_frame)
    assert len(cleaned) == 3  # one exact duplicate removed
    assert "ID_NO" not in cleaned.columns
    assert not np.isinf(cleaned["dur"]).any()
    assert cleaned["attack_cat"].str.strip().isin(["Normal", "DoS"]).all()


def test_split_xy_columns(sample_frame):
    cleaned = preprocessing.clean_frame(sample_frame)
    X, y, attack = preprocessing.split_xy(cleaned)
    assert list(X.columns) == config.ALL_FEATURE_COLS
    assert set(y.unique()) <= {0, 1}
    assert attack is not None


def test_preprocessor_output_is_finite(sample_frame):
    cleaned = preprocessing.clean_frame(sample_frame)
    X, y, _ = preprocessing.split_xy(cleaned)
    pre = preprocessing.build_preprocessor()
    matrix = pre.fit_transform(X, y)
    assert matrix.shape[0] == len(X)
    assert np.isfinite(matrix).all()


def test_feature_selector_slices_columns(sample_frame):
    cleaned = preprocessing.clean_frame(sample_frame)
    X, _, _ = preprocessing.split_xy(cleaned)
    pre = preprocessing.build_preprocessor()
    matrix = pre.fit_transform(X)
    selector = preprocessing.FeatureSelector([0, 2])
    out = selector.fit_transform(matrix)
    assert out.shape == (matrix.shape[0], 2)


def test_correlation_filter_removes_redundant(sample_frame):
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    matrix = np.column_stack([base, base * 2.0, rng.normal(size=500)])
    names = np.array(["a", "a_dup", "b"])
    kept, dropped = feature_selection._correlation_filter(matrix, names, 0.95)
    assert "b" in kept
    assert len(kept) == 2 and len(dropped) == 1


@pytest.mark.parametrize("objective", ["f1", "cost", "precision_at_recall"])
def test_best_threshold_objectives(objective):
    from src import evaluate

    y = np.array([0, 0, 1, 0, 1, 1, 0, 1] * 20)
    proba = np.clip(y * 0.6 + np.linspace(0.05, 0.45, len(y)), 0.01, 0.99)
    threshold, score = evaluate.best_threshold(y, proba, objective, min_recall=0.8)
    assert 0.0 <= threshold <= 1.0
    assert np.isfinite(score)
