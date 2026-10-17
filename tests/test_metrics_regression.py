"""Metric regression guard.

Fails CI if a retrained model drops below the expected quality bars.
"""
from __future__ import annotations

import json

import pytest

from src import config

pytestmark = pytest.mark.skipif(
    not config.METRICS_JSON.exists(), reason="metrics.json missing (train first)"
)

BINARY_FLOORS = {
    "accuracy": 0.82,
    "precision": 0.72,
    "recall": 0.85,
    "f1": 0.80,
    "roc_auc": 0.94,
    "mcc": 0.66,
}
MAX_FALSE_POSITIVE_RATE = 0.20
MULTICLASS_FLOORS = {"accuracy": 0.64, "macro_f1": 0.50}


@pytest.fixture(scope="module")
def binary_metrics() -> dict:
    return json.loads(config.METRICS_JSON.read_text())


@pytest.fixture(scope="module")
def multiclass_metrics() -> dict:
    if not config.MULTICLASS_METRICS_JSON.exists():
        pytest.skip("multiclass metrics missing")
    return json.loads(config.MULTICLASS_METRICS_JSON.read_text())


@pytest.mark.parametrize("model", ["RandomForest", "XGBoost"])
def test_binary_quality_floor(binary_metrics, model):
    scores = binary_metrics.get(model)
    if scores is None:
        pytest.skip(f"{model} not present")
    for metric, floor in BINARY_FLOORS.items():
        assert scores[metric] >= floor, f"{model}.{metric}={scores[metric]:.4f} < {floor}"
    assert scores["false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE, (
        f"{model} FPR={scores['false_positive_rate']:.4f} exceeds {MAX_FALSE_POSITIVE_RATE}"
    )


def test_best_model_is_saved(binary_metrics):
    meta = json.loads(config.METADATA_PATH.read_text())
    assert meta["best_model"] in binary_metrics
    assert config.MODEL_PATH.exists()
    assert config.MULTICLASS_MODEL_PATH.exists()


@pytest.mark.parametrize("model", ["RandomForest", "XGBoost"])
def test_multiclass_quality_floor(multiclass_metrics, model):
    scores = multiclass_metrics.get(model)
    if scores is None:
        pytest.skip(f"{model} not present")
    for metric, floor in MULTICLASS_FLOORS.items():
        assert scores[metric] >= floor, f"{model}.{metric}={scores[metric]:.4f} < {floor}"
