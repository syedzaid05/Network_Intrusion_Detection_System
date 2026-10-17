"""API contract tests (skipped when no trained model is present)."""
from __future__ import annotations

import pytest

from src import predict

pytestmark = pytest.mark.skipif(
    not predict.model_ready(), reason="requires a trained model (run scripts/run_pipeline.py)"
)


@pytest.fixture(scope="module")
def client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from api.main import app

    return fastapi_testclient.TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_ready"] is True


def test_model_info(client):
    data = client.get("/model-info").json()
    assert data["best_model"]
    assert data["decision_threshold"] is not None
    assert data["n_selected"] > 0


def test_example_and_predict(client):
    example = client.get("/example").json()
    resp = client.post("/predict", json=example)
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] in {"Normal", "Attack"}
    assert 0.0 <= body["attack_probability"] <= 1.0
    assert "attack_category" in body


def test_predict_batch(client):
    example = client.get("/example").json()
    resp = client.post("/predict/batch", json={"records": [example, example]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_invalid_input_rejected(client):
    resp = client.post("/predict", json={"dur": -5.0})
    assert resp.status_code == 422
