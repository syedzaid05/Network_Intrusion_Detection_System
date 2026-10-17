"""FastAPI prediction service for the Network Intrusion Detection System.

Run with:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.schemas import (  # noqa: E402
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    PcapSummary,
    PredictionResponse,
    TrafficRecord,
)
from src import pcap_extractor, predict  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models so the first request is not slow."""
    if predict.model_ready():
        predict.load_pipeline()
        predict.load_multiclass_pipeline()
        print("[api] models preloaded")
    yield


app = FastAPI(
    title="Network Intrusion Detection API",
    description="Classify network flows as normal or malicious using a trained UNSW-NB15 model.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    ready = predict.model_ready()
    return HealthResponse(status="ok" if ready else "model_missing", model_ready=ready)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["system"])
def model_info() -> ModelInfoResponse:
    meta = predict.load_metadata()
    mc_meta = predict.load_multiclass_metadata()
    if not meta:
        return ModelInfoResponse(
            model_ready=predict.model_ready(), multiclass_ready=predict.multiclass_ready()
        )
    return ModelInfoResponse(
        model_ready=predict.model_ready(),
        best_model=meta.get("best_model"),
        trained_at=meta.get("trained_at"),
        dataset=meta.get("dataset"),
        n_features_input=meta.get("n_features_input"),
        n_selected=meta.get("n_selected"),
        decision_threshold=meta.get("decision_threshold"),
        multiclass_ready=predict.multiclass_ready(),
        attack_categories=mc_meta.get("classes", []),
        selected_features=meta.get("selected_features", []),
        metrics=meta.get("metrics", {}),
    )


@app.get("/example", response_model=TrafficRecord, tags=["prediction"])
def example() -> TrafficRecord:
    return TrafficRecord(**predict.example_record())


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_one(record: TrafficRecord) -> PredictionResponse:
    try:
        result = predict.predict_records([record.model_dump()])[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(payload: BatchRequest) -> BatchResponse:
    records = [r.model_dump() for r in payload.records]
    try:
        results = predict.predict_records(records)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    predictions = [PredictionResponse(**r) for r in results]
    return BatchResponse(
        count=len(predictions),
        attacks=sum(p.label for p in predictions),
        predictions=predictions,
    )


@app.post("/predict/pcap", response_model=PcapSummary, tags=["prediction"])
async def predict_pcap(file: UploadFile = File(...), max_flows: int = 5000) -> PcapSummary:
    """Extract flows from an uploaded PCAP and score them."""
    if not predict.model_ready():
        raise HTTPException(status_code=503, detail="No trained model available.")
    suffix = Path(file.filename or "upload.pcap").suffix or ".pcap"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        df = pcap_extractor.pcap_to_dataframe(tmp_path, max_flows=max_flows)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse PCAP: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if df.empty:
        raise HTTPException(status_code=422, detail="No IP flows found in the capture.")

    scored = predict.predict_dataframe(df)
    attacks = scored[scored["verdict"] == "Attack"]
    top = (
        attacks.sort_values("attack_probability", ascending=False)
        .head(10)[["proto", "service", "state", "attack_category", "attack_probability"]]
        .to_dict("records")
    )
    return PcapSummary(
        pcap=file.filename or "upload.pcap",
        flows=len(scored),
        attacks=len(attacks),
        attack_rate=round(len(attacks) / len(scored), 6),
        severity_counts=scored["severity"].value_counts().to_dict(),
        category_counts=scored["attack_category"].value_counts().to_dict(),
        top_flows=top,
    )
