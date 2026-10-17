"""Shared inference helpers used by the API and the dashboard."""
from __future__ import annotations

import functools
import json
from typing import Any

import joblib
import pandas as pd

from . import config

SUSPICIOUS_THRESHOLD = 0.50
MALICIOUS_THRESHOLD = 0.80


def model_ready() -> bool:
    return config.MODEL_PATH.exists()


def multiclass_ready() -> bool:
    return config.MULTICLASS_MODEL_PATH.exists()


@functools.lru_cache(maxsize=1)
def load_pipeline():
    if not model_ready():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}. Run: python scripts/run_pipeline.py"
        )
    return joblib.load(config.MODEL_PATH)


@functools.lru_cache(maxsize=1)
def load_multiclass_pipeline():
    if not multiclass_ready():
        return None
    return joblib.load(config.MULTICLASS_MODEL_PATH)


@functools.lru_cache(maxsize=1)
def load_metadata() -> dict[str, Any]:
    if config.METADATA_PATH.exists():
        return json.loads(config.METADATA_PATH.read_text())
    return {}


@functools.lru_cache(maxsize=1)
def load_multiclass_metadata() -> dict[str, Any]:
    if config.MULTICLASS_METADATA_PATH.exists():
        return json.loads(config.MULTICLASS_METADATA_PATH.read_text())
    return {}


def decision_threshold() -> float:
    return float(load_metadata().get("decision_threshold", SUSPICIOUS_THRESHOLD))


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in config.ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return df[config.ALL_FEATURE_COLS].copy()


def severity(probability: float, threshold: float | None = None) -> str:
    threshold = SUSPICIOUS_THRESHOLD if threshold is None else threshold
    if probability >= max(MALICIOUS_THRESHOLD, threshold):
        return "Malicious"
    if probability >= threshold:
        return "Suspicious"
    return "Normal"


def predict_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict raw feature dicts, returning per-row verdict + attack category."""
    df = _normalise(pd.DataFrame(records))
    pipeline = load_pipeline()
    threshold = decision_threshold()
    proba = pipeline.predict_proba(df)[:, 1]
    preds = (proba >= threshold).astype(int)

    mc_pipeline = load_multiclass_pipeline()
    categories: list[str | None]
    if mc_pipeline is not None:
        classes = load_multiclass_metadata().get("classes", [])
        encoded = mc_pipeline.predict(df)
        categories = [classes[i] if classes and i < len(classes) else str(i) for i in encoded]
    else:
        categories = [None] * len(df)

    results = []
    for p, pr, cat in zip(preds, proba, categories):
        category = cat if (p == 1 and cat) else ("Normal" if p == 0 else (cat or "Unknown"))
        results.append(
            {
                "label": int(p),
                "verdict": "Attack" if p == 1 else "Normal",
                "severity": severity(float(pr), threshold),
                "attack_probability": round(float(pr), 6),
                "confidence": round(float(max(pr, 1 - pr)), 6),
                "attack_category": category,
            }
        )
    return results


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return an enriched copy of ``df`` with prediction columns appended."""
    verdicts = predict_records(df.to_dict("records"))
    out = df.reset_index(drop=True).copy()
    out["prediction"] = [v["label"] for v in verdicts]
    out["verdict"] = [v["verdict"] for v in verdicts]
    out["severity"] = [v["severity"] for v in verdicts]
    out["attack_category"] = [v["attack_category"] for v in verdicts]
    out["attack_probability"] = [v["attack_probability"] for v in verdicts]
    return out


def example_record() -> dict[str, Any]:
    """A representative normal-traffic row (for docs / dashboard defaults)."""
    sample = {
        "dur": 0.12, "spkts": 6, "dpkts": 4, "sbytes": 258, "dbytes": 172,
        "rate": 74.09, "sttl": 252, "dttl": 254, "sload": 14158.9, "dload": 8495.4,
        "sloss": 0, "dloss": 0, "sinpkt": 24.3, "dinpkt": 8.4, "sjit": 30.2,
        "djit": 11.8, "swin": 255, "stcpb": 621772692, "dtcpb": 2202533631,
        "dwin": 255, "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 43,
        "dmean": 43, "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 1,
        "ct_state_ttl": 0, "ct_dst_ltm": 1, "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1, "is_ftp_login": 0,
        "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1, "ct_srv_dst": 1,
        "is_sm_ips_ports": 0, "proto": "tcp", "service": "-", "state": "FIN",
    }
    return {k: sample.get(k, 0) for k in config.ALL_FEATURE_COLS}
