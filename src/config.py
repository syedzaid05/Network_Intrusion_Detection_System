"""Central configuration for the Network Intrusion Detection System."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_RAW / "UNSW_NB15_training-set.csv"
TEST_CSV = DATA_RAW / "UNSW_NB15_testing-set.csv"

# Dataset mirrors (filenames on some mirrors are swapped, loader auto-detects).
DATA_URLS = {
    "UNSW_NB15_training-set.csv": (
        "https://raw.githubusercontent.com/TalhaUsuf/unsw_data_analysis/master/"
        "UNSW_NB15_training-set.csv"
    ),
    "UNSW_NB15_testing-set.csv": (
        "https://raw.githubusercontent.com/TalhaUsuf/unsw_data_analysis/master/"
        "UNSW_NB15_testing-set.csv"
    ),
}

RANDOM_STATE = 42

TARGET_COL = "label"
ATTACK_CAT_COL = "attack_cat"
DROP_COLS = ["ID_NO"]

CATEGORICAL_COLS = ["proto", "service", "state"]

# 40 numeric behavioural features (everything that is not id/cat/target).
NUMERIC_COLS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]

ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


@dataclass(frozen=True)
class Config:
    """Runtime knobs for the training pipeline."""

    random_state: int = RANDOM_STATE
    test_size: float = 0.2
    correction_threshold: float = 0.95  # drop features above |corr|
    importance_sample: int = 60_000
    top_k_features: int = 25
    proto_min_frequency: float = 0.01
    tune_n_iter: int = 12
    cv_folds: int = 3

    # Threshold / imbalance handling
    tune_threshold: bool = True
    threshold_metric: str = "f1"
    use_class_weight: bool = True

    # Probability calibration (reliable probabilities -> better thresholds)
    calibrate: bool = True
    calibration_method: str = "sigmoid"

    # Threshold objective: "f1" | "cost" | "precision_at_recall"
    # "precision_at_recall" pushes the threshold up to cut false positives while
    # keeping recall >= min_recall (the main false-positive reducer).
    threshold_objective: str = "precision_at_recall"
    min_recall: float = 0.90
    fp_cost: float = 1.0
    fn_cost: float = 3.0

    # Optional randomized hyper-parameter search (expensive on 8 GB laptop)
    hyperparameter_search: bool = False
    search_n_iter: int = 8

    rf_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        }
    )
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
    )
    # Multiclass attack-category model
    mc_rf_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        }
    )
    mc_xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": 8,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "n_jobs": -1,
        }
    )


CONFIG = Config()

MODEL_PATH = MODELS_DIR / "ids_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
MULTICLASS_MODEL_PATH = MODELS_DIR / "ids_multiclass.joblib"
MULTICLASS_METADATA_PATH = MODELS_DIR / "multiclass_metadata.json"
METRICS_CSV = REPORTS_DIR / "metrics.csv"
METRICS_JSON = REPORTS_DIR / "metrics.json"
MULTICLASS_METRICS_CSV = REPORTS_DIR / "multiclass_metrics.csv"
MULTICLASS_METRICS_JSON = REPORTS_DIR / "multiclass_metrics.json"
