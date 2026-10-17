# Network Intrusion Detection System (NIDS)

Detect suspicious network traffic and classify it as **normal** or **malicious**
using machine learning, then identify the **attack category** — exposed through a
REST API and a security dashboard.

Built for a modest laptop (Intel i5 / 8 GB RAM): the **UNSW-NB15** dataset and
tree-based models keep full training under ~4 minutes with no GPU.

## Table of contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run](#run)
- [Configuration reference](#configuration-reference)
- [Dataset & features](#dataset--features)
- [Pipeline details](#pipeline-details)
- [Prediction API](#prediction-api)
- [Dashboard](#dashboard)
- [PCAP scoring](#pcap-scoring)
- [Sample data & helper scripts](#sample-data--helper-scripts)
- [Reference results (UNSW-NB15 official test set)](#reference-results-unsw-nb15-official-test-set)
- [False-positive reduction](#false-positive-reduction)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

## Highlights

- **Two model families** per stage: Random Forest and XGBoost, best one auto-selected and saved.
- **Binary detector** (Normal vs Attack) with *calibrated* probabilities and a tuned decision threshold.
- **Multiclass classifier** for 10 attack categories (including `Normal`).
- **Feature engineering** pipeline: median imputation, `RobustScaler`, one-hot encoding, correlation filter + mutual-information / RF ranking.
- **FastAPI** service with Pydantic validation (42 raw features, JSON or PCAP upload).
- **Streamlit** security dashboard: overview, single/batch predictions, PCAP analysis, live-replay monitoring, and report downloads.
- **PCAP ingestion** via scapy → UNSW-NB15-style flow records.
- **Reproducible** (`random_state=42`) and **regression-guarded** by the pytest suite.

## Architecture

```
Raw traffic (UNSW-NB15 CSV)          Live traffic (PCAP)
   │  data_loader.py                    │  pcap_extractor.py (scapy)
   ▼                                    ▼
Preprocessing                         UNSW-style flow features
   │  missing values · duplicates · one-hot · RobustScaler
   ▼
Feature Selection
   │  correlation filter + mutual-information / RF importance ranking
   ▼
Model Training
   ├── binary      Random Forest + XGBoost   (Normal / Attack)
   └── multiclass  Random Forest + XGBoost   (10 attack categories)
   ▼
Evaluation        precision · recall · F1 · ROC-AUC · PR-AUC · MCC · FPR · calibration · threshold tuning
   ▼
Prediction API    FastAPI + Pydantic validation  (JSON records or PCAP upload)
   ▼
Security Dashboard  Streamlit  (Normal / Suspicious / Malicious + attack type)
```

Every stage lives in the `src/` library; the API and dashboard are thin clients
that call the same `src.predict` helpers, so online and offline scoring stay
consistent.

## Project layout

```
├── data/
│   ├── raw/              UNSW-NB15 training/testing CSVs (auto-downloaded)
│   ├── csvs/             sample feature CSVs for batch testing
│   └── pcaps/            sample captures for PCAP analysis
├── models/              ids_pipeline.joblib, ids_multiclass.joblib + metadata JSON
├── reports/             metrics.{csv,json} + figures/*.png + predictions
├── src/                 pipeline library
│   ├── config.py            paths, feature lists, hyper-parameters, threshold objective
│   ├── data_loader.py       download + load (auto-detects swapped train/test filenames)
│   ├── preprocessing.py     cleaning, ColumnTransformer, FeatureSelector
│   ├── feature_selection.py correlation + importance selection
│   ├── train.py             orchestration (binary + multiclass)
│   ├── evaluate.py          metrics, threshold tuning, plots
│   ├── predict.py           shared inference helpers (loading, scoring, severity)
│   └── pcap_extractor.py    PCAP -> UNSW-NB15 feature records (scapy)
├── api/                 FastAPI service (main.py, schemas.py)
├── dashboard/           Streamlit app (app.py)
├── tests/               pytest suite + metric regression guard
└── scripts/
    ├── run_pipeline.py      train everything
    ├── score_pcap.py        score a capture file
    ├── make_sample_csvs.py  generate sample feature CSVs
    ├── make_sample_pcaps.py generate sample capture files
    └── make_sample_batch.py build one larger batch CSV
```

## Requirements

- Python **3.10+** (developed and tested on **3.13**)
- ~1 GB free disk for the dataset + artifacts; no GPU required
- Windows / Linux / macOS

Key dependencies (see `requirements.txt`): `scikit-learn>=1.5.2`, `xgboost>=2.1.3`,
`fastapi` + `uvicorn`, `streamlit`, `scapy`, `pandas`, `numpy`, `joblib`,
`imbalanced-learn`, `matplotlib`/`seaborn`, `pytest`, `httpx`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
# 1. Train (downloads dataset on first run, ~4 min on i5)
python scripts\run_pipeline.py

# 2. Prediction API  -> http://localhost:8000/docs
uvicorn api.main:app --reload --port 8000

# 3. Dashboard       -> http://localhost:8501
streamlit run dashboard\app.py

# 4. Score a capture file directly
python scripts\score_pcap.py capture.pcap --max-flows 5000
```

Running order matters only in that both the API and the dashboard need a trained
model; without one, `/health` reports `model_missing` and the dashboard shows a
warning. The API/dashboard can talk to each other (see [Dashboard](#dashboard)).

## Configuration reference

All runtime knobs live in `src/config.py` (`Config` dataclass) and the module-level
path constants. The most useful ones:

| Setting | Default | Meaning |
|---------|---------|---------|
| `random_state` | `42` | Seed for split, sampling, models |
| `correction_threshold` | `0.95` | Drop features whose \|correlation\| exceeds this |
| `importance_sample` | `60000` | Rows sampled for feature selection |
| `top_k_features` | `25` | Number of features kept for training |
| `proto_min_frequency` | `0.01` | One-hot categories below this frequency are grouped |
| `use_class_weight` | `True` | `balanced` for RF, `scale_pos_weight` for XGB (binary) |
| `calibrate` / `calibration_method` | `True` / `sigmoid` | Probability calibration wrapper |
| `tune_threshold` | `True` | Sweep thresholds on the validation split |
| `threshold_objective` | `precision_at_recall` | `f1` \| `cost` \| `precision_at_recall` |
| `min_recall` | `0.90` | Floor used by `precision_at_recall` |
| `fp_cost` / `fn_cost` | `1.0` / `3.0` | Weights for the `cost` objective |
| `hyperparameter_search` | `False` | Optional RandomizedSearchCV (expensive) |
| `rf_params.n_estimators` | `300` | Binary Random Forest trees |
| `xgb_params.n_estimators` | `400`, `max_depth=6`, `lr=0.1` | Binary XGBoost |
| `mc_xgb_params.max_depth` | `8` | Multiclass XGBoost |

Module paths: `MODEL_PATH`, `MULTICLASS_MODEL_PATH`, `METADATA_PATH`,
`MULTICLASS_METADATA_PATH`, `METRICS_CSV/JSON`, `MULTICLASS_METRICS_CSV/JSON`,
`DATA_RAW`, `MODELS_DIR`, `REPORTS_DIR`, `FIGURES_DIR`.

## Dataset & features

- **UNSW-NB15** (Moustafa & Slay, 2015), free for academic use. Downloaded
  automatically from a public mirror on first run.
- Input features = **42** raw columns: **39 numeric** behavioural features
  (duration, byte/packet counts, rate, TTLs, load, jitter, TCP details, connection
  counters, etc.) + **3 categorical** columns: `proto`, `service`, `state`.
- **ID_NO** is dropped; `attack_cat` (when present) is used for the multiclass
  target; `label` is the binary target.
- Raw files contain many duplicate flows; `clean_frame` drops exact duplicates, so
  training uses **~107,740 unique rows** (test: **~55,945**).
- The loader auto-corrects mirrors where the train/test filenames are swapped
  (by comparing row counts).

## Pipeline details

1. **Cleaning** (`preprocessing.clean_frame`): strip column names, drop `ID_NO`
   and duplicates, coerce numerics, replace ±inf with NaN, normalise categorical
   blanks to `unknown`, map empty `attack_cat` to `Normal`.
2. **Preprocessing transformer** (`build_preprocessor`): numeric branch =
   `SimpleImputer(median)` + `RobustScaler`; categorical branch =
   `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore",
   min_frequency=0.01)`. Wrapped in a `ColumnTransformer`.
3. **Feature selection** (`feature_selection.select_features`): fit the encoder on
   a 60k-row sample, drop zero-variance columns, greedily remove features with
   \|corr\| > 0.95, rank the rest by mutual information (RF importances computed
   for reference), keep the **top 25**.
4. **Binary training** (`train.train_binary`): split training data 80/20 into
   fit/validation, fit RF + XGBoost, wrap each in sigmoid calibration
   (`CalibratedClassifierCV` over `FrozenEstimator`), tune a decision threshold on
   the validation split, then evaluate on the official test set. The model with
   the best **F1** is saved to `models/ids_pipeline.joblib`.
5. **Multiclass training** (`train.train_multiclass`): `LabelEncoder` over the 10
   categories (incl. `Normal`), fit RF + XGBoost with class weighting, pick the
   best **macro-F1** model → `models/ids_multiclass.joblib`.
6. **Evaluation & artifacts** (`evaluate.py`): accuracy, precision, recall, F1,
   ROC-AUC, PR-AUC, MCC, FPR and confusion counts; confusion matrices, ROC curve,
   threshold curve, model comparison, feature importance, correlation heatmap and
   class distribution figures; plus `test_predictions.csv` with per-row scores.
7. **Metadata**: `model_metadata.json` records the best model, threshold,
   selected/dropped features, per-model metrics and row counts;
   `multiclass_metadata.json` records the class list and metrics.

## Prediction API

Base URL when running locally: `http://localhost:8000` (Swagger at `/docs`,
ReDoc at `/redoc`). Models are preloaded on startup, so the first request responds
in ~300 ms.

| Method | Endpoint          | Description                                     |
|--------|-------------------|-------------------------------------------------|
| GET    | `/health`         | Service + model status                          |
| GET    | `/model-info`     | Model, threshold, categories, selected features, metrics |
| GET    | `/example`        | Example request body (a representative normal flow) |
| POST   | `/predict`        | Score one flow (returns verdict + `attack_category`) |
| POST   | `/predict/batch`  | Score up to 1000 flows                          |
| POST   | `/predict/pcap`   | Upload a `.pcap`/`.pcapng` and get a summary    |

**Severity bands** (`src/predict.py`): `Normal` below the decision threshold,
`Suspicious` at/above it, `Malicious` at/above `max(0.80, threshold)`.

**Request** — a single flow uses the 42 raw feature names (`TrafficRecord`,
`extra="forbid"`, all numerics `>= 0`). Get a ready-made body from `GET /example`.
Illustrative:

```json
{
  "dur": 0.12, "spkts": 6, "dpkts": 4, "sbytes": 258, "dbytes": 172,
  "rate": 74.09, "sttl": 252, "dttl": 254, "sload": 14158.9, "dload": 8495.4,
  "sloss": 0, "dloss": 0, "sinpkt": 24.3, "dinpkt": 8.4, "sjit": 30.2, "djit": 11.8,
  "swin": 255, "stcpb": 621772692, "dtcpb": 2202533631, "dwin": 255,
  "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 43, "dmean": 43,
  "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 1, "ct_state_ttl": 0,
  "ct_dst_ltm": 1, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1,
  "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1,
  "ct_srv_dst": 1, "is_sm_ips_ports": 0, "proto": "tcp", "service": "-", "state": "FIN"
}
```

**Response** (`PredictionResponse`):

```json
{
  "label": 0,
  "verdict": "Normal",
  "severity": "Normal",
  "attack_probability": 0.0312,
  "confidence": 0.9688,
  "attack_category": "Normal"
}
```

Batch requests wrap records in `{"records": [ ... ]}` (1–1000 items) and return
`{"count", "attacks", "predictions"}`. `/predict/pcap` accepts a multipart file
plus an optional `max_flows` query param (default `5000`) and returns
`flows`, `attacks`, `attack_rate`, `severity_counts`, `category_counts` and the
top 10 suspicious flows.

**Status codes:** `200` success; `422` validation / no flows found; `400`
unparseable PCAP; `503` when no trained model is present.

### Calling the library directly

```python
from src import predict

predict.predict_records([{...}])         # list of raw records -> list of verdict dicts
predict.predict_dataframe(df)            # DataFrame in -> enriched DataFrame out
```

## Dashboard

`streamlit run dashboard\app.py` → `http://localhost:8501`. Tabs:

- **Overview** — dataset/model/threshold/F1 metrics, per-model drill-down, selected
  features, and a downloadable self-contained HTML security report.
- **Single Prediction** — form over all 42 features with verdict, severity,
  probability, confidence and attack category.
- **Batch Analysis** — score an uploaded CSV or a CSV from a local folder.
- **PCAP Analysis** — extract and score an uploaded capture or one from a local folder.
- **Monitoring** — live-replay feed of `test_predictions.csv` + a filterable alert
  explorer (severity, predicted category, actual, min probability) with CSV export.
- **Model Reports** — all evaluation figures.

The sidebar has a **"Use FastAPI service"** toggle. Off (default) runs inference
locally with the same `src.predict` code; on, it calls the API. The API base URL
comes from the `NIDS_API_URL` environment variable (default
`http://localhost:8000`).

## PCAP scoring

`src/pcap_extractor.py` groups packets into bidirectional flows with scapy and
computes the UNSW-NB15 feature set. A few features that need deep packet
inspection (HTTP transaction depth, FTP commands, retransmission analysis) are
approximated as `0`, and the `ct_*` last-100-flow counters use a simplified
window — good enough for live triage, not bit-exact with the reference exporter.
`sloss`/`dloss` are set to 0; `tcprtt`/`synack`/`ackdat` are derived from the TCP
handshake.

```powershell
# CLI
python scripts\score_pcap.py capture.pcap --max-flows 5000 --out results.csv

# API
curl -F "file=@capture.pcap" "http://localhost:8000/predict/pcap?max_flows=5000"
```

## Sample data & helper scripts

Ready-made fixtures ship in the repo:

- `data/csvs/` — `sample_normal_flows.csv`, `sample_attack_flows.csv`,
  `sample_mixed_flows.csv`, `sample_dos_generic.csv`,
  `sample_exploits_recon.csv`, `sample_small_batch.csv`
- `data/pcaps/` — `normal_http.pcap`, `port_scan.pcap`, `syn_flood.pcap`,
  `udp_dns.pcap`, `icmp_ping.pcap`, `mixed_traffic.pcap`

Regenerate or resize them (both need the raw dataset, i.e. run training first):

```powershell
python scripts\make_sample_csvs.py  --out data\csvs  --rows 200
python scripts\make_sample_pcaps.py --out data\pcaps
python scripts\make_sample_batch.py --rows 2000 --out data\samples\batch_sample.csv
```

## Reference results (UNSW-NB15 official test set)

Binary classification (Normal vs Attack). Probabilities are **calibrated** and the
decision threshold is tuned on a validation split to maximise precision subject to
`recall >= 0.90` (see [False-positive reduction](#false-positive-reduction)):

| Model        | Acc    | Prec   | Recall | F1     | ROC-AUC | PR-AUC | MCC    | FPR    | Thr  | Fit (s) |
|--------------|--------|--------|--------|--------|---------|--------|--------|--------|------|---------|
| RandomForest | 0.8475 | 0.7586 | 0.8909 | 0.8195 | 0.9513  | 0.9319 | 0.6956 | 0.1801 | 0.72 | 22.5    |
| XGBoost ★    | 0.8546 | 0.7707 | 0.8908 | 0.8264 | 0.9572  | 0.9443 | 0.7079 | 0.1684 | 0.76 | 7.8     |

★ best model (selected by F1), saved to `models/ids_pipeline.joblib`.

Attack-category classification (10 classes incl. Normal):

| Model        | Accuracy | Macro F1 | Weighted F1 | Fit (s) |
|--------------|----------|----------|-------------|---------|
| RandomForest | 0.6699   | 0.5205   | 0.7177      | 23.6    |
| XGBoost ★    | 0.7452   | 0.5317   | 0.7730      | 130.6   |

★ best model (selected by macro-F1), saved to `models/ids_multiclass.joblib`.

Categories: `Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance, Shellcode, Worms`.

Numbers reflect deduplicated UNSW-NB15 (`107,740` train / `55,945` test rows);
re-running `python scripts\run_pipeline.py` reproduces them (`random_state=42`).

## False-positive reduction

Probabilities are calibrated (`CalibratedClassifierCV`, sigmoid) and the threshold
is chosen by a configurable objective in `src/config.py`:

| Objective              | Behaviour                                                        |
|------------------------|------------------------------------------------------------------|
| `precision_at_recall`  | maximise precision with `recall >= min_recall` **(default)**      |
| `f1`                   | maximise F1                                                      |
| `cost`                 | minimise `fp_cost*FP + fn_cost*FN`                               |

Impact for XGBoost (see `reports/threshold_analysis.json`):

| Metric            | F1-optimal thr | Deployed thr | Change |
|-------------------|----------------|--------------|--------|
| Threshold         | 0.43           | 0.76         | —      |
| Precision         | 0.6977         | 0.7707       | +0.073 |
| Recall            | 0.9446         | 0.8908       | −0.054 |
| False-positive rate | 0.2602       | 0.1684       | −0.092 (3,137 fewer FPs) |

## Testing

```powershell
python -m pytest -q
```

The suite covers preprocessing (cleaning, imputation, encoding), feature selection
(correlation filter, threshold objectives), the PCAP extractor, the API contract
(via `TestClient`), and a **metric regression guard** that fails if a retrained
model drops below quality floors (`tests/test_metrics_regression.py` — e.g. binary
F1 ≥ 0.80, ROC-AUC ≥ 0.94, FPR ≤ 0.20; multiclass accuracy ≥ 0.64, macro-F1 ≥ 0.50).
API/regression tests skip automatically when no trained model is present.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/health` returns `model_missing`, `503` on predict | Train first: `python scripts\run_pipeline.py` |
| `FileNotFoundError: No trained model at models/ids_pipeline.joblib` | Same as above |
| Dataset download fails | Place `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` in `data/raw/` manually |
| `scapy is required` | `pip install scapy` (or `pip install -r requirements.txt`) |
| Placeholder for calibration `calibration unavailable` | Requires scikit-learn ≥ 1.6 (`FrozenEstimator`); training proceeds uncalibrated |
| Dashboard can't reach API | Start `uvicorn api.main:app --port 8000`; confirm the URL / `NIDS_API_URL` |
| PCAP yields "No IP flows found" | The capture has no IPv4 traffic, or raise `--max-flows` |

## Notes

- Dataset: UNSW-NB15 (Moustafa & Slay, 2015), free for academic use.
- Raw UNSW-NB15 files contain many duplicate flows; `clean_frame` drops them, so
  training uses ~107,740 unique rows.
- Class weighting is applied for the multiclass model (heavily imbalanced); the
  binary XGBoost uses `scale_pos_weight`.
- The PCAP exporter is approximate (see [PCAP scoring](#pcap-scoring)) — use it
  for triage, not exact replication of the reference feature extractor.
- Tune threshold behaviour and hyper-parameters in `src/config.py`.
- Generated artifacts (`data/raw/*.csv`, `models/*.joblib`, `reports/**`) are
  git-ignored and reproducible from the pipeline.
