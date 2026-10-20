# Network Intrusion Detection System (NIDS)

Detect suspicious network traffic and classify it as **normal** or **malicious**
using machine learning, then identify the **attack category** — exposed through a
REST API and a security dashboard.

Built for a modest laptop (Intel i5 / 8 GB RAM): the **UNSW-NB15** dataset and
tree-based models keep full training under ~4 minutes with no GPU.

> **⚠️ Live demo note for reviewers:** This app is deployed on the free
> **Streamlit Community Cloud**. Community Cloud apps go to sleep after
> **~12 hours of inactivity** (and reboot after ~7 days of sleeping). If the
> deployed app greets you with "This app has gone to sleep…", simply click
> **"Yes, get this app back up!"** — it wakes in ~30 seconds. Data is not lost:
> the models, metrics and sample files are baked into the repository, so the
> cold start rebuilds everything it needs. There is **no paid tier trick
> involved** — this behaviour is automatic and applies to every free
> Streamlit Cloud app.

## Table of contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Quickstart (no training needed)](#quickstart-no-training-needed)
- [Setup](#setup)
- [Run](#run)
- [Deploy](#deploy)
- [Cloud sleep behaviour (Streamlit Community Cloud)](#cloud-sleep-behaviour-streamlit-community-cloud)
- [Configuration reference](#configuration-reference)
- [Dataset & features](#dataset--features)
- [Pipeline details](#pipeline-details)
- [Prediction API](#prediction-api)
- [Dashboard](#dashboard)
- [Dashboard UI walkthrough (every element explained)](#dashboard-ui-walkthrough-every-element-explained)
- [Feature glossary (the 42 inputs you see in the UI)](#feature-glossary-the-42-inputs-you-see-in-the-ui)
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

## Quickstart (no training needed)

The repo ships with trained models and sample data, so you can try everything in
under a minute after installing dependencies:

```powershell
pip install -r requirements.txt

# Dashboard with samples you can click through immediately
streamlit run dashboard\app.py
# → Batch Analysis tab: pick data\csvs\sample_mixed_flows.csv
# → PCAP Analysis tab:  pick data\pcaps\port_scan.pcap  (or syn_flood.pcap, ...)
```

Or go straight to the CLI / API with the bundled sample captures and CSVs:

```powershell
# CLI: score a suspicious capture
python scripts\score_pcap.py data\pcaps\syn_flood.pcap

# API
uvicorn api.main:app --port 8000
# then in another terminal:
curl -F "file=@data/pcaps/syn_flood.pcap" http://localhost:8000/predict/pcap
curl "http://localhost:8000/example"          # ready-made 42-feature request body
```

Sample files included (`data/csvs/`, `data/pcaps/`):

| File | What it demonstrates |
|------|----------------------|
| `sample_normal_flows.csv` | Benign traffic — should score `Normal` |
| `sample_attack_flows.csv` | Malicious flows — should score `Malicious` |
| `sample_mixed_flows.csv`  | Blend of both — best for the dashboard demo |
| `sample_dos_generic.csv`  | DoS / Generic categories |
| `sample_exploits_recon.csv` | Exploits / Reconnaissance categories |
| `sample_small_batch.csv`  | Tiny file for a super-fast first run |
| `normal_http.pcap`, `port_scan.pcap`, `syn_flood.pcap`, `udp_dns.pcap`, `icmp_ping.pcap`, `mixed_traffic.pcap` | Captures of common benign and attack patterns |

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

## Deploy

### Streamlit Community Cloud (free, easiest)

The repository is already deployment-ready: the dark theme (`.streamlit/config.toml`),
metrics, evaluation figures and `reports/test_predictions.csv` are committed, and
`requirements.txt` contains every dependency.

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select
   this repository and branch `main`.
2. Set **Main file path** to `dashboard/app.py`.
3. Under **Advanced settings** choose **Python 3.12** (xgboost 2.1 and
   imbalanced-learn 0.12 don't ship wheels for 3.13 yet), then click **Deploy**.
4. The app boots in degraded mode (metrics, batch CSV analysis, live monitoring
   and reports all work without the model binaries).

To also run **local-model predictions** (Single Prediction / Batch / PCAP without
the API) the ~24 MB model binaries — `models/ids_pipeline.joblib` and
`models/ids_multiclass.joblib` — are already committed to the repository, so the
Streamlit Cloud clone contains everything it needs. If you fork or strip large
files, you can still score through the API instead: enable **Use FastAPI
service** in the sidebar and set the **API URL** (see below), or pre-fill it with
the `NIDS_API_URL` secret/env var.

### Self-hosting (Docker-ready host / private server)

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000   # optional backend
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Optional env vars / Streamlit secrets for the dashboard:

| Variable | Purpose |
|----------|---------|
| `NIDS_API_URL` | Default FastAPI URL pre-filled in the sidebar |

## Cloud sleep behaviour (Streamlit Community Cloud)

The hosted demo runs on the free Streamlit Community Cloud, which has one
behaviour you must know about as a reviewer or interviewer:

- After **~12 hours with no visitors** the container is **put to sleep**.
- After sleeping for **~7 more days** the app is fully rebooted (fresh VM).
- On the next visit you may see the *"This app has gone to sleep to save
  resources"* banner; clicking **"Yes, get this app back up!"** restarts it —
  the first visit after a wake-up takes ~30–60 s while dependencies install.
- **Nothing is lost**: model binaries (`models/*.joblib`), metrics JSON,
  evaluation figures, sample CSVs and sample PCAPs are all **committed to the
  repository**, so a cold start restores the exact same state.
- Why sleep exists: Streamlit's free tier gives each user one shared
  container (~1 GB RAM). To keep that cost-free, idle containers are
  hibernated. The only ways around it are paid tiers (Streamlit teams / other
  hosts like Render or Railway with a keep-alive ping) or a scheduled cron
  ping — not needed for this demo.
- **Interview one-liner:** "The demo sleeps after 12 idle hours because
  Streamlit's free Cloud hibernates idle containers; the full system runs
  permanently with `uvicorn` + `streamlit run` on any host."

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

## Dashboard UI walkthrough (every element explained)

This section exists so you can open the app, point at anything on screen, and
explain *what it is, why it's there, and how it works underneath* — useful for
demos and interviews.

### Sidebar — "🛡️ NIDS Control"

- **"Use FastAPI service" toggle (default: off).** Chooses the inference
  backend. *Off* → the dashboard imports the same `src.predict` library and
  scores in-process with the committed `.joblib` models; nothing else needs to
  run. *On* → every prediction becomes an HTTP call to the FastAPI service,
  which demonstrates the deployment pattern analysts actually use (UI and
  model on separate servers). Both paths produce identical results because the
  API endpoints wrap the same `predict_records()` function.
- **API URL box.** Pre-filled from the `NIDS_API_URL` environment variable /
  Streamlit secret, falling back to `http://localhost:8000`. When the toggle is
  on, the sidebar immediately does a `GET /health` health-check and shows
  **"API reachable"** (green) or **"API not reachable"** (red). This is the
  same health endpoint a load-balancer or Kubernetes probe would use.
- **"Local model loaded" / "No model found"** status. If no `.joblib` exists
  the app degrades gracefully — Overview metrics, monitoring, reports and
  sample-driven features still work; only local scoring is disabled.
- **Pipeline mini-diagram** (Preprocess → Feature select → Train → Evaluate →
  Predict) — a visible reminder that the UI is just the last stage of an
  offline-trained ML pipeline.

### Header

Shows the app identity: *"Real-time detection of malicious traffic ·
UNSW-NB15 · Random Forest + XGBoost"*. The two model families are trained per
stage and the better one is auto-selected — no single-library lock-in.

### Tab 1 — Overview (the "model zoo" screen)

| UI element | What it shows | Why it matters (talking points) |
|---|---|---|
| **Dataset** metric card | `UNSW-NB15` | The benchmark dataset used for training/testing — 42-flow features + 10 attack categories. |
| **Best model** card | e.g. `XGBoost` | Winner of the RF vs XGBoost bake-off, selected automatically by F1 (binary) / macro-F1 (multiclass). |
| **Decision threshold** card | e.g. `0.76` | Not the default 0.5! Tuned on a validation split to **maximise precision subject to recall ≥ 0.90** — the core false-positive-reduction story. |
| **Best F1** card | e.g. `0.8264` | Headline quality metric on the official test set. |
| **Attack categories** caption | the 10 classes | `Normal, DoS, Exploits, Generic, Reconnaissance, Fuzzers, Analysis, Backdoor, Shellcode, Worms` — shows the second (multiclass) model. |
| **Model comparison table** | accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC, threshold per model | Lets you argue RF vs XGBoost with numbers. **ROC-AUC** = ranking quality across all thresholds; **PR-AUC** emphasises the rare-attack class; **MCC** (Matthews) is a balanced single number robust to class imbalance. |
| **Model comparison / ROC curves images** | bar chart + ROC plot | The ROC tells you how much better than random the detector is (AUC ≈ 0.957 vs 0.5). |
| **Per-model drill-down** (dropdown) | confusion matrix image + full metric table for the selected model | The confusion matrix answers the two operational questions: *how many attacks were missed* (FN — cost = intrusion) and *how many benign flows were flagged* (FP — cost = analyst time). |
| **Download HTML security report** button | self-contained report | Generates a portable HTML file with the metrics table and embedded (base64) figures — simulates the compliance/reporting deliverable a SOC would archive. |
| **Selected features** chips | e.g. `sttl`, `ct_dst_sport_ltm`, … | Shows the **top 25 of 42** features kept by the selection stage (correlation filter + mutual-information ranking). An interviewer will appreciate knowing the model was slimmed to 25 inputs for speed and overfitting control. |

If no trained model is found, this tab shows *"No metrics yet…"* and works in
degraded mode — by design, so the demo never crashes on cold start.

### Tab 2 — Single Prediction (42-feature form)

- **Form layout:** three columns — *volume/timing* (duration, packets, bytes,
  rate, loads), *TCP-baseline* (TTLs, window, base sequence, RTT/synack/ackdat,
  packet-size means), and *content/context counters* (`trans_depth`,
  `response_body_len`, the `ct_*` connection counters, FTP/HTTP flags) plus the
  three categoricals `proto` / `service` / `state`.
- **Dropdowns for categoricals** (`tcp/udp/icmp…`, `http/dns/ssh…`,
  `FIN/CON/RST…`): these are one-hot encoded by the pipeline
  (`OneHotEncoder(handle_unknown="ignore")`) — unknown categories never crash
  the model, they just don't match learned columns.
- **Every numeric is clamped to ≥ 0.** The model was trained on physically
  non-negative quantities; negative input would be out-of-distribution.
- **"Analyze traffic"** button runs the form and returns a chime-style
  verdict: 🚨 red alert banner for Attack, ✅ green success for Normal, plus
  three metric cards:
  - **Attack probability** — the *calibrated* probability that this flow is
    malicious (calibration means "0.9 really means 90%", not just a raw score).
  - **Confidence** — `1 − attack_probability` for normal, i.e. how sure the
    model is about the shown verdict.
  - **Attack category** — the multiclass model's guess (`Normal`, `DoS`,
    `Exploits`, …). Both models run on every prediction.
- **Severity bands** (explained): probability < threshold → `Normal`; ≥
  threshold → `Suspicious`; ≥ max(0.80, threshold) → `Malicious`. This gives
  analysts a triage vocabulary instead of a raw 0–1 number.
- Pre-filled values come from `GET /example` (a realistic benign flow), so
  clicking Analyze immediately demonstrates a working prediction.

### Tab 3 — Batch Analysis (bulk scoring)

- **Upload CSV / Local-folder switcher.** Any CSV containing the 42 raw UNSW
  features (labels not required). The folder mode is for local bulk files and
  can include subfolders.
- **One-click sample buttons** (normal / attack / mixed / DoS-Generic) — each
  loads a bundled fixture so a reviewer can produce results in one click with
  zero downloads. The **mixed** sample is the demo sweet-spot because it
  produces both verdicts.
- **Result cards:** **Flows analyzed** (total), **Attacks detected** (count of
  Attack verdicts), **Attack rate** (percentage — the operational KPI a SOC
  cares about).
- **Severity bar chart** — distribution across Normal / Suspicious /
  Malicious; a spike in Suspicious+Malicious visually flags an incident even
  before reading the table.
- **Results table + Download CSV** — every original row enriched with
  `verdict`, `severity`, `attack_probability`; downloadable as
  `nids_predictions.csv` for offline SIEM ingestion.
- Internally batch calls `/predict/batch` (API mode, max 1000 flows) or the
  same shared `predict_dataframe()` locally — one code path, two transports.

### Tab 4 — PCAP Analysis (from packets to verdicts)

- Accepts `.pcap` / `.pcapng`, either uploaded, picked one-click
  (Port scan / SYN flood / Normal HTTP / Mixed) or from a local folder.
- **"Max flows to score"** slider (100–200,000, default 5,000) — a safety valve
  to bound memory/CPU on large captures.
- Pipeline behind the button: **scapy→ flow table→ model→ UI**. The extractor
  groups packets into bidirectional flows and computes the UNSW-NB15 features
  (duration, packet/byte counts, TTLs, jitter, TCP handshake timing, `ct_*`
  window counters). Deep-inspection features (HTTP depth, FTP commands) are
  approximated as 0 — an honest limitation worth stating.
- **Outputs:**
  - *Flows extracted / Attacks detected / Attack rate* metric cards;
  - left chart: **severity distribution**, right chart: **attack-category
    distribution** (which DoS/Exploits/Generic mixture was found);
  - **Top suspicious flows** table — the 10 highest attack-probability flows
    with their protocol/service/state and category — This is "analyst
    triage": who do you investigate first.
- Sample captures make this reproducible: e.g. `port_scan.pcap` should show a
  high attack rate driven by thousands of small, fast connection-count records.

### Tab 5 — Monitoring (live ops simulation)

- **Play live simulation toggle** — replays the recorded test-set predictions
  (`reports/test_predictions.csv`) through a `st.fragment(run_every=2.0)`
  auto-refreshing fragment, i.e. a **streaming-dashboard simulation** without
  any backend socket. In production you'd swap the feed for a Kafka/socket
  source; the UI contract is identical.
- **Events per tick slider** (5–100) controls the replay rate; **Reset
  stream** rewinds to 0.
- **Windowed metrics:** *Streamed x/total*, *Attacks in window*,
  *Malicious* (highest severity seen in the current window), *Window size* —
  fixed 150-event rolling window to keep the view readable, mimicking an
  analyst's recent-events pane.
- **Rolling event table** (`actual / predicted / severity / attack_cat /
  attack_cat_pred / attack_probability`) — showing *actual* next to
  *predicted* makes it easy to demo true/false positives live. What each
  streamed column means (worth knowing for interviews):

  | Column | What it is | How it's produced |
  |---|---|---|
  | `attack_probability` | The model's *calibrated* probability (0–1) that the flow is malicious | The saved binary model's output after sigmoid calibration — "0.83" genuinely means 83% confidence. |
  | `predicted` | Binary verdict: `Normal` or `Attack` | `attack_probability` compared against the tuned decision threshold (0.76). |
  | `severity` | Triage bucket: `Normal` / `Suspicious` / `Malicious` | Bands: < threshold = Normal; ≥ threshold = Suspicious; ≥ max(0.80, threshold) = Malicious. Tells the analyst *how loudly* to react. |
  | `actual` | Ground truth: `Normal` or `Attack` (from the test set labels) | Only known because this stream replays labelled `test_predictions.csv`; in true live traffic it would populate after forensics. Comparing `actual` vs `predicted` is exactly how FP/FN are spotted. |
  | `attack_cat_pred` | The multiclass model's guess: which of the *10 categories* this attack is (`DoS`, `Exploits`, `Generic`, …, or `Normal`) | The second (multiclass) model, always run in parallel with the binary one — binary answers *"is it an attack?"*, this answers *"what kind?"*. |
  | `attack_cat` | The true category label from the benchmark | Lets you demo category-level accuracy (e.g. predicted `DoS`, actually `DoS`). |

  Note that all six columns are **model outputs or labels — none of the raw
  42 flow features are streamed here**. The event table is a *verdict feed*,
  not a packet dump; this separation mirrors real SIEM tools (alerts stream,
  raw evidence lives in packet storage). If you need the underlying flow
  features, they're on the Single Prediction tab (all 42, live-editable) or
  in the batch CSV you upload.
- **Alert explorer** below, with four filters:
  - **Severity multi-select** (defaults to Malicious + Suspicious — the two
    buckets an analyst acts on);
  - **Predicted category** multi-select (e.g. show only DoS);
  - **Actual** radio (All/Attack/Normal) — lets you check FP/FN mix;
  - **Min probability** slider (evidence-strength cutoff).
  - Three follow-up cards: **Matching alerts**, **Precision (filtered)** —
    the fraction of currently shown alerts that were real attacks, computed
    live from `y_true` — and **Mean probability**. Export button downloads
    the filtered alert set as CSV.
- Communication: this tab demonstrates threshold/severity triage on real
  model output — a favourite interview question: *"How would an analyst use
  this in practice?"* Answer: start at Malicious+0.8, widen to Suspicious if
  the precision stays high.

### Tab 6 — Model Reports (evidence wall)

Grid of all training-time artifacts, each with a story:

| Figure | Interview explanation |
|---|---|
| `confusion_matrix_XGBoost/RandomForest.png` | Per-model FP/FN breakdown; the operational cost view. |
| `roc_curves.png` | ROC-AUC ≈ 0.957 — detector ranks well across all thresholds. |
| `threshold_curve.png` | Why 0.76 was picked — precision rises while recall floor stays ≥ 0.90. |
| `model_comparison.png` | RF vs XGBoost side-by-side; XGBoost wins on F1. |
| `multiclass_confusion_matrix.png` | Where the 10-category model confuses classes (DoS vs Generic, Exploits vs Fuzzers). |
| `multiclass_comparison.png` | Same bake-off for the category model (macro-F1 criterion). |
| `feature_importance.png` | The top-25 selected features — good for the *"which features matter?"* question (`sttl`, connection counters dominate). |
| `correlation_heatmap.png` | Justifies the \|corr\| > 0.95 redundancy filter with a picture. |
| `class_distribution.png` | Shows UNSW-NB15's class imbalance, motivating class weighting and macro-F1. |

(On the free cloud these figures come straight from the committed
`reports/` folder; they regenerate identically when you retrain locally
because `random_state=42`.)

### Cross-cutting UI facts worth mentioning in an interview

- **Dark SOC-style theme** is custom CSS (`.streamlit/config.toml` + in-app
  stylesheet) — shows you can go beyond Streamlit defaults.
- **Caching:** metrics and prediction tables are cached (`@st.cache_data`,
  TTL 300 s) so repeated tab switches don't re-read CSVs.
- **Graceful degradation:** every tab checks whether its artefacts exist and
  degrades to an info banner instead of crashing — required because the
  cloud app may cold-start before files are materialised.
- **Same brain everywhere:** dashboard, API and CLI all call
  `src.predict.predict_records/ dataframe`, so verdicts can never disagree
  between surfaces.

## Feature glossary (the 42 inputs you see in the UI)

Explaining these instantly signals dataset fluency. All are per-flow records
derived from the UNSW-NB15 exporter:

| Group | Features | Plain-English meaning |
|---|---|---|
| **Basic volumetrics** | `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes` | How long the flow lasted and how many packets/bytes went each direction. Scans = many tiny packets; exfil = many upstream bytes. |
| **Rates & loads** | `rate`, `sload`, `dload` | Packets/s and bytes/s per direction. SYN floods produce extreme source load. |
| **Loss & pacing** | `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit` | Retransmitted packets and inter-packet time/jitter — irregular pacing betrays tooling (e.g. fuzzers). |
| **TCP session** | `swin`, `dwin`, `stcpb`, `dtcpb`, `tcprtt`, `synack`, `ackdat`, `sttl`, `dttl` | TCP window/base-sequence discipline and handshake RTT components; TTLs expose OS fingerprint and spoofing (e.g. `sttl=0` often = crafted packets). |
| **Payload stats** | `smean`, `dmean`, `trans_depth`, `response_body_len` | Mean bytes/packet and HTTP transaction depth/response size (0 for non-HTTP). |
| **Windowed connection counters (`ct_*`)** | `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_src_ltm`, `ct_srv_dst`, `ct_flw_http_mthd`, `is_ftp_login`, `ct_ftp_cmd` | Count of related connections in the last 100 flows sharing this src/dst/service/port set — the *behavioural* features that make port scans pop out. |
| **Flags** | `is_sm_ips_ports` | Same src/dst IP and port in the window — a small-but-lethal DoS signal. |
| **Categoricals** | `proto`, `service`, `state` | Protocol (tcp/udp/…), application service (http/dns/-), and TCP connection state (FIN/CON/RST/…). One-hot encoded. |

## What is a PCAP? (and how it becomes a prediction)

A **PCAP** (short for *packet capture*) is a file format (`.pcap` / `.pcapng`)
that records **raw network traffic exactly as it crossed a network interface** —
every individual packet, its full headers and payload, and the exact timestamp,
captured by tools like **Wireshark**, **tcpdump**, **tshark** or Python's
**scapy**. It's the "black-box recorder" of a network: rather than pre-computed
summaries, it stores the actual traffic so anything can be reconstructed
afterwards.

- **Why IDS projects love it:** a PCAP is the closest thing to "live traffic"
  that you can replay offline. Whatever your sensor saw, analysts can re-see.
- **Limitation:** it sees only what crossed that one interface — encrypted
  payloads stay encrypted, and anything on other paths/directions is missing.

**How this project turns a PCAP into a verdict** (`src/pcap_extractor.py`):

1. **Packets are read** with scapy — Ethernet → IP → TCP/UDP/ICMP frames.
2. **Packets are grouped into bidirectional flows**, keyed by
   `(src IP, src port, dst IP, dst port, protocol)`. A "flow" is one logical
   conversation (e.g. one HTTP request-response), which is exactly what the
   UNSW-NB15 feature set describes.
3. **42 flow features are computed** from the grouped packets: duration, packet
   and byte counts per direction, rates, TTLs, TCP windows and handshake
   timings (`tcprtt`/`synack`/`ackdat`), packet-size means, inter-arrival
   times, jitter, and the `ct_*` connection-window counters. A few
   deep-packet-inspection features (HTTP transaction depth, FTP commands,
   retransmission loss) are approximated as `0` — honest approximation, not
   bit-exact with the official exporter.
4. **The 42-feature records pass through the exact same model** as CSVs and
   API requests (`src.predict`), so a PCAP, a spreadsheet row, and a live API
   call all produce the identical verdict, severity and attack category.

That's why the dashboard has a dedicated **PCAP Analysis** tab: it's the
"replay real captured traffic" demo — capture packets with Wireshark, drop the
file in, and watch the SOC triage (attack rate, severity/category charts, top
suspicious flows) appear.

Interview one-liner: *"PCAP = raw recorded traffic; my pipeline reconstructs
the UNSW-NB15 flow statistics from those packets with scapy, then scores the
flows with the same calibrated model used for CSV and API inputs."*

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
- Generated artifacts (`data/raw/*.csv`, `reports/**`) are git-ignored and
  reproducible from the pipeline; the trained model binaries and sample
  data/models are committed so the repo works out of the box.
