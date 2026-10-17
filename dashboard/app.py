"""Streamlit security dashboard for the Network Intrusion Detection System.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, pcap_extractor, predict  # noqa: E402

st.set_page_config(page_title="NIDS Security Dashboard", page_icon="🛡️", layout="wide")

PROTO_OPTIONS = ["tcp", "udp", "icmp", "arp", "ospf", "sctp", "igmp", "any"]
SERVICE_OPTIONS = [
    "-", "http", "https", "dns", "ftp", "ftp-data", "smtp", "pop3", "ssh",
    "snmp", "dhcp", "radius", "irc", "ssl", "other",
]
STATE_OPTIONS = ["FIN", "CON", "INT", "REQ", "RST", "ECO", "PAR", "URN", "no", "acc", "clo"]


def figure_path(name: str) -> Path:
    return config.FIGURES_DIR / name


def load_metrics() -> dict:
    if config.METRICS_JSON.exists():
        return json.loads(config.METRICS_JSON.read_text())
    return predict.load_metadata().get("metrics", {})


def load_predictions() -> pd.DataFrame:
    path = config.REPORTS_DIR / "test_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    thr = predict.decision_threshold()
    df["severity"] = df["attack_probability"].apply(lambda p: predict.severity(float(p), thr))
    df["predicted"] = df["y_pred_tuned"].map({0: "Normal", 1: "Attack"})
    df["actual"] = df["y_true"].map({0: "Normal", 1: "Attack"})
    df["attack_cat_pred"] = df.get("attack_cat_pred", "n/a")
    return df


def _img_b64(name: str) -> str:
    path = figure_path(name)
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


def build_html_report() -> str:
    meta = predict.load_metadata()
    metrics = load_metrics()
    mc = predict.load_multiclass_metadata()
    rows = "".join(
        f"<tr><td>{m}</td>"
        + "".join(f"<td>{metrics[m].get(k, float('nan')):.4f}</td>"
                  for k in ["precision", "recall", "f1", "roc_auc", "mcc", "false_positive_rate"])
        + "</tr>"
        for m in metrics
    )
    figures = "".join(
        f"<h3>{n.replace('.png','')}</h3><img src='data:image/png;base64,{_img_b64(n)}' />"
        for n in ["model_comparison.png", "roc_curves.png", "confusion_matrix_XGBoost.png",
                  "threshold_curve.png", "multiclass_confusion_matrix.png"]
        if _img_b64(n)
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>NIDS Security Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#1a202c}}
table{{border-collapse:collapse;margin:12px 0}}th,td{{border:1px solid #cbd5e0;padding:6px 10px}}
th{{background:#2d3748;color:#fff}}img{{max-width:720px;border:1px solid #e2e8f0;margin-bottom:16px}}
</style></head><body>
<h1>Network Intrusion Detection Report</h1>
<p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · dataset {meta.get('dataset')} ·
best model <b>{meta.get('best_model')}</b> · threshold {meta.get('decision_threshold')}</p>
<h2>Binary metrics (test set)</h2>
<table><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>MCC</th><th>FPR</th></tr>{rows}</table>
<h2>Attack categories</h2>
<p>{', '.join(mc.get('classes', [])) or 'n/a'}</p>
<h2>Figures</h2>{figures}
</body></html>"""


def api_predict(url: str, record: dict) -> dict:
    resp = requests.post(f"{url.rstrip('/')}/predict", json=record, timeout=15)
    resp.raise_for_status()
    return resp.json()


def find_csv_files(folder: str, recursive: bool = False) -> list[Path]:
    """List CSV files in a local folder, sorted by name."""
    path = Path(folder).expanduser()
    if not path.is_dir():
        return []
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(p for p in path.glob(pattern) if p.is_file())


def find_capture_files(folder: str, recursive: bool = False) -> list[Path]:
    """List .pcap/.pcapng files in a local folder, sorted by name."""
    path = Path(folder).expanduser()
    if not path.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower() in {".pcap", ".pcapng"}
    )


def score_batch(raw: pd.DataFrame, use_api: bool, url: str) -> pd.DataFrame:
    if not use_api:
        return predict.predict_dataframe(raw)
    payload = {"records": raw[config.ALL_FEATURE_COLS].to_dict("records")}
    resp = requests.post(f"{url.rstrip('/')}/predict/batch", json=payload, timeout=60)
    resp.raise_for_status()
    preds = resp.json()["predictions"]
    df = raw.copy()
    df["verdict"] = [p["verdict"] for p in preds]
    df["severity"] = [p["severity"] for p in preds]
    df["attack_probability"] = [p["attack_probability"] for p in preds]
    return df


def render_batch_results(df: pd.DataFrame) -> None:
    total = len(df)
    attacks = int((df["verdict"] == "Attack").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Flows analyzed", f"{total:,}")
    c2.metric("Attacks detected", f"{attacks:,}")
    c3.metric("Attack rate", f"{attacks / max(total, 1):.1%}")

    st.bar_chart(df["severity"].value_counts())
    st.dataframe(df, width="stretch")
    st.download_button(
        "Download results",
        df.to_csv(index=False).encode("utf-8"),
        file_name="nids_predictions.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ NIDS Control")
st.sidebar.caption("UNSW-NB15 · Random Forest + XGBoost")

use_api = st.sidebar.toggle("Use FastAPI service", value=False)
api_url = st.sidebar.text_input("API URL", os.environ.get("NIDS_API_URL", "http://localhost:8000"))
model_ready = predict.model_ready()

if use_api:
    try:
        requests.get(f"{api_url.rstrip('/')}/health", timeout=3).raise_for_status()
        st.sidebar.success("API reachable")
    except Exception:
        st.sidebar.error("API not reachable")
elif model_ready:
    st.sidebar.success("Local model loaded")
else:
    st.sidebar.warning("No model found. Run scripts/run_pipeline.py")

st.sidebar.divider()
st.sidebar.markdown(
    "**Pipeline**\n\n"
    "1. Preprocess\n2. Feature select\n3. Train\n4. Evaluate\n5. Predict"
)

# ---------------------------------------------------------------- header
st.title("Network Intrusion Detection Dashboard")
st.caption("Detect suspicious network traffic and classify it as normal or malicious.")

tab_overview, tab_single, tab_batch, tab_pcap, tab_monitor, tab_reports = st.tabs(
    ["Overview", "Single Prediction", "Batch Analysis", "PCAP Analysis", "Monitoring", "Model Reports"]
)

# ---------------------------------------------------------------- overview
with tab_overview:
    meta = predict.load_metadata()
    metrics = load_metrics()

    c1, c2, c3, c4 = st.columns(4)
    n_test = meta.get("test_rows", 0)
    best = meta.get("best_model", "n/a")
    best_f1 = metrics.get(best, {}).get("f1")
    c1.metric("Dataset", meta.get("dataset", "UNSW-NB15"))
    c2.metric("Best model", best)
    c3.metric("Decision threshold", f"{meta.get('decision_threshold', 0.5):.2f}")
    c4.metric("Best F1", f"{best_f1:.4f}" if best_f1 else "n/a")

    st.caption(
        f"Attack categories: {', '.join(predict.load_multiclass_metadata().get('classes', [])) or 'n/a'}"
    )

    if metrics:
        st.subheader("Model comparison (official test set)")
        df = pd.DataFrame(metrics).T[
            ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "threshold"]
        ]
        st.dataframe(df.style.format("{:.4f}"), width="stretch")

        left, right = st.columns(2)
        if figure_path("model_comparison.png").exists():
            left.image(str(figure_path("model_comparison.png")), caption="Model comparison")
        if figure_path("roc_curves.png").exists():
            right.image(str(figure_path("roc_curves.png")), caption="ROC curves")

        st.subheader("Per-model drill-down")
        chosen = st.selectbox("Model", list(metrics.keys()), index=len(metrics) - 1)
        mc1, mc2 = st.columns([1, 1])
        with mc1:
            cm = figure_path(f"confusion_matrix_{chosen}.png")
            if cm.exists():
                st.image(str(cm), caption=f"Confusion matrix — {chosen}")
        with mc2:
            st.dataframe(
                pd.Series(metrics[chosen], name=chosen).to_frame().style.format("{:.4f}"),
                width="stretch",
            )

        st.download_button(
            "Download HTML security report",
            build_html_report().encode("utf-8"),
            file_name="nids_report.html",
            mime="text/html",
        )
    else:
        st.info("No metrics yet. Train the models first.")

    st.subheader("Selected features")
    selected = meta.get("selected_features", [])
    if selected:
        st.write(", ".join(f"`{f}`" for f in selected))

# ---------------------------------------------------------------- single
with tab_single:
    st.subheader("Score a single network flow")
    template = predict.example_record()

    with st.form("single_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            proto = st.selectbox("proto", PROTO_OPTIONS, index=0)
            service = st.selectbox("service", SERVICE_OPTIONS, index=0)
            state = st.selectbox("state", STATE_OPTIONS, index=0)
            dur = st.number_input("dur", value=float(template["dur"]), min_value=0.0, format="%.6f")
            spkts = st.number_input("spkts", value=float(template["spkts"]), min_value=0.0)
            dpkts = st.number_input("dpkts", value=float(template["dpkts"]), min_value=0.0)
            sbytes = st.number_input("sbytes", value=float(template["sbytes"]), min_value=0.0)
            dbytes = st.number_input("dbytes", value=float(template["dbytes"]), min_value=0.0)
            rate = st.number_input("rate", value=float(template["rate"]), min_value=0.0)
            sttl = st.number_input("sttl", value=float(template["sttl"]), min_value=0.0)
            dttl = st.number_input("dttl", value=float(template["dttl"]), min_value=0.0)
            sload = st.number_input("sload", value=float(template["sload"]), min_value=0.0)
            dload = st.number_input("dload", value=float(template["dload"]), min_value=0.0)
            sloss = st.number_input("sloss", value=float(template["sloss"]), min_value=0.0)
            dloss = st.number_input("dloss", value=float(template["dloss"]), min_value=0.0)
        with col_b:
            sinpkt = st.number_input("sinpkt", value=float(template["sinpkt"]), min_value=0.0)
            dinpkt = st.number_input("dinpkt", value=float(template["dinpkt"]), min_value=0.0)
            sjit = st.number_input("sjit", value=float(template["sjit"]), min_value=0.0)
            djit = st.number_input("djit", value=float(template["djit"]), min_value=0.0)
            swin = st.number_input("swin", value=float(template["swin"]), min_value=0.0)
            stcpb = st.number_input("stcpb", value=float(template["stcpb"]), min_value=0.0)
            dtcpb = st.number_input("dtcpb", value=float(template["dtcpb"]), min_value=0.0)
            dwin = st.number_input("dwin", value=float(template["dwin"]), min_value=0.0)
            tcprtt = st.number_input("tcprtt", value=float(template["tcprtt"]), min_value=0.0)
            synack = st.number_input("synack", value=float(template["synack"]), min_value=0.0)
            ackdat = st.number_input("ackdat", value=float(template["ackdat"]), min_value=0.0)
            smean = st.number_input("smean", value=float(template["smean"]), min_value=0.0)
            dmean = st.number_input("dmean", value=float(template["dmean"]), min_value=0.0)
        with col_c:
            trans_depth = st.number_input("trans_depth", value=float(template["trans_depth"]), min_value=0.0)
            response_body_len = st.number_input("response_body_len", value=float(template["response_body_len"]), min_value=0.0)
            ct_srv_src = st.number_input("ct_srv_src", value=float(template["ct_srv_src"]), min_value=0.0)
            ct_state_ttl = st.number_input("ct_state_ttl", value=float(template["ct_state_ttl"]), min_value=0.0)
            ct_dst_ltm = st.number_input("ct_dst_ltm", value=float(template["ct_dst_ltm"]), min_value=0.0)
            ct_src_dport_ltm = st.number_input("ct_src_dport_ltm", value=float(template["ct_src_dport_ltm"]), min_value=0.0)
            ct_dst_sport_ltm = st.number_input("ct_dst_sport_ltm", value=float(template["ct_dst_sport_ltm"]), min_value=0.0)
            ct_dst_src_ltm = st.number_input("ct_dst_src_ltm", value=float(template["ct_dst_src_ltm"]), min_value=0.0)
            is_ftp_login = st.number_input("is_ftp_login", value=float(template["is_ftp_login"]), min_value=0.0)
            ct_ftp_cmd = st.number_input("ct_ftp_cmd", value=float(template["ct_ftp_cmd"]), min_value=0.0)
            ct_flw_http_mthd = st.number_input("ct_flw_http_mthd", value=float(template["ct_flw_http_mthd"]), min_value=0.0)
            ct_src_ltm = st.number_input("ct_src_ltm", value=float(template["ct_src_ltm"]), min_value=0.0)
            ct_srv_dst = st.number_input("ct_srv_dst", value=float(template["ct_srv_dst"]), min_value=0.0)
            is_sm_ips_ports = st.number_input("is_sm_ips_ports", value=float(template["is_sm_ips_ports"]), min_value=0.0)

        submitted = st.form_submit_button("Analyze traffic", type="primary")

    if submitted:
        record = {
            "dur": dur, "spkts": spkts, "dpkts": dpkts, "sbytes": sbytes, "dbytes": dbytes,
            "rate": rate, "sttl": sttl, "dttl": dttl, "sload": sload, "dload": dload,
            "sloss": sloss, "dloss": dloss, "sinpkt": sinpkt, "dinpkt": dinpkt,
            "sjit": sjit, "djit": djit, "swin": swin, "stcpb": stcpb, "dtcpb": dtcpb,
            "dwin": dwin, "tcprtt": tcprtt, "synack": synack, "ackdat": ackdat,
            "smean": smean, "dmean": dmean, "trans_depth": trans_depth,
            "response_body_len": response_body_len, "ct_srv_src": ct_srv_src,
            "ct_state_ttl": ct_state_ttl, "ct_dst_ltm": ct_dst_ltm,
            "ct_src_dport_ltm": ct_src_dport_ltm, "ct_dst_sport_ltm": ct_dst_sport_ltm,
            "ct_dst_src_ltm": ct_dst_src_ltm, "is_ftp_login": is_ftp_login,
            "ct_ftp_cmd": ct_ftp_cmd, "ct_flw_http_mthd": ct_flw_http_mthd,
            "ct_src_ltm": ct_src_ltm, "ct_srv_dst": ct_srv_dst,
            "is_sm_ips_ports": is_sm_ips_ports, "proto": proto, "service": service,
            "state": state,
        }
        try:
            result = api_predict(api_url, record) if use_api else predict.predict_records([record])[0]
            if result["label"] == 1:
                st.error(f"🚨 {result['verdict']} — severity: **{result['severity']}**")
            else:
                st.success(f"✅ {result['verdict']} traffic")
            m1, m2, m3 = st.columns(3)
            m1.metric("Attack probability", f"{result['attack_probability']:.2%}")
            m2.metric("Confidence", f"{result['confidence']:.2%}")
            m3.metric("Attack category", result.get("attack_category") or "—")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")

# ---------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Batch analysis")
    st.caption("Score a CSV with the 42 raw UNSW-NB15 feature columns (no label required).")
    source = st.radio(
        "Data source", ["Upload a CSV", "Local folder on this PC"], horizontal=True
    )

    raw: pd.DataFrame | None = None
    source_name = ""

    if source == "Upload a CSV":
        uploaded = st.file_uploader("Traffic CSV", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            source_name = uploaded.name
    else:
        folder = st.text_input(
            "Folder path", value=str(config.DATA_RAW), placeholder=r"C:\path\to\csvs"
        )
        recursive = st.checkbox("Include subfolders", value=False)
        files = find_csv_files(folder, recursive) if folder else []
        if folder and not Path(folder).expanduser().is_dir():
            st.error(f"Folder not found: {folder}")
        elif not files:
            st.warning("No CSV files found in that folder.")
        else:
            choice = st.selectbox(
                f"Found {len(files)} CSV file(s)", files, format_func=lambda p: p.name
            )
            st.caption(str(choice))
            if st.button("Analyze selected file", type="primary"):
                raw = pd.read_csv(choice)
                source_name = choice.name

    if raw is not None:
        try:
            df = score_batch(raw, use_api, api_url)
            if source_name:
                st.caption(f"Source: {source_name}")
            render_batch_results(df)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Batch analysis failed: {exc}")

# ---------------------------------------------------------------- pcap
with tab_pcap:
    st.subheader("PCAP analysis")
    st.caption(
        "Extract flows from a packet capture (.pcap/.pcapng) with scapy and score them. "
        "For CSV feature files, use the Batch Analysis tab instead."
    )
    pcap_source = st.radio(
        "Data source",
        ["Upload a capture", "Local folder on this PC"],
        horizontal=True,
        key="pcap_source",
    )
    max_flows = st.number_input("Max flows to score", 100, 200_000, 5_000, step=100)

    pcap_bytes: bytes | None = None
    pcap_name = ""
    selected_path: Path | None = None

    if pcap_source == "Upload a capture":
        pcap_file = st.file_uploader("Packet capture", type=["pcap", "pcapng"], key="pcap")
        if pcap_file is not None:
            pcap_bytes = pcap_file.getvalue()
            pcap_name = pcap_file.name
    else:
        folder = st.text_input(
            "Folder path", value="", placeholder=r"C:\path\to\captures", key="pcap_folder"
        )
        recursive = st.checkbox("Include subfolders", value=False, key="pcap_recursive")
        files = find_capture_files(folder, recursive) if folder else []
        if folder and not Path(folder).expanduser().is_dir():
            st.error(f"Folder not found: {folder}")
        elif not files:
            st.warning("No .pcap or .pcapng files found in that folder.")
        else:
            selected_path = st.selectbox(
                f"Found {len(files)} capture file(s)",
                files,
                format_func=lambda p: p.name,
                key="pcap_choice",
            )
            st.caption(str(selected_path))

    if st.button("Analyze capture", type="primary", key="pcap_analyze"):
        if selected_path is not None:
            pcap_bytes = selected_path.read_bytes()
            pcap_name = selected_path.name

        if pcap_bytes is None:
            st.warning("Select or upload a capture file first.")
        else:
            try:
                with st.spinner("Extracting and scoring flows..."):
                    if use_api:
                        files = {"file": (pcap_name, pcap_bytes, "application/octet-stream")}
                        resp = requests.post(
                            f"{api_url.rstrip('/')}/predict/pcap",
                            files=files,
                            params={"max_flows": int(max_flows)},
                            timeout=300,
                        )
                        resp.raise_for_status()
                        summary = resp.json()
                        total, attacks = summary["flows"], summary["attacks"]
                        sev = summary["severity_counts"]
                        cats = summary["category_counts"]
                        top = pd.DataFrame(summary["top_flows"])
                    else:
                        suffix = Path(pcap_name).suffix or ".pcap"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(pcap_bytes)
                            tmp_path = tmp.name
                        try:
                            raw = pcap_extractor.pcap_to_dataframe(tmp_path, max_flows=int(max_flows))
                        finally:
                            Path(tmp_path).unlink(missing_ok=True)
                        scored = predict.predict_dataframe(raw)
                        total = len(scored)
                        attacks = int((scored["verdict"] == "Attack").sum())
                        sev = scored["severity"].value_counts().to_dict()
                        cats = scored["attack_category"].value_counts().to_dict()
                        top = (
                            scored[scored["verdict"] == "Attack"]
                            .sort_values("attack_probability", ascending=False)
                            .head(10)[["proto", "service", "state", "attack_category", "attack_probability"]]
                        )

                c1, c2, c3 = st.columns(3)
                c1.metric("Flows extracted", f"{total:,}")
                c2.metric("Attacks detected", f"{attacks:,}")
                c3.metric("Attack rate", f"{attacks / max(total, 1):.1%}")

                left, right = st.columns(2)
                left.bar_chart(pd.Series(sev, name="flows"))
                right.bar_chart(pd.Series(cats, name="flows"))
                if not top.empty:
                    st.markdown("**Top suspicious flows**")
                    st.dataframe(top, width="stretch")
            except Exception as exc:  # noqa: BLE001
                st.error(f"PCAP analysis failed: {exc}")

# ---------------------------------------------------------------- monitoring
with tab_monitor:
    st.subheader("Live monitoring & alerts")
    events = load_predictions()
    if events.empty:
        st.info("No predictions yet. Train the models to generate reports/test_predictions.csv.")
    else:
        ss = st.session_state
        if "live_idx" not in ss:
            ss.live_idx = 0
        if "live_playing" not in ss:
            ss.live_playing = False

        c1, c2, c3 = st.columns([1, 1, 2])
        c1.toggle("Play live simulation", key="live_playing")
        if c2.button("Reset stream"):
            ss.live_idx = 0
        batch = c3.slider("Events per tick", 5, 100, 25, step=5)

        @st.fragment(run_every=2.0)
        def feed():
            if ss.live_playing:
                ss.live_idx = min(ss.live_idx + batch, len(events))
            start = max(0, ss.live_idx - 150)
            window = events.iloc[start:ss.live_idx]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Streamed", f"{ss.live_idx:,}/{len(events):,}")
            m2.metric("Attacks in window", int((window["predicted"] == "Attack").sum()))
            m3.metric("Malicious", int((window["severity"] == "Malicious").sum()))
            m4.metric("Window size", len(window))
            st.dataframe(
                window[["actual", "predicted", "severity", "attack_cat",
                        "attack_cat_pred", "attack_probability"]].tail(30),
                width="stretch",
            )

        feed()

        st.divider()
        st.subheader("Alert explorer")
        f1, f2, f3, f4 = st.columns(4)
        sev = f1.multiselect("Severity", ["Malicious", "Suspicious", "Normal"],
                             default=["Malicious", "Suspicious"])
        cats = f2.multiselect("Predicted category", sorted(events["attack_cat_pred"].dropna().unique()))
        actual = f3.radio("Actual", ["All", "Attack", "Normal"], horizontal=True)
        min_prob = f4.slider("Min probability", 0.0, 1.0, 0.5, 0.01)

        view = events[events["severity"].isin(sev) & (events["attack_probability"] >= min_prob)]
        if cats:
            view = view[view["attack_cat_pred"].isin(cats)]
        if actual != "All":
            view = view[view["actual"] == actual]

        k1, k2, k3 = st.columns(3)
        k1.metric("Matching alerts", f"{len(view):,}")
        k2.metric("Precision (filtered)",
                  f"{view['y_true'].mean() if len(view) else 0:.2%}")
        k3.metric("Mean probability", f"{view['attack_probability'].mean() if len(view) else 0:.3f}")
        st.dataframe(
            view[["actual", "predicted", "severity", "attack_cat", "attack_cat_pred",
                  "attack_probability"]].head(1000),
            width="stretch",
        )
        st.download_button(
            "Download filtered alerts",
            view.to_csv(index=False).encode("utf-8"),
            file_name="nids_alerts.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------- reports
with tab_reports:
    st.subheader("Evaluation artifacts")
    names = [
        "confusion_matrix_XGBoost.png",
        "confusion_matrix_RandomForest.png",
        "roc_curves.png",
        "threshold_curve.png",
        "model_comparison.png",
        "multiclass_confusion_matrix.png",
        "multiclass_comparison.png",
        "feature_importance.png",
        "correlation_heatmap.png",
        "class_distribution.png",
    ]
    available = [n for n in names if figure_path(n).exists()]
    if not available:
        st.info("No figures found. Run the training pipeline first.")
    else:
        cols = st.columns(2)
        for i, name in enumerate(available):
            cols[i % 2].image(str(figure_path(name)), caption=name.replace(".png", ""), width="stretch")
