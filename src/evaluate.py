"""Model evaluation metrics and figures."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config  # noqa: E402

sns.set_theme(style="whitegrid")


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "false_positive_rate": float(fpr),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def _confusion_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp


def best_threshold(
    y_true,
    y_proba,
    objective: str = "f1",
    min_recall: float = 0.90,
    fp_cost: float = 1.0,
    fn_cost: float = 3.0,
) -> tuple[float, float]:
    """Sweep thresholds and return the (threshold, score) that best meets ``objective``.

    * ``f1``                  - maximise F1.
    * ``cost``                - minimise ``fp_cost*FP + fn_cost*FN``.
    * ``precision_at_recall`` - maximise precision subject to ``recall >= min_recall``
      (raises the threshold to reduce false positives).
    """
    y_true = np.asarray(y_true)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 3)
    best_t, best_s = 0.5, -np.inf
    for t in grid:
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = _confusion_counts(y_true, pred)
        if objective == "cost":
            score = -(fp_cost * fp + fn_cost * fn)
        elif objective == "precision_at_recall":
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            score = precision if recall >= min_recall else -1.0
        else:  # f1
            score = f1_score(y_true, pred, zero_division=0)
        if score >= best_s:  # ties prefer the higher (more conservative) threshold
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def save_confusion_matrix(y_true, y_pred, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
        xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=130)
    plt.close(fig)


def save_roc_curve(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=130)
    plt.close(fig)


def roc_points(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    return fpr, tpr, float(roc_auc_score(y_test, proba))


def save_importance_plot(importance: pd.DataFrame, top_n: int = 20) -> None:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    sns.barplot(data=top, y="feature", x="mutual_info", color="#2b6cb0", ax=ax)
    ax.set_title(f"Top {top_n} Features by Mutual Information")
    ax.set_xlabel("Mutual Information")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)


def save_correlation_heatmap(matrix: np.ndarray, names: np.ndarray, top_n: int = 20) -> None:
    frame = pd.DataFrame(matrix, columns=names)
    variances = frame.var().sort_values(ascending=False).head(top_n).index
    corr = frame[variances].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Correlation Heatmap - Top {top_n} Most Variable Features")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "correlation_heatmap.png", dpi=130)
    plt.close(fig)


def save_target_distribution(y: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    y.map({0: "Normal", 1: "Attack"}).value_counts().plot.bar(
        color=["#38a169", "#e53e3e"], ax=ax
    )
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    for label in ax.containers:
        ax.bar_label(label, fmt="%,d")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "class_distribution.png", dpi=130)
    plt.close(fig)


def save_comparison(metrics: dict[str, dict]) -> None:
    rows = [{"model": name, **vals} for name, vals in metrics.items()]
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(config.METRICS_CSV)
    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2))

    plot_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ax = df[plot_cols].plot.bar(figsize=(8, 5), rot=0, colormap="viridis")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison (Test Set)")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", ncol=3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=1)
    ax.figure.tight_layout()
    ax.figure.savefig(config.FIGURES_DIR / "model_comparison.png", dpi=130)
    plt.close(ax.figure)
    return df


def save_threshold_curve(y_true, y_proba, chosen: float) -> None:
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true, pred, zero_division=0))
        precs.append(precision_score(y_true, pred, zero_division=0))
        recs.append(recall_score(y_true, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, f1s, label="F1")
    ax.plot(thresholds, precs, label="Precision")
    ax.plot(thresholds, recs, label="Recall")
    ax.axvline(chosen, color="k", ls="--", lw=1, label=f"chosen={chosen:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "threshold_curve.png", dpi=130)
    plt.close(fig)


def save_multiclass_confusion_matrix(y_true, y_pred, labels, name: str = "XGBoost") -> pd.DataFrame:
    label_ids = list(range(len(labels)))
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", cbar=True,
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Attack Category Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "multiclass_confusion_matrix.png", dpi=130)
    plt.close(fig)

    report = classification_report(
        y_true, y_pred, labels=label_ids, target_names=labels,
        output_dict=True, zero_division=0,
    )
    frame = pd.DataFrame(report).T
    frame.to_csv(config.REPORTS_DIR / "multiclass_report.csv")
    return frame


def save_multiclass_comparison(metrics: dict[str, dict]) -> pd.DataFrame:
    rows = [{"model": name, **{k: v for k, v in vals.items() if not isinstance(v, dict)}}
            for name, vals in metrics.items()]
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(config.MULTICLASS_METRICS_CSV)
    config.MULTICLASS_METRICS_JSON.write_text(json.dumps(metrics, indent=2))

    plot_cols = ["accuracy", "macro_f1", "weighted_f1"]
    ax = df[plot_cols].plot.bar(figsize=(7, 4.5), rot=0, colormap="magma")
    ax.set_ylim(0, 1.05)
    ax.set_title("Attack Category Model Comparison")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=1)
    ax.figure.tight_layout()
    ax.figure.savefig(config.FIGURES_DIR / "multiclass_comparison.png", dpi=130)
    plt.close(ax.figure)
    return df
