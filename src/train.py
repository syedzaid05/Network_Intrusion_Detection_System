"""End-to-end training pipeline: preprocess -> select -> train -> evaluate.

Trains two model families:
  * binary  : Normal vs Attack                (models/ids_pipeline.joblib)
  * multiclass: attack category + Normal      (models/ids_multiclass.joblib)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config, data_loader, evaluate, feature_selection, preprocessing


def _make_pipeline(preprocessor, selector, estimator) -> Pipeline:
    return Pipeline(
        [
            ("prep", clone(preprocessor)),
            ("select", clone(selector)),
            ("clf", estimator),
        ]
    )


def _binary_estimators(cfg: config.Config, y_train: pd.Series) -> dict:
    rf = dict(cfg.rf_params)
    if cfg.use_class_weight:
        rf["class_weight"] = "balanced"

    xgb = dict(cfg.xgb_params)
    xgb.setdefault("eval_metric", "logloss")
    if cfg.use_class_weight:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        xgb["scale_pos_weight"] = round(neg / max(pos, 1), 3)

    return {
        "RandomForest": RandomForestClassifier(random_state=cfg.random_state, **rf),
        "XGBoost": XGBClassifier(random_state=cfg.random_state, **xgb),
    }


def _maybe_calibrate(base_pipeline, X_val, y_val, cfg):
    """Wrap a fitted pipeline in sigmoid/isotonic calibration (sklearn >= 1.6)."""
    if not cfg.calibrate:
        return base_pipeline
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(
            FrozenEstimator(base_pipeline), method=cfg.calibration_method
        )
        calibrated.fit(X_val, y_val)
        return calibrated
    except Exception as exc:  # noqa: BLE001
        print(f"[train] calibration unavailable ({exc}); using uncalibrated model")
        return base_pipeline


def train_binary(preprocessor, selector, X_train, y_train, X_test, y_test, cfg):
    """Fit both binary models, calibrate probabilities, tune a decision threshold."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2,
        random_state=cfg.random_state, stratify=y_train,
    )

    metrics: dict[str, dict] = {}
    fitted: dict = {}
    thresholds: dict[str, float] = {}
    roc_data: dict = {}
    analysis: dict[str, dict] = {}

    for name, estimator in _binary_estimators(cfg, y_train).items():
        print(f"\n[train] fitting binary {name}...")
        t0 = time.perf_counter()
        base = _make_pipeline(preprocessor, selector, estimator)
        base.fit(X_fit, y_fit)
        model = _maybe_calibrate(base, X_val, y_val, cfg)
        fit_seconds = time.perf_counter() - t0

        val_proba = model.predict_proba(X_val)[:, 1]
        if cfg.tune_threshold:
            threshold, val_score = evaluate.best_threshold(
                y_val, val_proba, cfg.threshold_objective,
                cfg.min_recall, cfg.fp_cost, cfg.fn_cost,
            )
            f1_threshold, f1_val = evaluate.best_threshold(y_val, val_proba, "f1")
            print(f"[train] {name}: tuned threshold={threshold:.3f} "
                  f"({cfg.threshold_objective} val={val_score:.4f}; f1@thr={f1_val:.4f})")
        else:
            threshold = f1_threshold = 0.5

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        scores = evaluate.compute_metrics(y_test, pred, proba)
        scores["threshold"] = round(float(threshold), 4)
        scores["f1_optimal_threshold"] = round(float(f1_threshold), 4)
        scores["fit_seconds"] = round(fit_seconds, 2)
        metrics[name] = scores
        fitted[name] = model
        thresholds[name] = float(threshold)

        at_f1 = evaluate.compute_metrics(y_test, (proba >= f1_threshold).astype(int), proba)
        analysis[name] = {
            "objective": cfg.threshold_objective,
            "chosen_threshold": round(float(threshold), 4),
            "precision@chosen": round(scores["precision"], 4),
            "recall@chosen": round(scores["recall"], 4),
            "false_positive_rate@chosen": round(scores["false_positive_rate"], 4),
            "f1_optimal_threshold": round(float(f1_threshold), 4),
            "precision@f1": round(at_f1["precision"], 4),
            "recall@f1": round(at_f1["recall"], 4),
            "false_positive_rate@f1": round(at_f1["false_positive_rate"], 4),
            "fp_reduction": round(at_f1["false_positive"] - scores["false_positive"], 1),
        }
        print(f"[train] {name}: acc={scores['accuracy']:.4f} prec={scores['precision']:.4f} "
              f"rec={scores['recall']:.4f} f1={scores['f1']:.4f} auc={scores['roc_auc']:.4f} "
              f"mcc={scores['mcc']:.4f} fpr={scores['false_positive_rate']:.4f} "
              f"({fit_seconds:.1f}s)")
        evaluate.save_confusion_matrix(y_test, pred, name)
        evaluate.save_threshold_curve(y_test, proba, threshold)
        roc_data[name] = evaluate.roc_points(model, X_test, y_test)

    evaluate.save_roc_curve(roc_data)
    comparison = evaluate.save_comparison(metrics)
    (config.REPORTS_DIR / "threshold_analysis.json").write_text(json.dumps(analysis, indent=2))
    return metrics, fitted, thresholds, comparison


def train_multiclass(preprocessor, selector, X_train, y_cat_train, X_test, y_cat_test, cfg):
    """Fit attack-category classifiers (includes the Normal class)."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y_tr = encoder.fit_transform(y_cat_train)
    y_te = encoder.transform(y_cat_test)
    classes = list(encoder.classes_)
    print(f"\n[train] multiclass categories ({len(classes)}): {classes}")

    rf = dict(cfg.mc_rf_params)
    if cfg.use_class_weight:
        rf["class_weight"] = "balanced"
    estimators = {
        "RandomForest": RandomForestClassifier(random_state=cfg.random_state, **rf),
        "XGBoost": XGBClassifier(
            random_state=cfg.random_state, eval_metric="mlogloss", **cfg.mc_xgb_params
        ),
    }

    metrics: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in estimators.items():
        print(f"[train] fitting multiclass {name}...")
        t0 = time.perf_counter()
        pipeline = _make_pipeline(preprocessor, selector, estimator)
        pipeline.fit(X_train, y_tr)
        fit_seconds = time.perf_counter() - t0

        pred = pipeline.predict(X_test)
        scores = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_te, pred, average="weighted", zero_division=0)),
            "fit_seconds": round(fit_seconds, 2),
        }
        metrics[name] = scores
        fitted[name] = pipeline
        print(f"[train] {name}: acc={scores['accuracy']:.4f} "
              f"macroF1={scores['macro_f1']:.4f} ({fit_seconds:.1f}s)")

    best_name = max(metrics, key=lambda k: metrics[k]["macro_f1"])
    best_pipeline = fitted[best_name]
    evaluate.save_multiclass_confusion_matrix(
        y_te, best_pipeline.predict(X_test), classes, best_name
    )
    comparison = evaluate.save_multiclass_comparison(metrics)
    return best_name, best_pipeline, classes, metrics, comparison


def run(cfg: config.Config = config.CONFIG) -> dict:
    started = time.perf_counter()
    print("=" * 70)
    print("Network Intrusion Detection System - training pipeline")
    print("=" * 70)

    train_raw, test_raw = data_loader.load_raw()
    train = preprocessing.clean_frame(train_raw)
    test = preprocessing.clean_frame(test_raw)

    X_train, y_train, attack_train = preprocessing.split_xy(train)
    X_test, y_test, attack_test = preprocessing.split_xy(test)
    print(f"[data] features={X_train.shape[1]} train={X_train.shape[0]} test={X_test.shape[0]}")
    print(f"[data] train balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")

    evaluate.save_target_distribution(y_train)

    preprocessor = preprocessing.build_preprocessor()
    selection = feature_selection.select_features(preprocessor, X_train, y_train, cfg)
    selector = preprocessing.FeatureSelector(selection["selected_indices"])
    selection["importance"].to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
    feature_selection.profile_features(
        selection["sample_matrix"], np.asarray(selection["sample_names"])
    ).to_csv(config.REPORTS_DIR / "top_correlations.csv", index=False)
    evaluate.save_importance_plot(selection["importance"])
    evaluate.save_correlation_heatmap(
        selection["sample_matrix"], np.asarray(selection["sample_names"])
    )

    # ---- binary ---------------------------------------------------------
    metrics, fitted, thresholds, comparison = train_binary(
        preprocessor, selector, X_train, y_train, X_test, y_test, cfg
    )
    best_name = max(metrics, key=lambda k: metrics[k]["f1"])
    best_pipeline = fitted[best_name]
    best_threshold = thresholds[best_name]
    joblib.dump(best_pipeline, config.MODEL_PATH)
    print(f"\n[model] best binary = {best_name} (F1={metrics[best_name]['f1']:.4f}, "
          f"threshold={best_threshold:.3f}) -> {config.MODEL_PATH}")

    # ---- multiclass -----------------------------------------------------
    mc_name, mc_pipeline, mc_classes = None, None, []
    mc_metrics: dict = {}
    if attack_train is not None:
        mc_name, mc_pipeline, mc_classes, mc_metrics, _ = train_multiclass(
            preprocessor, selector, X_train, attack_train, X_test, attack_test, cfg
        )
        joblib.dump(mc_pipeline, config.MULTICLASS_MODEL_PATH)
        print(f"[model] best multiclass = {mc_name} "
              f"(macroF1={mc_metrics[mc_name]['macro_f1']:.4f}) -> {config.MULTICLASS_MODEL_PATH}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "decision_threshold": round(float(best_threshold), 4),
        "threshold_objective": cfg.threshold_objective,
        "min_recall": cfg.min_recall,
        "calibrated": cfg.calibrate,
        "dataset": "UNSW-NB15",
        "n_features_input": len(config.ALL_FEATURE_COLS),
        "input_features": config.ALL_FEATURE_COLS,
        "numeric_features": config.NUMERIC_COLS,
        "categorical_features": config.CATEGORICAL_COLS,
        "selected_features": selection["selected_names"],
        "n_selected": len(selection["selected_names"]),
        "dropped_correlated": selection["dropped_correlated"],
        "class_weighted": cfg.use_class_weight,
        "metrics": metrics,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    if mc_pipeline is not None:
        config.MULTICLASS_METADATA_PATH.write_text(
            json.dumps(
                {
                    "best_model": mc_name,
                    "classes": mc_classes,
                    "metrics": mc_metrics,
                    "trained_at": metadata["trained_at"],
                },
                indent=2,
            )
        )

    preds = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": best_pipeline.predict(X_test),
            "y_pred_tuned": (best_pipeline.predict_proba(X_test)[:, 1] >= best_threshold).astype(int),
            "attack_probability": best_pipeline.predict_proba(X_test)[:, 1].round(6),
            "attack_cat": (attack_test.to_numpy() if attack_test is not None else ""),
        }
    )
    if mc_pipeline is not None:
        preds["attack_cat_pred"] = [mc_classes[i] for i in mc_pipeline.predict(X_test)]
    preds.to_csv(config.REPORTS_DIR / "test_predictions.csv", index=False)

    print(f"\n[done] total time {time.perf_counter() - started:.1f}s")
    print(f"[done] reports -> {config.REPORTS_DIR}")
    return {"binary": comparison, "multiclass": mc_metrics}


if __name__ == "__main__":
    run()
