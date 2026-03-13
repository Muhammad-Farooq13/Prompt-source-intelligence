"""
src/models/evaluator.py
───────────────────────
Computes comprehensive evaluation metrics for all trained models and
packages everything into the model bundle that the dashboard consumes.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURE_CONFIG, PATHS

logger = logging.getLogger(__name__)


# ── Metric computation ────────────────────────────────────────────────────────

def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_names: List[str],
) -> Dict[str, Any]:
    """Return a dict of classification metrics for one model."""
    y_pred  = pipeline.predict(X_test)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)
    except AttributeError:
        pass

    metrics = {
        "accuracy":          float(accuracy_score(y_test, y_pred)),
        "f1_weighted":       float(f1_score(y_test, y_pred, average="weighted")),
        "f1_macro":          float(f1_score(y_test, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted":   float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=label_names, output_dict=True
        ),
    }

    if y_proba is not None and len(label_names) > 2:
        lb = LabelBinarizer().fit(y_test)
        y_bin = lb.transform(y_test)
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
            )
        except ValueError:
            pass

    return metrics


def compute_confusion_matrices(
    results: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, np.ndarray]:
    cms = {}
    for name, spec in results.items():
        y_pred = spec["pipeline"].predict(X_test)
        cms[name] = confusion_matrix(y_test, y_pred)
    return cms


def extract_feature_importance(
    results: Dict[str, Any],
    feature_names: List[str],
    top_n: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Extract feature importance for supported model types.
    Returns a dict of DataFrames with columns [feature, importance].
    """
    importances = {}
    for name, spec in results.items():
        pipeline = spec["pipeline"]
        clf = pipeline.named_steps.get("clf")
        if clf is None:
            continue

        try:
            if hasattr(clf, "feature_importances_"):
                imp = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                coef = clf.coef_
                imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef[0])
            elif hasattr(clf, "estimator") and hasattr(clf.estimator, "coef_"):
                # CalibratedClassifierCV wrapping LinearSVC
                coefs = [c.coef_ for c in clf.calibrated_classifiers_]
                imp = np.abs(np.vstack(coefs)).mean(axis=0)
            else:
                continue

            imp_len = min(len(imp), len(feature_names))
            df_imp = pd.DataFrame({
                "feature":    feature_names[:imp_len],
                "importance": imp[:imp_len],
            }).sort_values("importance", ascending=False).head(top_n)

            importances[name] = df_imp.reset_index(drop=True)
            logger.info("Extracted top %d features for %s", top_n, name)

        except Exception as exc:
            logger.warning("Could not extract importance for %s: %s", name, exc)

    return importances


# ── Bundle builder ────────────────────────────────────────────────────────────

def build_bundle(
    results: Dict[str, Any],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_encoder: LabelEncoder,
    feature_names: List[str],
    config: dict,
) -> Dict[str, Any]:
    """
    Evaluate all models on the test set and assemble the model bundle that
    will be persisted and consumed by the Streamlit dashboard.
    """
    label_names = list(label_encoder.classes_)
    X_test = df_test
    y_test = df_test["target"].values

    logger.info("Evaluating on test set …")
    all_metrics = {}
    for name, spec in results.items():
        logger.info("  → %s", name)
        all_metrics[name] = evaluate_model(
            spec["pipeline"], X_test, y_test, label_names
        )

    best_model_name = max(
        all_metrics, key=lambda n: all_metrics[n]["f1_weighted"]
    )
    logger.info("Best model: %s  (test F1=%.4f)", best_model_name,
                all_metrics[best_model_name]["f1_weighted"])

    cms = compute_confusion_matrices(results, X_test, y_test)

    importances = extract_feature_importance(
        results, feature_names
    )

    # Sample DF for the dashboard analytics tab
    sample_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    keep_cols = [
        "id", "question", "response", "source_label", "complexity_label",
        "question_clean", "label_name", "target",
    ]
    keep_cols = [c for c in keep_cols if c in sample_df.columns]
    sample_df = sample_df[keep_cols].sample(
        n=min(8000, len(sample_df)), random_state=42
    ).reset_index(drop=True)

    # Aggregate analytics
    analytics = {
        "class_distribution": sample_df["label_name"].value_counts().to_dict(),
        "complexity_distribution": (
            sample_df["complexity_label"].value_counts().to_dict()
            if "complexity_label" in sample_df.columns else {}
        ),
        "avg_question_len": float(sample_df["question"].str.len().mean()),
        "total_samples": len(sample_df),
        "train_size": len(df_train),
        "val_size":   len(df_val),
        "test_size":  len(df_test),
    }

    import datetime
    bundle = {
        "models": {n: s["pipeline"] for n, s in results.items()},
        "best_model_name": best_model_name,
        "best_model": results[best_model_name]["pipeline"],
        "label_encoder": label_encoder,
        "label_names": label_names,
        "metrics": all_metrics,
        "cv_results": {n: s.get("cv_scores", []) for n, s in results.items()},
        "cv_scores": {n: s.get("cv_score", 0.0) for n, s in results.items()},
        "val_scores": {n: s.get("val_score", 0.0) for n, s in results.items()},
        "confusion_matrices": cms,
        "feature_importance": importances,
        "feature_names": feature_names,
        "sample_df": sample_df,
        "analytics": analytics,
        "training_date": datetime.datetime.now().isoformat(),
        "config": config,
    }

    out_path = Path(PATHS["model_bundle"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path, compress=3)
    logger.info("Model bundle saved → %s  (%.1f MB)",
                out_path, out_path.stat().st_size / 1e6)

    return bundle
