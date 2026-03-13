"""
src/models/trainer.py
─────────────────────
Trains multiple classifiers via RandomizedSearchCV, builds full sklearn
Pipelines (feature extraction → classifier), and returns a ranked results
dictionary sorted by held-out weighted F1.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MODEL_CONFIG
from src.features.engineer import build_feature_pipeline

logger = logging.getLogger(__name__)

# Conditional imports for optional boosting libraries
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("xgboost not installed — XGBoost will be skipped.")

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logger.warning("lightgbm not installed — LightGBM will be skipped.")


# ── Classifier registry ───────────────────────────────────────────────────────

def _build_classifier_grid() -> Dict[str, Dict]:
    """Return a dict of {name: {"estimator": obj, "params": grid}}."""
    cfg = MODEL_CONFIG["models"]
    registry = {}

    # Logistic Regression
    registry["LogisticRegression"] = {
        "estimator": LogisticRegression(),
        "params": {f"clf__{k}": v for k, v in cfg["LogisticRegression"].items()},
    }

    # Random Forest
    registry["RandomForest"] = {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {f"clf__{k}": v for k, v in cfg["RandomForest"].items()},
    }

    # LinearSVC wrapped in CalibratedClassifierCV for predict_proba support
    registry["LinearSVC"] = {
        "estimator": CalibratedClassifierCV(LinearSVC(random_state=42)),
        "params": {
            f"clf__estimator__{k}": v
            for k, v in cfg["LinearSVC"].items()
        },
    }

    if _HAS_XGB:
        registry["XGBoost"] = {
            "estimator": XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
            "params": {f"clf__{k}": v for k, v in cfg["XGBoost"].items()},
        }

    if _HAS_LGB:
        registry["LightGBM"] = {
            "estimator": LGBMClassifier(random_state=42, n_jobs=-1),
            "params": {f"clf__{k}": v for k, v in cfg["LightGBM"].items()},
        }

    return registry


# ── Training orchestrator ─────────────────────────────────────────────────────

def train_all(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    n_iter: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train every registered model using RandomizedSearchCV on df_train,
    then evaluate on df_val.

    Returns
    -------
    dict mapping model_name → {
        "pipeline"  : fitted sklearn Pipeline,
        "best_params": dict,
        "cv_scores" : array of CV scores,
        "val_score" : float (weighted F1 on val set),
    }
    sorted best → worst by val_score.
    """
    from sklearn.metrics import f1_score

    registry = _build_classifier_grid()
    results: Dict[str, Any] = {}
    cv = StratifiedKFold(
        n_splits=MODEL_CONFIG["cv_folds"],
        shuffle=True,
        random_state=random_state,
    )

    X_train = df_train
    y_train = df_train["target"].values
    X_val   = df_val
    y_val   = df_val["target"].values

    for name, spec in registry.items():
        logger.info("─" * 60)
        logger.info("Training: %s", name)

        feat_pipeline = build_feature_pipeline()
        full_pipeline = Pipeline([
            ("features", feat_pipeline.named_steps["features"]),
            ("clf",      spec["estimator"]),
        ])

        search = RandomizedSearchCV(
            estimator=full_pipeline,
            param_distributions=spec["params"],
            n_iter=n_iter,
            cv=cv,
            scoring=MODEL_CONFIG["scoring"],
            n_jobs=MODEL_CONFIG["n_jobs"],
            random_state=random_state,
            verbose=MODEL_CONFIG["verbose"],
            refit=True,
            error_score="raise",
        )

        try:
            search.fit(X_train, y_train)
        except Exception as exc:
            logger.error("Model %s failed: %s", name, exc)
            continue

        best_pipeline = search.best_estimator_
        y_pred = best_pipeline.predict(X_val)
        val_f1 = f1_score(y_val, y_pred, average="weighted")

        logger.info(
            "%s — best CV score: %.4f  |  val F1: %.4f",
            name, search.best_score_, val_f1,
        )

        results[name] = {
            "pipeline":    best_pipeline,
            "best_params": search.best_params_,
            "cv_scores":   search.cv_results_["mean_test_score"],
            "val_score":   float(val_f1),
            "cv_score":    float(search.best_score_),
        }

    # Sort by validation F1
    results = dict(
        sorted(results.items(), key=lambda x: x[1]["val_score"], reverse=True)
    )
    logger.info(
        "Final ranking: %s",
        {k: round(v["val_score"], 4) for k, v in results.items()},
    )
    return results
