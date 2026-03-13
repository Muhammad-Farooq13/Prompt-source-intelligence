"""
Central configuration for the OpenOrca Query Intelligence project.
All paths, hyperparameters, and constants are defined here.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ── Data configuration ────────────────────────────────────────────────────────
DATA_CONFIG = {
    "dataset_name": "Open-Orca/OpenOrca",
    "sample_size": 80_000,          # rows sampled from the full dataset
    "random_state": 42,
    "test_size": 0.15,              # fraction reserved for final test
    "val_size": 0.15,               # fraction (of train) reserved for validation
    "min_question_len": 10,         # drop questions shorter than N chars
    "max_question_len": 2000,       # drop questions longer than N chars
    "min_class_samples": 500,       # drop classes with fewer samples
}

# ── Target label ──────────────────────────────────────────────────────────────
# Available targets: "source_label" | "complexity_label"
TARGET = "source_label"

# ── Feature engineering ───────────────────────────────────────────────────────
FEATURE_CONFIG = {
    "tfidf_max_features": 25_000,
    "tfidf_ngram_range": (1, 2),
    "tfidf_sublinear_tf": True,
    "tfidf_min_df": 3,
    "tfidf_max_df": 0.95,
    "stat_features": [              # hand-crafted statistical features
        "q_char_len",
        "q_word_count",
        "q_sentence_count",
        "q_avg_word_len",
        "q_unique_word_ratio",
        "q_punct_count",
        "q_digit_count",
        "q_upper_ratio",
        "q_question_mark_count",
        "q_has_code",
        "resp_word_count",
        "resp_sentence_count",
    ],
}

# ── Model hyperparameters ─────────────────────────────────────────────────────
MODEL_CONFIG = {
    "cv_folds": 5,
    "scoring": "f1_weighted",
    "n_jobs": -1,
    "verbose": 1,
    "models": {
        "LogisticRegression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
            "max_iter": [1000],
        },
        "RandomForest": {
            "n_estimators": [200, 400],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced"],
        },
        "XGBoost": {
            "n_estimators": [200, 400],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "eval_metric": ["mlogloss"],
        },
        "LightGBM": {
            "n_estimators": [300, 500],
            "max_depth": [6, 8],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 63],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "class_weight": ["balanced"],
            "verbose": [-1],
        },
        "LinearSVC": {
            "C": [0.1, 1.0, 5.0],
            "max_iter": [2000],
            "class_weight": ["balanced"],
        },
    },
}

# ── File paths ────────────────────────────────────────────────────────────────
PATHS = {
    "raw_data":         ROOT_DIR / "data" / "raw" / "openorca_sample.parquet",
    "processed_data":   ROOT_DIR / "data" / "processed" / "features.parquet",
    "model_bundle":     ROOT_DIR / "models" / "model_bundle.pkl",
    "reports":          ROOT_DIR / "reports",
    "figures":          ROOT_DIR / "reports" / "figures",
    "logs":             ROOT_DIR / "logs",
}

# Create directories if they don't exist
for p in PATHS.values():
    path = Path(p) if isinstance(p, str) else p
    if not path.suffix:          # it's a directory
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = PATHS["logs"] / "training.log"
