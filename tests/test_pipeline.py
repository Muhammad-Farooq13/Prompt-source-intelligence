"""
tests/test_pipeline.py
──────────────────────
Unit and integration tests for the preprocessing, feature engineering,
and prediction components.

Run with:  pytest tests/ -v --cov=src
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import _extract_complexity, _extract_source
from src.data.preprocessor import clean_text, preprocess, split
from src.features.engineer import (
    DataFrameToRecords,
    StatFeatureExtractor,
    build_feature_pipeline,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_df():
    """Minimal in-memory DataFrame mimicking the raw loader output."""
    rows = [
        {
            "id": "flan.1",
            "system_prompt": "You are a helpful assistant.",
            "question": "What is the capital of France?",
            "response": "Paris.",
            "source_label": "FLAN",
            "complexity_label": "SHORT",
        },
        {
            "id": "cot.22",
            "system_prompt": "",
            "question": "If Alice has 5 apples and gives 2 to Bob, how many does she have left?",
            "response": "Alice starts with 5 apples. She gives 2 to Bob. 5 - 2 = 3. Alice has 3 apples.",
            "source_label": "CoT",
            "complexity_label": "SHORT",
        },
        {
            "id": "t0.333",
            "system_prompt": "Summarize the following.",
            "question": "Summarize: The quick brown fox jumps over the lazy dog.",
            "response": "A fox jumps over a dog.",
            "source_label": "T0",
            "complexity_label": "SHORT",
        },
        {
            "id": "niv.4444",
            "system_prompt": "",
            "question": "Classify the sentiment of this review: I loved the movie!",
            "response": "Positive.",
            "source_label": "NIV",
            "complexity_label": "SHORT",
        },
    ] * 30   # 120 rows across 4 classes
    return pd.DataFrame(rows).reset_index(drop=True)


# ── Label extraction ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("id_str, expected_label", [
    ("flan.1",     "FLAN"),
    ("FLAN.2",     "FLAN"),
    ("cot.123",    "CoT"),
    ("t0.45",      "T0"),
    ("niv.789",    "NIV"),
    ("sharegpt.1", "ShareGPT"),
    ("cod.55",     "CoD"),
    ("unknown.1",  "UNKNOWN"),
])
def test_extract_source(id_str, expected_label):
    assert _extract_source(id_str) == expected_label


@pytest.mark.parametrize("response, expected", [
    ("yes",                              "SHORT"),     # 1 word
    ("This is a medium length response " * 10, "MEDIUM"),  # ~50 words
    ("detail " * 250,                   "LONG"),      # >200 words
])
def test_extract_complexity(response, expected):
    assert _extract_complexity(response) == expected


# ── Text cleaning ─────────────────────────────────────────────────────────────

def test_clean_removes_html():
    dirty = "<p>Hello <b>world</b></p>"
    assert "<" not in clean_text(dirty)
    assert "Hello" in clean_text(dirty)


def test_clean_replaces_url():
    dirty = "Visit https://example.com for more info."
    cleaned = clean_text(dirty)
    assert "https" not in cleaned
    assert "URL" in cleaned


def test_clean_collapses_whitespace():
    dirty = "too   many   spaces\t\there"
    assert "  " not in clean_text(dirty)


# ── Preprocessor ─────────────────────────────────────────────────────────────

def test_preprocess_adds_columns(tiny_df):
    df_out, le = preprocess(tiny_df, target="source_label")
    assert "question_clean" in df_out.columns
    assert "target" in df_out.columns
    assert "label_name" in df_out.columns
    assert len(le.classes_) > 1


def test_label_encoder_round_trip(tiny_df):
    df_out, le = preprocess(tiny_df, target="source_label")
    labels = df_out["label_name"].unique()
    for lbl in labels:
        idx = le.transform([lbl])[0]
        assert le.inverse_transform([idx])[0] == lbl


def test_split_sizes(tiny_df):
    df_pre, _ = preprocess(tiny_df, target="source_label")
    df_train, df_val, df_test = split(df_pre, test_size=0.15, val_size=0.15)
    total = len(df_train) + len(df_val) + len(df_test)
    assert total == len(df_pre)
    assert len(df_test) > 0
    assert len(df_val) > 0
    assert len(df_train) > len(df_test)


# ── Feature engineering ───────────────────────────────────────────────────────

def test_stat_feature_extractor_shape(tiny_df):
    df_pre, _ = preprocess(tiny_df, target="source_label")
    records = DataFrameToRecords().transform(df_pre)
    features = StatFeatureExtractor().transform(records)
    assert features.shape == (len(df_pre), 12)   # 12 stat features


def test_stat_features_non_negative(tiny_df):
    df_pre, _ = preprocess(tiny_df, target="source_label")
    records = DataFrameToRecords().transform(df_pre)
    features = StatFeatureExtractor().transform(records)
    # All stat features should be >= 0
    assert (features >= 0).all()


def test_feature_pipeline_fits_and_transforms(tiny_df):
    df_pre, _ = preprocess(tiny_df, target="source_label")
    pipe = build_feature_pipeline()
    X = pipe.fit_transform(df_pre)
    assert X.shape[0] == len(df_pre)
    assert X.shape[1] > 12   # TF-IDF + stat features


def test_feature_pipeline_consistent_transform_shape(tiny_df):
    df_pre, _ = preprocess(tiny_df, target="source_label")
    df_train, _, df_test = split(df_pre, test_size=0.15, val_size=0.15)
    pipe = build_feature_pipeline()
    pipe.fit(df_train)
    X_train = pipe.transform(df_train)
    X_test  = pipe.transform(df_test)
    assert X_train.shape[1] == X_test.shape[1]


# ── End-to-end mini training ──────────────────────────────────────────────────

def test_end_to_end_training(tiny_df):
    """Smoke test: the full pipeline should fit and predict without errors."""
    df_pre, le = preprocess(tiny_df, target="source_label")
    df_train, df_val, df_test = split(df_pre, test_size=0.15, val_size=0.15)

    feat_pipe = build_feature_pipeline()
    full_pipe  = Pipeline([
        ("features", feat_pipe.named_steps["features"]),
        ("clf",      LogisticRegression(max_iter=500)),
    ])
    full_pipe.fit(df_train, df_train["target"])
    preds = full_pipe.predict(df_test)

    assert len(preds) == len(df_test)
    assert set(preds).issubset(set(df_pre["target"].unique()))


# ── predict.py helpers ────────────────────────────────────────────────────────

def test_predict_single_no_bundle(monkeypatch, tmp_path):
    """predict_single raises FileNotFoundError if bundle path does not exist."""
    import predict as pred_module
    pred_module._BUNDLE_CACHE.clear()
    # Point PATHS at a guaranteed non-existent path
    monkeypatch.setitem(
        __import__("config").PATHS,
        "model_bundle",
        tmp_path / "nonexistent_bundle.pkl",
    )
    from predict import predict_single
    with pytest.raises(FileNotFoundError):
        predict_single("What is Python?", bundle=None)


# ── train_demo.py tests ───────────────────────────────────────────────────────

def test_generate_synthetic_dataset():
    """Synthetic data has the right structure."""
    from train_demo import generate_synthetic_dataset
    df = generate_synthetic_dataset(n_samples=500, random_state=0)
    assert len(df) == 500
    assert "question" in df.columns
    assert "source_label" in df.columns
    assert df["source_label"].nunique() == 5


def test_train_demo_quick(tmp_path, monkeypatch):
    """Quick demo training produces a valid bundle."""
    import train_demo as td
    # Redirect bundle paths to tmp to avoid polluting real models/
    demo_path = tmp_path / "demo_bundle.pkl"
    main_path = tmp_path / "model_bundle.pkl"
    monkeypatch.setattr(td, "BUNDLE_PATH", demo_path)
    monkeypatch.setattr(
        "src.models.evaluator.PATHS",
        {"model_bundle": main_path, "logs": tmp_path},
    )
    # Also patch PATHS used inside build_bundle
    import config as cfg
    original = cfg.PATHS["model_bundle"]
    cfg.PATHS["model_bundle"] = main_path
    try:
        bundle = td.train_demo(quick=True, rebuild=True)
    finally:
        cfg.PATHS["model_bundle"] = original

    assert "models" in bundle
    assert "metrics" in bundle
    assert bundle["best_model_name"] in bundle["models"]
    assert len(bundle["label_names"]) == 5
