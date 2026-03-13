"""
src/data/preprocessor.py
────────────────────────
Cleans text, applies length filters, encodes labels, and produces
stratified train / validation / test splits.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_CONFIG, TARGET

logger = logging.getLogger(__name__)


# ── Text cleaning ─────────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE         = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE    = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Basic text cleaning: strip HTML, URLs, excess whitespace, control chars."""
    text = str(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" URL ", text)
    text = text.replace("\r", " ").replace("\t", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ── Main preprocessing ────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """
    Clean text columns, apply length filters, and encode the target label.

    Parameters
    ----------
    df     : raw DataFrame from loader.load_raw()
    target : column to use as y  ("source_label" | "complexity_label")

    Returns
    -------
    DataFrame with columns:  question_clean, target (int), label_name (str)
    plus all original columns retained.
    """
    df = df.copy()

    # ── Text cleaning
    logger.info("Cleaning question text …")
    df["question_clean"] = df["question"].apply(clean_text)
    df["response_clean"] = df["response"].apply(clean_text)

    # ── Length filters on the *original* question
    min_len = DATA_CONFIG.get("min_question_len", 10)
    max_len = DATA_CONFIG.get("max_question_len", 2000)
    mask = (
        (df["question_clean"].str.len() >= min_len) &
        (df["question_clean"].str.len() <= max_len)
    )
    before = len(df)
    df = df[mask].reset_index(drop=True)
    logger.info(
        "Length filter removed %d rows  (kept %d / %d)",
        before - len(df), len(df), before,
    )

    # ── Encode target
    le = LabelEncoder()
    df["label_name"] = df[target].astype(str)
    df["target"]     = le.fit_transform(df["label_name"])

    logger.info(
        "Target '%s' | classes: %s | distribution:\n%s",
        target,
        list(le.classes_),
        df["label_name"].value_counts().to_string(),
    )

    return df, le


def split(
    df: pd.DataFrame,
    test_size: float | None = None,
    val_size: float | None = None,
    random_state: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split.

    Returns
    -------
    (df_train, df_val, df_test)
    """
    test_size    = test_size    or DATA_CONFIG["test_size"]
    val_size     = val_size     or DATA_CONFIG["val_size"]
    random_state = random_state or DATA_CONFIG["random_state"]

    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )
    # val_size is relative to the full dataset, adjust for the reduced pool
    adjusted_val = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=adjusted_val,
        random_state=random_state,
        stratify=df_train_val["target"],
    )

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(df_train), len(df_val), len(df_test),
    )
    return df_train, df_val, df_test
