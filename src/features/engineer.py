"""
src/features/engineer.py
────────────────────────
Builds a scikit-learn Pipeline that combines:
  1. TF-IDF on the cleaned question text
  2. Hand-crafted statistical / linguistic features

The resulting sparse + dense feature matrix is ready for any sklearn estimator.
"""

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURE_CONFIG

logger = logging.getLogger(__name__)


# ── Stat feature transformer ──────────────────────────────────────────────────

class StatFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Accepts a DataFrame (or list of dicts) and returns a 2-D numpy array of
    hand-crafted statistical features derived from *question_clean* and
    *response_clean* columns.

    When fit/transform receives a plain Series or list of strings (text-only
    input), only the question-level features are computed.
    """

    _PUNCT_RE = re.compile(r"[^\w\s]")
    _SENT_RE  = re.compile(r"[.!?]+")
    _CODE_RE  = re.compile(r"```|def |class |import |#include|<\?php")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for rec in X:
            q = str(rec.get("question_clean", rec)) if isinstance(rec, dict) else str(rec)
            r = str(rec.get("response_clean", "")) if isinstance(rec, dict) else ""
            rows.append(self._featurize(q, r))
        return np.array(rows, dtype=np.float32)

    def _featurize(self, q: str, r: str) -> list:
        words_q   = q.split()
        words_r   = r.split()
        unique_q  = set(w.lower() for w in words_q)

        return [
            len(q),                                              # q_char_len
            len(words_q),                                        # q_word_count
            max(1, len(self._SENT_RE.findall(q))),               # q_sentence_count
            np.mean([len(w) for w in words_q]) if words_q else 0,  # q_avg_word_len
            len(unique_q) / max(1, len(words_q)),                # q_unique_word_ratio
            len(self._PUNCT_RE.findall(q)),                      # q_punct_count
            sum(c.isdigit() for c in q),                         # q_digit_count
            sum(c.isupper() for c in q) / max(1, len(q)),        # q_upper_ratio
            q.count("?"),                                         # q_question_mark_count
            float(bool(self._CODE_RE.search(q))),                # q_has_code
            len(words_r),                                        # resp_word_count
            max(1, len(self._SENT_RE.findall(r))),               # resp_sentence_count
        ]


class DataFrameTextSelector(BaseEstimator, TransformerMixin):
    """Extracts a single text column from a DataFrame as a list of strings."""

    def __init__(self, column: str = "question_clean"):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.column].tolist()
        return list(X)


class DataFrameToRecords(BaseEstimator, TransformerMixin):
    """Converts a DataFrame to a list of dicts for StatFeatureExtractor."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cols = [c for c in ("question_clean", "response_clean") if c in X.columns]
            return X[cols].to_dict(orient="records")
        # Assume list of strings (question only)
        return [{"question_clean": s} for s in X]


# ── Feature pipeline factory ──────────────────────────────────────────────────

def build_feature_pipeline() -> Pipeline:
    """
    Returns an unfitted sklearn Pipeline that accepts a DataFrame and
    emits a sparse feature matrix combining TF-IDF and stat features.
    """
    cfg = FEATURE_CONFIG

    tfidf = TfidfVectorizer(
        max_features=cfg["tfidf_max_features"],
        ngram_range=cfg["tfidf_ngram_range"],
        sublinear_tf=cfg["tfidf_sublinear_tf"],
        min_df=cfg["tfidf_min_df"],
        max_df=cfg["tfidf_max_df"],
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        strip_accents="unicode",
    )

    tfidf_branch = Pipeline([
        ("selector", DataFrameTextSelector("question_clean")),
        ("tfidf",    tfidf),
    ])

    stat_branch = Pipeline([
        ("to_records", DataFrameToRecords()),
        ("stats",      StatFeatureExtractor()),
        ("scaler",     MaxAbsScaler()),
    ])

    feature_union = FeatureUnion([
        ("tfidf", tfidf_branch),
        ("stats", stat_branch),
    ])

    pipeline = Pipeline([("features", feature_union)])
    return pipeline


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Return a list of feature names from a fitted pipeline."""
    fu = pipeline.named_steps["features"]
    tfidf_names = (
        fu.transformer_list[0][1]
        .named_steps["tfidf"]
        .get_feature_names_out()
        .tolist()
    )
    stat_names = FEATURE_CONFIG["stat_features"]
    return tfidf_names + stat_names
