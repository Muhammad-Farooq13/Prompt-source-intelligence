"""
src/data/loader.py
──────────────────
Downloads the Open-Orca/OpenOrca dataset from HuggingFace, draws a stratified
random sample, and persists it as a Parquet file so the rest of the pipeline
never has to re-download.
"""

import logging
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to path when running this module directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_CONFIG, PATHS

logger = logging.getLogger(__name__)


# ── Label extraction ──────────────────────────────────────────────────────────

# Maps raw ID prefixes to human-readable source labels
_PREFIX_MAP = {
    "flan":   "FLAN",
    "t0":     "T0",
    "cot":    "CoT",
    "niv":    "NIV",
    "sharegpt": "ShareGPT",
    "cod":    "CoD",
}

def _extract_source(id_str: str) -> str:
    """Return a clean source label from a raw `id` string."""
    id_str = str(id_str).lower().strip()
    for prefix, label in _PREFIX_MAP.items():
        if id_str.startswith(prefix):
            return label
    # Fallback: take everything before the first '.' or digit run
    match = re.match(r"^([a-z_\-]+)", id_str)
    return match.group(1).upper() if match else "OTHER"


def _extract_complexity(response: str) -> str:
    """Bucket responses into SHORT / MEDIUM / LONG by word count."""
    wc = len(str(response).split())
    if wc < 50:
        return "SHORT"
    elif wc < 200:
        return "MEDIUM"
    return "LONG"


# ── Core loader ───────────────────────────────────────────────────────────────

def download_and_save(
    sample_size: int | None = None,
    random_state: int | None = None,
    force: bool = False,
) -> Path:
    """
    Download the OpenOrca dataset, sample it, add derived labels, and save
    as Parquet.  Skips the download if the file already exists (use
    ``force=True`` to re-download).

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    out_path = Path(PATHS["raw_data"])
    if out_path.exists() and not force:
        logger.info("Raw data already exists at %s — skipping download.", out_path)
        return out_path

    sample_size   = sample_size   or DATA_CONFIG["sample_size"]
    random_state  = random_state  or DATA_CONFIG["random_state"]

    logger.info("Loading dataset '%s' from HuggingFace …", DATA_CONFIG["dataset_name"])
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required. Install it with: "
            "pip install datasets"
        ) from exc

    ds = load_dataset(DATA_CONFIG["dataset_name"], split="train", streaming=False)
    logger.info("Full dataset size: %d rows", len(ds))

    # Convert to pandas and sample
    logger.info("Converting to DataFrame and sampling %d rows …", sample_size)
    df = ds.to_pandas()

    df = df.sample(
        n=min(sample_size, len(df)),
        random_state=random_state,
        replace=False,
    ).reset_index(drop=True)

    # Drop rows with null question / response
    df.dropna(subset=["question", "response"], inplace=True)

    # Derived labels
    logger.info("Extracting labels …")
    df["source_label"]     = df["id"].apply(_extract_source)
    df["complexity_label"] = df["response"].apply(_extract_complexity)

    # Filter rare classes
    min_samples = DATA_CONFIG.get("min_class_samples", 500)
    for col in ("source_label", "complexity_label"):
        counts = df[col].value_counts()
        valid  = counts[counts >= min_samples].index
        df     = df[df[col].isin(valid)]
    df.reset_index(drop=True, inplace=True)

    logger.info("Final sample shape: %s", df.shape)
    logger.info("Source distribution:\n%s", df["source_label"].value_counts().to_string())
    logger.info("Complexity distribution:\n%s", df["complexity_label"].value_counts().to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved to %s", out_path)
    return out_path


def load_raw() -> pd.DataFrame:
    """Load the persisted raw sample, downloading it first if necessary."""
    raw_path = Path(PATHS["raw_data"])
    if not raw_path.exists():
        download_and_save()
    return pd.read_parquet(raw_path)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s  %(message)s")
    download_and_save(force=False)
