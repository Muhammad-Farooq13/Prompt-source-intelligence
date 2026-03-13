"""
predict.py
──────────
CLI and importable inference API for the trained model bundle.

CLI usage
---------
    python predict.py --text "What is the capital of France?"
    python predict.py --file questions.txt        # one question per line
    python predict.py --interactive               # REPL mode

Programmatic usage
------------------
    from predict import load_bundle, predict_single

    bundle = load_bundle()
    result = predict_single("Explain gradient descent in detail.", bundle)
    print(result)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PATHS

logging.basicConfig(level="INFO", format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Bundle loader ─────────────────────────────────────────────────────────────

_BUNDLE_CACHE: Dict[str, Any] = {}

def load_bundle(bundle_path: str | Path | None = None) -> Dict[str, Any]:
    """Load (and cache) the model bundle from disk."""
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE:
        return _BUNDLE_CACHE
    path = Path(bundle_path or PATHS["model_bundle"])
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {path}.\n"
            "Run 'python train.py' to train the models first."
        )
    logger.info("Loading bundle from %s …", path)
    _BUNDLE_CACHE = joblib.load(path)
    return _BUNDLE_CACHE


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_single(
    question: str,
    bundle: Dict[str, Any] | None = None,
    model_name: str | None = None,
) -> Dict[str, Any]:
    """
    Run inference for a single question string.

    Parameters
    ----------
    question   : raw question text
    bundle     : pre-loaded bundle dict (loads from disk if None)
    model_name : specific model to use (uses best model if None)

    Returns
    -------
    dict with keys:
      predicted_class  : str
      predicted_index  : int
      probabilities    : {class: prob}
      model_used       : str
    """
    if bundle is None:
        bundle = load_bundle()

    model_name = model_name or bundle["best_model_name"]
    if model_name not in bundle["models"]:
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available: {list(bundle['models'].keys())}"
        )

    pipeline     = bundle["models"][model_name]
    label_names  = bundle["label_names"]
    le           = bundle["label_encoder"]

    # Build a minimal DataFrame with the columns the pipeline expects
    from src.data.preprocessor import clean_text
    question_clean = clean_text(question)
    df_input = pd.DataFrame({
        "question":       [question],
        "question_clean": [question_clean],
        "response_clean": [""],         # not known at inference time
    })

    pred_idx = pipeline.predict(df_input)[0]
    pred_label = label_names[pred_idx] if pred_idx < len(label_names) else str(pred_idx)

    probabilities = {}
    try:
        proba = pipeline.predict_proba(df_input)[0]
        probabilities = {
            label_names[i]: float(round(p, 4))
            for i, p in enumerate(proba)
        }
    except AttributeError:
        probabilities = {pred_label: 1.0}

    return {
        "predicted_class":  pred_label,
        "predicted_index":  int(pred_idx),
        "probabilities":    probabilities,
        "model_used":       model_name,
        "question_preview": question[:120] + ("…" if len(question) > 120 else ""),
    }


def predict_batch(
    questions: list[str],
    bundle: Dict[str, Any] | None = None,
    model_name: str | None = None,
) -> list[Dict[str, Any]]:
    """Run predict_single for a list of questions."""
    if bundle is None:
        bundle = load_bundle()
    return [predict_single(q, bundle, model_name) for q in questions]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Predict query class with trained models")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",        help="Single question string to classify")
    group.add_argument("--file",        help="Path to file with one question per line")
    group.add_argument("--interactive", action="store_true",
                       help="Enter REPL mode for live predictions")
    p.add_argument("--model",  default=None,
                   help="Model name to use (default: best model)")
    p.add_argument("--bundle", default=None,
                   help="Override path to model bundle .pkl file")
    p.add_argument("--json",   action="store_true",
                   help="Output results as JSON")
    return p.parse_args()


def main():
    args = _parse_args()
    bundle = load_bundle(args.bundle)

    def _fmt(result: dict) -> str:
        if args.json:
            return json.dumps(result, indent=2)
        lines = [
            f"\n  Question  : {result['question_preview']}",
            f"  Predicted : {result['predicted_class']}",
            f"  Model     : {result['model_used']}",
            "  Probabilities:",
        ]
        for cls, prob in sorted(
            result["probabilities"].items(), key=lambda x: -x[1]
        ):
            bar = "█" * int(prob * 20)
            lines.append(f"    {cls:<15} {prob:5.1%}  {bar}")
        return "\n".join(lines)

    if args.text:
        result = predict_single(args.text, bundle, args.model)
        print(_fmt(result))

    elif args.file:
        path = Path(args.file)
        if not path.exists():
            sys.exit(f"File not found: {path}")
        questions = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        logger.info("Running inference on %d questions …", len(questions))
        results = predict_batch(questions, bundle, args.model)
        for r in results:
            print(_fmt(r))
            print()

    elif args.interactive:
        print(f"\nOpenOrca Query Intelligence — Interactive Prediction Mode")
        print(f"Using model: {args.model or bundle['best_model_name']}")
        print("Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                text = input("Question > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = predict_single(text, bundle, args.model)
            print(_fmt(result))
            print()


if __name__ == "__main__":
    main()
