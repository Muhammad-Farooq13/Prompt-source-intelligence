"""
train.py
────────
End-to-end training script.

Usage
-----
    python train.py                          # use defaults from config.py
    python train.py --target complexity_label --sample 40000
    python train.py --quick                  # fast run (2 CV fold, 5 iter)

The script:
  1. Downloads / loads the raw OpenOrca sample
  2. Preprocesses text and encodes labels
  3. Splits into train / val / test
  4. Trains all registered models with RandomizedSearchCV
  5. Evaluates on the test set
  6. Saves a model bundle to models/model_bundle.pkl
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from src.data.loader import load_raw
from src.data.preprocessor import preprocess, split
from src.features.engineer import build_feature_pipeline, get_feature_names
from src.models.evaluator import build_bundle
from src.models.trainer import train_all

# ── Logging setup ─────────────────────────────────────────────────────────────
def _setup_logging(level: str = "INFO"):
    log_file = Path(cfg.PATHS["logs"]) / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
    )

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train OpenOrca query classifiers")
    p.add_argument(
        "--target", default=cfg.TARGET,
        choices=["source_label", "complexity_label"],
        help="Target label to classify (default: %(default)s)",
    )
    p.add_argument(
        "--sample", type=int, default=None,
        help="Number of rows to sample (overrides config)",
    )
    p.add_argument(
        "--n-iter", type=int, default=10,
        help="RandomizedSearchCV iterations per model (default: 10)",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick run: 2 CV folds, 3 iterations, 10k sample")
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    _setup_logging(args.log_level)
    logger.info("=" * 70)
    logger.info("OpenOrca Query Intelligence — Training Pipeline")
    logger.info("=" * 70)

    # Override config if --quick flag set
    if args.quick:
        cfg.MODEL_CONFIG["cv_folds"] = 2
        args.n_iter                  = 3
        args.sample                  = args.sample or 10_000
        logger.info("Quick mode: 2 folds, 3 iter, 10k sample")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    logger.info("STEP 1 — Loading data …")
    df_raw = load_raw()
    if args.sample:
        df_raw = df_raw.sample(
            n=min(args.sample, len(df_raw)),
            random_state=cfg.DATA_CONFIG["random_state"],
        ).reset_index(drop=True)
    logger.info("Raw data shape: %s", df_raw.shape)

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    logger.info("STEP 2 — Preprocessing …")
    df_processed, label_encoder = preprocess(df_raw, target=args.target)
    logger.info("Processed shape: %s | Classes: %s",
                df_processed.shape, list(label_encoder.classes_))

    # ------------------------------------------------------------------
    # 3. Train / val / test split
    # ------------------------------------------------------------------
    logger.info("STEP 3 — Splitting …")
    df_train, df_val, df_test = split(df_processed)

    # ------------------------------------------------------------------
    # 4. Feature names (need a fitted pipeline for name extraction)
    # ------------------------------------------------------------------
    logger.info("STEP 4 — Fitting feature pipeline for name extraction …")
    _feat_pipe = build_feature_pipeline()
    _feat_pipe.fit(df_train)
    feature_names = get_feature_names(_feat_pipe)
    logger.info("Total features: %d", len(feature_names))

    # ------------------------------------------------------------------
    # 5. Train all models
    # ------------------------------------------------------------------
    logger.info("STEP 5 — Training models …")
    results = train_all(df_train, df_val, n_iter=args.n_iter)

    # ------------------------------------------------------------------
    # 6. Evaluate & save bundle
    # ------------------------------------------------------------------
    logger.info("STEP 6 — Evaluating on test set & saving bundle …")
    bundle = build_bundle(
        results=results,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        label_encoder=label_encoder,
        feature_names=feature_names,
        config={
            "target": args.target,
            "sample_size": len(df_raw),
            "data_config": cfg.DATA_CONFIG,
            "feature_config": cfg.FEATURE_CONFIG,
        },
    )

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    for model_name, m in bundle["metrics"].items():
        star = " ← BEST" if model_name == bundle["best_model_name"] else ""
        logger.info(
            "  %-20s  Accuracy=%.4f  F1=%.4f  ROC-AUC=%s%s",
            model_name,
            m["accuracy"],
            m["f1_weighted"],
            f"{m.get('roc_auc_ovr', float('nan')):.4f}",
            star,
        )

    logger.info("")
    logger.info("Bundle saved to: %s", cfg.PATHS["model_bundle"])
    logger.info("Run the dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
