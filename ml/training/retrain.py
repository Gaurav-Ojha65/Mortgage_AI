"""
Master retraining script for Mortgage AI.

Runs the complete pipeline:
  1. Check if training data exists, run data pipeline if not
  2. Train XGBoost, LightGBM, LogisticRegression
  3. Evaluate on test set
  4. Save comparison report
  5. Set best model as active

Usage:
    python -m ml.retrain
    python -m ml.retrain --force-data    # Re-run data pipeline even if CSVs exist
    python -m ml.retrain --model-only    # Skip data pipeline, just retrain models
"""

import sys
import argparse
import logging
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR   = _ROOT / "data"
MODELS_DIR = _ROOT / "models"


def check_data_exists() -> bool:
    """Check if processed train/test CSVs exist."""
    return (DATA_DIR / "train.csv").exists() and (DATA_DIR / "test.csv").exists()


def run_data_pipeline():
    """Run the data processing pipeline to generate train/val/test CSVs."""
    logger.info("Running data pipeline...")
    try:
        from data.pipeline import run_pipeline
        result = run_pipeline(
            data_dir=str(DATA_DIR),
            output_dir=str(DATA_DIR),
        )
        logger.info("Data pipeline complete.")
        logger.info(f"  Train: {len(result['train_df']):,} rows")
        logger.info(f"  Val:   {len(result['val_df']):,} rows")
        logger.info(f"  Test:  {len(result['test_df']):,} rows")
        return True
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        logger.warning("Will train with synthetic data instead.")
        return False


def run_training():
    """Run the multi-model training script."""
    logger.info("Starting model training...")
    try:
        from ml.training.train import run as train_run
        winner = train_run()
        logger.info(f"Training complete. Winner: {winner}")
        return winner
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def clear_model_cache():
    """Clear the in-memory model cache after retraining."""
    try:
        from ml.inference.predict import clear_cache
        clear_cache()
        logger.info("Model cache cleared.")
    except Exception:
        pass


def print_model_summary():
    """Print current model status."""
    print("\n" + "=" * 62)
    print("  MODEL STATUS")
    print("=" * 62)

    txt = MODELS_DIR / "best_model_name.txt"
    if txt.exists():
        print(f"  Active model: {txt.read_text().strip()}")
    else:
        print("  Active model: not set")

    report_path = MODELS_DIR / "comparison_report.json"
    if report_path.exists():
        import json
        with open(report_path) as f:
            report = json.load(f)
        metrics = report.get("metrics", {})
        print(f"\n  {'Model':<22} {'AUC':>7} {'F1':>6} {'Precision':>10} {'Recall':>8}")
        print(f"  {'-'*56}")
        for name, m in metrics.items():
            winner_flag = "  <-- active" if name == report.get("winner") else ""
            print(
                f"  {name:<22} {m['roc_auc']:>7.4f} {m['f1']:>6.4f} "
                f"{m['precision']:>10.4f} {m['recall']:>8.4f}{winner_flag}"
            )
    else:
        print("  No comparison report found.")

    print("\n  Model files in models/:")
    for f in sorted(MODELS_DIR.glob("*.joblib")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"    {f.name:<30} ({size_mb:.1f} MB)")
    print("=" * 62)


def main():
    parser = argparse.ArgumentParser(description="Mortgage AI — Master Retrain Script")
    parser.add_argument("--force-data",  action="store_true",
                        help="Re-run data pipeline even if CSVs already exist")
    parser.add_argument("--model-only",  action="store_true",
                        help="Skip data pipeline, train models with existing data")
    parser.add_argument("--status",      action="store_true",
                        help="Print model status and exit")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    if args.status:
        print_model_summary()
        return

    print("=" * 62)
    print("  MORTGAGE AI — MASTER RETRAIN")
    print("=" * 62)

    # Step 1: Data
    if not args.model_only:
        if args.force_data or not check_data_exists():
            run_data_pipeline()
        else:
            logger.info("Training data found. Skipping pipeline. Use --force-data to regenerate.")
    else:
        logger.info("--model-only: skipping data pipeline")

    # Step 2: Train
    winner = run_training()

    # Step 3: Clear cache
    clear_model_cache()

    # Step 4: Summary
    print_model_summary()
    print(f"\n  Done! Active model: {winner}")
    print(f"  Start server: python run_server.py")


if __name__ == "__main__":
    main()
