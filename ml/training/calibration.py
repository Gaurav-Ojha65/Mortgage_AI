"""
OOF Probability Calibration & Model Packaging — Mortgage AI
============================================================
Fits probability calibrators (Isotonic Regression and Platt Sigmoid Scaling)
strictly on out-of-fold (OOF) predicted probabilities generated across the real
training data split.

Key Guarantees:
- Calibrators are fit ONLY on (p_oof, y_train_real) [N=99,856, natural default prevalence 6.76%].
- No validation labels (val.csv) or test labels (test.csv) are used during calibrator fitting.
- Calibrator artifacts and self-contained inference wrappers (CalibratedPredictor) are persisted.
- Full provenance metadata is recorded in calibration_metadata.json and training_metadata.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.inference.predict import MODEL_FEATURES
from ml.training.eval_utils import compute_calibration_metrics
from ml.training.oof_calibration import run_oof_workflow, DEFAULT_LIGHTGBM_PARAMS
from ml.training.calibrated_predictor import CalibratedPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = _PROJECT_ROOT / "ml" / "models"
DATA_DIR = _PROJECT_ROOT / "data"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _compute_sha256(filepath: Path) -> str:
    """Compute sha256 hash of a file for provenance tracking."""
    if not filepath.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit_sha() -> str:
    """Retrieve current Git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted_or_git_unavailable"


def fit_oof_calibrators(
    p_oof: np.ndarray,
    y_true: np.ndarray,
    model_name: str = "lightgbm",
) -> Dict[str, Any]:
    """
    Fit Isotonic and Platt Sigmoid calibrators on OOF predictions.
    """
    logger.info(f"Fitting OOF calibrators on {len(p_oof):,} OOF probability pairs...")

    # 1. Isotonic Regression Calibrator
    t0_iso = time.time()
    iso_cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_cal.fit(p_oof, y_true)
    iso_p = iso_cal.predict(p_oof)
    iso_time = time.time() - t0_iso

    iso_auc = roc_auc_score(y_true, iso_p)
    iso_pr_auc = average_precision_score(y_true, iso_p)
    iso_brier = brier_score_loss(y_true, iso_p)
    iso_metrics = compute_calibration_metrics(y_true, iso_p, n_bins=10)

    logger.info(f"  [Isotonic] Fitted in {iso_time:.2f}s | OOF Brier: {iso_brier:.4f} | ECE: {iso_metrics['ece']:.4f}")

    # 2. Platt Sigmoid Calibrator (Logistic Regression)
    t0_sig = time.time()
    sig_cal = LogisticRegression(random_state=42, max_iter=1000)
    sig_cal.fit(p_oof.reshape(-1, 1), y_true)
    sig_p = sig_cal.predict_proba(p_oof.reshape(-1, 1))[:, 1]
    sig_time = time.time() - t0_sig

    sig_auc = roc_auc_score(y_true, sig_p)
    sig_pr_auc = average_precision_score(y_true, sig_p)
    sig_brier = brier_score_loss(y_true, sig_p)
    sig_metrics = compute_calibration_metrics(y_true, sig_p, n_bins=10)

    logger.info(f"  [Sigmoid]  Fitted in {sig_time:.2f}s | OOF Brier: {sig_brier:.4f} | ECE: {sig_metrics['ece']:.4f}")

    # Persist standalone calibrator artifacts
    iso_path = MODELS_DIR / f"{model_name}_oof_calibrator_isotonic.joblib"
    sig_path = MODELS_DIR / f"{model_name}_oof_calibrator_sigmoid.joblib"

    joblib.dump(iso_cal, iso_path)
    joblib.dump(sig_cal, sig_path)
    logger.info(f"Saved calibrator artifacts to {MODELS_DIR}")

    return {
        "isotonic_calibrator": iso_cal,
        "sigmoid_calibrator": sig_cal,
        "isotonic_oof_metrics": {
            "roc_auc": round(float(iso_auc), 4),
            "pr_auc": round(float(iso_pr_auc), 4),
            "brier_score": round(float(iso_brier), 4),
            "ece": iso_metrics["ece"],
            "macro_ece": iso_metrics["macro_ece"],
            "fit_time_s": round(iso_time, 2),
        },
        "sigmoid_oof_metrics": {
            "roc_auc": round(float(sig_auc), 4),
            "pr_auc": round(float(sig_pr_auc), 4),
            "brier_score": round(float(sig_brier), 4),
            "ece": sig_metrics["ece"],
            "macro_ece": sig_metrics["macro_ece"],
            "fit_time_s": round(sig_time, 2),
        },
    }


def train_final_model_and_save_pipeline(
    calibrator_result: Dict[str, Any],
    seed: int = 42,
) -> CalibratedPredictor:
    """
    Train final LightGBM base model on full training split and bundle into
    CalibratedPredictor.
    """
    import lightgbm as lgb

    train_path = DATA_DIR / "train.csv"
    train_df = pd.read_csv(train_path)

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df["target"]

    logger.info(f"Training final LightGBM on full train.csv ({len(train_df):,} samples)...")
    final_lgb = lgb.LGBMClassifier(**DEFAULT_LIGHTGBM_PARAMS)
    final_lgb.fit(X_train, y_train)

    # Save raw base model
    lgb_save_path = MODELS_DIR / "lightgbm.joblib"
    joblib.dump(final_lgb, lgb_save_path)
    logger.info(f"Saved final raw base model to {lgb_save_path}")

    # Build calibrated pipeline wrapper
    iso_cal = calibrator_result["isotonic_calibrator"]
    calibrated_pipeline = CalibratedPredictor(
        base_model=final_lgb,
        calibrator=iso_cal,
        calibration_method="isotonic",
        model_name="lightgbm",
        version="v3.0-oof-calibrated",
    )

    pipeline_path = MODELS_DIR / "lightgbm_calibrated_pipeline.joblib"
    joblib.dump(calibrated_pipeline, pipeline_path)
    logger.info(f"Saved complete CalibratedPredictor pipeline to {pipeline_path}")

    return calibrated_pipeline


def save_comprehensive_metadata(
    oof_result: Dict[str, Any],
    calibrator_result: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate and persist calibration_metadata.json and training_metadata.json.
    """
    import sklearn
    import lightgbm

    real_train_path = DATA_DIR / "real_train.csv"
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"

    real_hash = _compute_sha256(real_train_path)
    schema_hash = hashlib.sha256(",".join(MODEL_FEATURES).encode("utf-8")).hexdigest()
    git_sha = _get_git_commit_sha()

    cal_metadata = {
        "model_name": "LightGBM",
        "model_version": "v3.0-oof-calibrated",
        "calibration_version": "oof-iso-v3.0",
        "methodology": "5-Fold Stratified OOF Calibration on Real Training Split",
        "git_commit_sha": git_sha,
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "lightgbm_version": lightgbm.__version__,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        },
        "dataset_provenance": {
            "real_train_csv_sha256": real_hash,
            "feature_schema_sha256": schema_hash,
            "real_train_samples": int(len(oof_result["oof_probs"])),
            "smote_train_samples": int(len(pd.read_csv(train_path))),
            "val_samples": int(len(pd.read_csv(val_path))),
            "test_samples": int(len(pd.read_csv(test_path))),
            "natural_default_prevalence": round(float(oof_result["y_true"].mean()), 4),
        },
        "oof_configuration": {
            "n_splits": oof_result["n_splits"],
            "random_seed": oof_result["random_seed"],
            "fold_balancing": "SMOTE(0.5) applied strictly to fold-training partition",
            "early_stopping_used_in_oof": False,
            "lightgbm_params": DEFAULT_LIGHTGBM_PARAMS,
            "feature_names": MODEL_FEATURES,
        },
        "measured_oof_metrics": {
            "raw_lightgbm": oof_result["raw_metrics"],
            "isotonic_calibrated": calibrator_result["isotonic_oof_metrics"],
            "sigmoid_calibrated": calibrator_result["sigmoid_oof_metrics"],
        },
        "saved_artifacts": {
            "raw_base_model": "ml/models/lightgbm.joblib",
            "oof_calibrator_isotonic": "ml/models/lightgbm_oof_calibrator_isotonic.joblib",
            "oof_calibrator_sigmoid": "ml/models/lightgbm_oof_calibrator_sigmoid.joblib",
            "calibrated_pipeline": "ml/models/lightgbm_calibrated_pipeline.joblib",
        },
        "created_at": datetime.now().isoformat(),
    }

    cal_meta_path = MODELS_DIR / "calibration_metadata.json"
    with open(cal_meta_path, "w") as f:
        json.dump(cal_metadata, f, indent=2)

    training_meta_path = MODELS_DIR / "training_metadata.json"
    with open(training_meta_path, "w") as f:
        json.dump(cal_metadata, f, indent=2)

    logger.info(f"Saved calibration metadata to {cal_meta_path}")
    logger.info(f"Saved training metadata to {training_meta_path}")
    return cal_metadata


def run_calibration_pipeline(seed: int = 42) -> Tuple[CalibratedPredictor, Dict[str, Any]]:
    """Execute complete OOF calibration pipeline end-to-end."""
    logger.info("=" * 70)
    logger.info("  STARTING OOF CROSS-CALIBRATION PIPELINE")
    logger.info("=" * 70)

    # Step 1: Run OOF prediction generation
    oof_res = run_oof_workflow(seed=seed)

    # Step 2: Fit OOF calibrators
    cal_res = fit_oof_calibrators(oof_res["oof_probs"], oof_res["y_true"], model_name="lightgbm")

    # Step 3: Train final base model and bundle into CalibratedPredictor
    pipeline = train_final_model_and_save_pipeline(cal_res, seed=seed)

    # Step 4: Save comprehensive provenance metadata
    metadata = save_comprehensive_metadata(oof_res, cal_res, seed=seed)

    logger.info("=" * 70)
    logger.info("  OOF CALIBRATION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    return pipeline, metadata


if __name__ == "__main__":
    run_calibration_pipeline()
