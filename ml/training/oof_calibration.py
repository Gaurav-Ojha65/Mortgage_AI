"""
Out-Of-Fold (OOF) Prediction Generator — Mortgage AI
=====================================================
Performs 5-fold Stratified Cross-Validation strictly on real (un-augmented)
training data (`data/real_train.csv`).

Methodology:
1. Real training records (N=99,856, natural default prevalence 6.76%) are partitioned
   into 5 stratified folds.
2. In each fold:
   - SMOTE (50/50 balance) is applied ONLY to the fold's training partition.
   - LightGBM is fit with fixed hyperparameters on the SMOTE-balanced fold training data.
   - Probability predictions are generated ONLY for the held-out fold validation partition
     (which consists of 100% real records at natural 6.76% default prevalence).
3. Held-out predictions are mapped back to their original row indices, assembling a
   complete out-of-fold probability vector p_oof where each real training record receives
   exactly one out-of-sample prediction.
4. Validation (val.csv) and Test (test.csv) splits are NEVER touched during OOF generation.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.inference.predict import MODEL_FEATURES
from ml.training.eval_utils import compute_calibration_metrics
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = _PROJECT_ROOT / "ml" / "models"
DATA_DIR = _PROJECT_ROOT / "data"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LIGHTGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 25,
    "reg_alpha": 0.2,
    "reg_lambda": 2.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def load_real_train_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load canonical real (un-augmented) training dataset.

    Returns:
        X_real (DataFrame of shape (N, 15)), y_real (Series of length N)
    """
    d_dir = data_dir or DATA_DIR
    real_train_path = d_dir / "real_train.csv"

    if not real_train_path.exists():
        raise FileNotFoundError(
            f"Canonical real training split not found at: {real_train_path}\n"
            f"Run data pipeline or scratch/export_real_train.py first."
        )

    df = pd.read_csv(real_train_path)
    X = df[MODEL_FEATURES]
    y = df["target"]

    logger.info(
        f"Loaded real training data: {len(df):,} samples | "
        f"Defaults: {int(y.sum()):,} ({y.mean():.2%})"
    )
    return X, y


def generate_oof_predictions(
    X_real: pd.DataFrame,
    y_real: pd.Series,
    n_splits: int = 5,
    random_seed: int = 42,
    lgb_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate 5-fold out-of-fold probability predictions on real training data.

    Applies SMOTE strictly inside each fold's training split.
    """
    t0 = time.time()
    params = dict(DEFAULT_LIGHTGBM_PARAMS)
    if lgb_params:
        params.update(lgb_params)
    params["random_state"] = random_seed

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    n_samples = len(X_real)
    oof_probs = np.zeros(n_samples, dtype=np.float64)
    visited_indices = np.zeros(n_samples, dtype=bool)

    logger.info(f"Starting {n_splits}-fold OOF prediction generation on {n_samples:,} real samples...")

    X_mat = X_real.values
    y_arr = y_real.values

    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y_arr), 1):
        f_t0 = time.time()
        logger.info(f"  --- Fold {fold}/{n_splits} ---")

        # 1. Real fold partitions
        X_fold_train_real = X_mat[train_idx]
        y_fold_train_real = y_arr[train_idx]
        X_fold_val_real = X_mat[val_idx]
        y_fold_val_real = y_arr[val_idx]

        # 2. In-fold SMOTE augmentation on fold training data ONLY
        smote = SMOTE(sampling_strategy=0.5, random_state=random_seed + fold, k_neighbors=5)
        X_fold_train_smote, y_fold_train_smote = smote.fit_resample(
            X_fold_train_real, y_fold_train_real
        )
        X_fold_train_df = pd.DataFrame(X_fold_train_smote, columns=MODEL_FEATURES)
        X_fold_val_df = pd.DataFrame(X_fold_val_real, columns=MODEL_FEATURES)

        logger.info(
            f"    Train: {len(X_fold_train_real):,} real -> {len(X_fold_train_smote):,} SMOTE | "
            f"Val: {len(X_fold_val_real):,} real ({y_fold_val_real.mean():.2%} defaults)"
        )

        # 3. Fit LightGBM on augmented fold training data
        fold_lgb = lgb.LGBMClassifier(**params)
        fold_lgb.fit(X_fold_train_df, y_fold_train_smote)

        # 4. Predict probabilities on held-out REAL fold validation data
        fold_val_probs = fold_lgb.predict_proba(X_fold_val_df)[:, 1]

        # 5. Store at original row indices
        oof_probs[val_idx] = fold_val_probs
        visited_indices[val_idx] = True

        fold_models.append(fold_lgb)
        f_elapsed = time.time() - f_t0
        fold_auc = roc_auc_score(y_fold_val_real, fold_val_probs)
        logger.info(f"    Fold {fold} finished in {f_elapsed:.1f}s | Val ROC-AUC: {fold_auc:.4f}")

    # =========================================================================
    # Strict Provenance & Integrity Assertions
    # =========================================================================
    assert len(oof_probs) == n_samples, f"OOF length mismatch: {len(oof_probs)} != {n_samples}"
    assert np.all(visited_indices), "Not all real training samples received an OOF prediction"
    assert not np.any(np.isnan(oof_probs)), "NaN detected in OOF probability vector"
    assert not np.any(np.isinf(oof_probs)), "Inf detected in OOF probability vector"
    assert np.all((oof_probs >= 0.0) & (oof_probs <= 1.0)), "OOF probabilities outside [0.0, 1.0]"

    elapsed_total = time.time() - t0
    logger.info(f"All {n_splits} folds completed successfully in {elapsed_total:.1f}s.")

    # Compute raw OOF evaluation metrics
    raw_auc = roc_auc_score(y_arr, oof_probs)
    raw_pr_auc = average_precision_score(y_arr, oof_probs)
    raw_brier = brier_score_loss(y_arr, oof_probs)
    cal_metrics = compute_calibration_metrics(y_arr, oof_probs, n_bins=10)

    raw_metrics = {
        "roc_auc": round(float(raw_auc), 4),
        "pr_auc": round(float(raw_pr_auc), 4),
        "brier_score": round(float(raw_brier), 4),
        "ece": cal_metrics["ece"],
        "macro_ece": cal_metrics["macro_ece"],
        "n_samples": n_samples,
        "n_splits": n_splits,
        "execution_time_s": round(elapsed_total, 2),
    }

    logger.info("Raw OOF Performance Metrics:")
    logger.info(f"  ROC-AUC:   {raw_metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC:    {raw_metrics['pr_auc']:.4f}")
    logger.info(f"  Brier:     {raw_metrics['brier_score']:.4f}")
    logger.info(f"  ECE (10b): {raw_metrics['ece']:.4f}")
    logger.info(f"  Macro ECE: {raw_metrics['macro_ece']:.4f}")

    return {
        "oof_probs": oof_probs,
        "y_true": y_arr,
        "raw_metrics": raw_metrics,
        "cal_details": cal_metrics,
        "lgb_params": params,
        "n_splits": n_splits,
        "random_seed": random_seed,
    }


def run_oof_workflow(data_dir: Optional[Path] = None, seed: int = 42) -> Dict[str, Any]:
    """Execute complete OOF generation and save artifact."""
    X_real, y_real = load_real_train_data(data_dir)
    oof_result = generate_oof_predictions(X_real, y_real, n_splits=5, random_seed=seed)

    save_path = MODELS_DIR / "oof_predictions.joblib"
    joblib.dump(
        {
            "oof_probs": oof_result["oof_probs"],
            "y_true": oof_result["y_true"],
            "raw_metrics": oof_result["raw_metrics"],
            "n_splits": oof_result["n_splits"],
            "random_seed": oof_result["random_seed"],
        },
        save_path,
    )
    logger.info(f"Saved OOF predictions artifact to {save_path}")
    return oof_result


if __name__ == "__main__":
    run_oof_workflow()
