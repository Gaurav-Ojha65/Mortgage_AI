"""
LightGBM Probability Calibration Pipeline — Mortgage AI
========================================================
Fits probability calibrators (Sigmoid / Platt and Isotonic Regression)
on the held-out validation set and evaluates them once on the untouched test set.

METHODOLOGY & PROTOCOL:
1. Raw Model: Loaded from frozen baseline artifact (ml/models/lightgbm.joblib).
   The raw model artifact is NEVER overwritten.
2. Calibrators: Fit strictly on the VALIDATION split (val.csv, N=21,398).
3. Evaluation: Evaluated ONCE on the untouched TEST split (test.csv, N=21,398).
4. Threshold Optimization: Selected by maximizing F1 on the validation set,
   then applied out-of-sample to the test set.
5. Known Limitation: Calibrator fitting and threshold optimization currently
   share the same validation split. In future production iterations, a nested
   split or cross-calibration scheme can be employed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

from ml.inference.predict import MODEL_FEATURES, MODELS_DIR
from ml.training.train import load_data
from ml.training.eval_utils import compute_calibration_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "metrics"


def find_optimal_threshold(model, X_val, y_val) -> float:
    """Find threshold maximizing F1 on validation set."""
    y_prob_val = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y_val, y_prob_val >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return round(float(best_thresh), 3)


def evaluate_variant(
    name: str,
    model,
    X_val,
    y_val,
    X_test,
    y_test,
) -> Dict[str, Any]:
    """Evaluate a raw or calibrated model variant on validation and test sets."""
    # 1. Validation threshold search
    opt_thresh = find_optimal_threshold(model, X_val, y_val)
    y_prob_val = model.predict_proba(X_val)[:, 1]
    val_f1_opt = f1_score(y_val, y_prob_val >= opt_thresh, zero_division=0)

    # 2. Test predictions
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_opt = (y_prob_test >= opt_thresh).astype(int)
    y_pred_50 = (y_prob_test >= 0.50).astype(int)

    # Confusion matrix & standard metrics at optimal threshold
    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    acc_opt = float(accuracy_score(y_test, y_pred_opt))
    prec_opt = float(precision_score(y_test, y_pred_opt, zero_division=0))
    rec_opt = float(recall_score(y_test, y_pred_opt, zero_division=0))
    f1_opt = float(f1_score(y_test, y_pred_opt, zero_division=0))
    spec_opt = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Calibration metrics via canonical eval_utils
    cal_metrics = compute_calibration_metrics(y_test, y_prob_test, n_bins=10)

    return {
        "variant": name,
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_test)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob_test)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob_test)), 4),
        "ece": cal_metrics["ece"],
        "macro_ece": cal_metrics["macro_ece"],
        "val_optimal_threshold": opt_thresh,
        "val_f1_at_opt_thresh": round(float(val_f1_opt), 4),
        "test_f1": round(f1_opt, 4),
        "test_precision": round(prec_opt, 4),
        "test_recall": round(rec_opt, 4),
        "test_specificity": round(spec_opt, 4),
        "test_accuracy": round(acc_opt, 4),
        "test_f1_at_0.50": round(float(f1_score(y_test, y_pred_50, zero_division=0)), 4),
        "test_prec_at_0.50": round(float(precision_score(y_test, y_pred_50, zero_division=0)), 4),
        "test_rec_at_0.50": round(float(recall_score(y_test, y_pred_50, zero_division=0)), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "calibration": cal_metrics,
        "mean_predicted_prob": round(float(np.mean(y_prob_test)), 4),
        "test_true_prevalence": round(float(np.mean(y_test)), 4),
    }


def run_calibration(seed: int = 42) -> dict:
    """Run full calibration pipeline on frozen LightGBM baseline."""
    logger.info("=" * 70)
    logger.info("  MORTGAGE AI — LIGHTGBM PROBABILITY CALIBRATION PIPELINE")
    logger.info("=" * 70)

    # 1. Load canonical data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(seed)
    logger.info(f"Loaded data: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    logger.info(f"Test prevalence: {y_test.mean():.2%}")

    # 2. Load frozen raw baseline model (DO NOT OVERWRITE)
    raw_model_path = MODELS_DIR / "lightgbm.joblib"
    if not raw_model_path.exists():
        raise FileNotFoundError(f"Frozen LightGBM model not found at {raw_model_path}. Run train.py first.")

    raw_model = joblib.load(raw_model_path)
    logger.info(f"Loaded frozen LightGBM baseline from {raw_model_path}")

    # 3. Fit calibrators strictly on validation data
    logger.info("Fitting Sigmoid (Platt) calibrator on validation data...")
    calibrator_sigmoid = CalibratedClassifierCV(
        estimator=FrozenEstimator(raw_model), method="sigmoid"
    )
    calibrator_sigmoid.fit(X_val, y_val)
    joblib.dump(calibrator_sigmoid, MODELS_DIR / "lightgbm_calibrated_sigmoid.joblib")
    logger.info(f"Saved: {MODELS_DIR / 'lightgbm_calibrated_sigmoid.joblib'}")

    logger.info("Fitting Isotonic calibrator on validation data...")
    calibrator_isotonic = CalibratedClassifierCV(
        estimator=FrozenEstimator(raw_model), method="isotonic"
    )
    calibrator_isotonic.fit(X_val, y_val)
    joblib.dump(calibrator_isotonic, MODELS_DIR / "lightgbm_calibrated_isotonic.joblib")
    logger.info(f"Saved: {MODELS_DIR / 'lightgbm_calibrated_isotonic.joblib'}")

    # 4. Evaluate all variants on the untouched test set
    variants = {
        "Raw LightGBM (Uncalibrated)": raw_model,
        "Sigmoid (Platt Scaling)": calibrator_sigmoid,
        "Isotonic Regression": calibrator_isotonic,
    }

    report_data = {}
    for name, model in variants.items():
        logger.info(f"\nEvaluating: {name}...")
        eval_dict = evaluate_variant(name, model, X_val, y_val, X_test, y_test)
        report_data[name] = eval_dict

        logger.info(
            f"  ROC-AUC: {eval_dict['roc_auc']:.4f} | PR-AUC: {eval_dict['pr_auc']:.4f} | "
            f"Brier: {eval_dict['brier_score']:.4f} | ECE: {eval_dict['ece']:.4f}"
        )
        logger.info(
            f"  Val-Opt Thresh: {eval_dict['val_optimal_threshold']} -> "
            f"Test F1: {eval_dict['test_f1']:.4f} | Rec: {eval_dict['test_recall']:.4f} | "
            f"Prec: {eval_dict['test_precision']:.4f} | Spec: {eval_dict['test_specificity']:.4f} | "
            f"Acc: {eval_dict['test_accuracy']:.4f}"
        )

    # 5. Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "calibration_report.json"
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"\nSaved calibration report to {report_file}")

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LightGBM calibration benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()
    run_calibration(seed=args.seed)
