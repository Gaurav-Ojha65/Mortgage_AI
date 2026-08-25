"""
Out-of-Sample Test Set Evaluation & Comparative Benchmarking — Mortgage AI
===========================================================================
Evaluates the frozen OOF-calibrated pipeline and frozen policies on the untouched
test dataset (`data/test.csv`, N=21,398).

Performs a direct side-by-side comparison between:
1. Baseline: LightGBM + Val-fitted Isotonic Calibrator + Val-optimized Policy
2. New Proposed: LightGBM + OOF-fitted Isotonic Calibrator + Val-optimized Policy

Strict Protocols:
- Evaluates untouched test.csv exactly ONCE.
- No model parameter, calibrator, threshold, or policy is fit or tuned on test data.
- Outputs comprehensive comparative benchmark metrics and persists JSON report.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.inference.predict import MODEL_FEATURES
from ml.training.eval_utils import compute_calibration_metrics
from risk.decision_policy import CostModel, DecisionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = _PROJECT_ROOT / "ml" / "models"
DATA_DIR = _PROJECT_ROOT / "data"
REPORTS_DIR = _PROJECT_ROOT / "reports" / "metrics"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load canonical test split."""
    d_dir = data_dir or DATA_DIR
    test_path = d_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}")

    df = pd.read_csv(test_path)
    X = df[MODEL_FEATURES]
    y = df["target"]
    logger.info(f"Loaded untouched test split: {len(df):,} samples | Defaults: {int(y.sum()):,} ({y.mean():.2%})")
    return X, y


def evaluate_binary_policy(
    y_true: np.ndarray,
    p_test: np.ndarray,
    threshold: float,
    policy_name: str,
    cost_model: CostModel,
) -> Dict[str, Any]:
    """Evaluate a single binary decision threshold on test data."""
    pred = (p_test >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    n = len(y_true)

    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    acc = float(accuracy_score(y_true, pred))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    total_cost = float(fn * cost_model.cost_fn + fp * cost_model.cost_fp)
    cost_per_app = float(total_cost / n)

    return {
        "policy_name": policy_name,
        "thresholds": f"theta = {threshold:.3f}",
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "accuracy": round(acc, 4),
        "cost_per_applicant": round(cost_per_app, 2),
        "total_expected_cost": round(total_cost, 2),
        "distribution": {
            "approved": int(tn + fn),
            "approved_pct": round(float((tn + fn) / n), 4),
            "reviewed": 0,
            "reviewed_pct": 0.0,
            "rejected": int(tp + fp),
            "rejected_pct": round(float((tp + fp) / n), 4),
        },
        "cost_breakdown": {
            "approved_default_loss": float(fn * cost_model.cost_fn),
            "approved_defaults_count": int(fn),
            "rejected_good_borrower_loss": float(fp * cost_model.cost_fp),
            "rejected_good_count": int(fp),
            "manual_review_cost": 0.0,
            "manual_review_count": 0,
        },
    }


def evaluate_3tier_policy(
    y_true: np.ndarray,
    p_test: np.ndarray,
    approve_threshold: float,
    reject_threshold: float,
    policy_name: str,
    cost_model: CostModel,
) -> Dict[str, Any]:
    """Evaluate 3-tier routing policy on test data."""
    n = len(y_true)
    is_app = p_test <= approve_threshold
    is_rej = p_test >= reject_threshold
    is_rev = (~is_app) & (~is_rej)

    app_count = int(np.sum(is_app))
    rej_count = int(np.sum(is_rej))
    rev_count = int(np.sum(is_rev))

    # Approved defaults (FN)
    fn_count = int(np.sum((y_true == 1) & is_app))
    fn_cost = float(fn_count * cost_model.cost_fn)

    # Rejected non-defaults (FP)
    fp_count = int(np.sum((y_true == 0) & is_rej))
    fp_cost = float(fp_count * cost_model.cost_fp)

    # Manual review operational cost
    rev_cost = float(rev_count * cost_model.cost_manual_review)

    total_cost = fn_cost + fp_cost + rev_cost
    cost_per_app = float(total_cost / n)

    # Effective binary equivalent (treating auto-approve as 0, review+reject as flagged 1)
    effective_pred = (~is_app).astype(int)
    cm = confusion_matrix(y_true, effective_pred)
    tn, fp_eff, fn_eff, tp_eff = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    prec = float(precision_score(y_true, effective_pred, zero_division=0))
    rec = float(recall_score(y_true, effective_pred, zero_division=0))
    f1 = float(f1_score(y_true, effective_pred, zero_division=0))
    acc = float(accuracy_score(y_true, effective_pred))
    spec = float(tn / (tn + fp_eff)) if (tn + fp_eff) > 0 else 0.0
    fpr = float(fp_eff / (tn + fp_eff)) if (tn + fp_eff) > 0 else 0.0
    fnr = float(fn_eff / (tp_eff + fn_eff)) if (tp_eff + fn_eff) > 0 else 0.0

    return {
        "policy_name": policy_name,
        "thresholds": f"Approve <= {approve_threshold:.3f}, Reject >= {reject_threshold:.3f}",
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "accuracy": round(acc, 4),
        "cost_per_applicant": round(cost_per_app, 2),
        "total_expected_cost": round(total_cost, 2),
        "distribution": {
            "approved": app_count,
            "approved_pct": round(float(app_count / n), 4),
            "reviewed": rev_count,
            "reviewed_pct": round(float(rev_count / n), 4),
            "rejected": rej_count,
            "rejected_pct": round(float(rej_count / n), 4),
        },
        "cost_breakdown": {
            "approved_default_loss": fn_cost,
            "approved_defaults_count": fn_count,
            "rejected_good_borrower_loss": fp_cost,
            "rejected_good_count": fp_count,
            "manual_review_cost": rev_cost,
            "manual_review_count": rev_count,
        },
    }


def run_test_evaluation_benchmark() -> Dict[str, Any]:
    """
    Execute full comparative benchmark on untouched test set.
    """
    logger.info("=" * 80)
    logger.info("  FINAL OUT-OF-SAMPLE TEST SET EVALUATION BENCHMARK")
    logger.info("=" * 80)

    X_test, y_test = load_test_data()
    y_test_arr = y_test.values

    # Cost model
    cost_model = CostModel(cost_fn=10000.0, cost_fp=1000.0, cost_manual_review=150.0)

    # -------------------------------------------------------------------------
    # 1. Evaluate Baseline Model (LightGBM + Val-fitted Isotonic Calibrator)
    # -------------------------------------------------------------------------
    baseline_cal_path = MODELS_DIR / "lightgbm_calibrated_isotonic.joblib"
    if baseline_cal_path.exists():
        logger.info(f"Evaluating Baseline: {baseline_cal_path}...")
        baseline_cal = joblib.load(baseline_cal_path)
        p_base_test = baseline_cal.predict_proba(X_test)[:, 1]
    else:
        logger.warning(f"Baseline artifact not found at {baseline_cal_path}. Using raw model.")
        raw_lgb = joblib.load(MODELS_DIR / "lightgbm.joblib")
        p_base_test = raw_lgb.predict_proba(X_test)[:, 1]

    base_auc = roc_auc_score(y_test_arr, p_base_test)
    base_pr_auc = average_precision_score(y_test_arr, p_base_test)
    base_brier = brier_score_loss(y_test_arr, p_base_test)
    base_cal = compute_calibration_metrics(y_test_arr, p_base_test, n_bins=10)

    # Baseline Policies (Val-optimized)
    base_f1_pol = evaluate_binary_policy(y_test_arr, p_base_test, 0.155, "Baseline F1-Optimal (theta=0.155)", cost_model)
    base_3tier_pol = evaluate_3tier_policy(y_test_arr, p_base_test, 0.060, 0.330, "Baseline 3-Tier (0.060/0.330)", cost_model)

    # -------------------------------------------------------------------------
    # 2. Evaluate New Proposed System (LightGBM + OOF-fitted Calibrated Pipeline)
    # -------------------------------------------------------------------------
    oof_pipeline_path = MODELS_DIR / "lightgbm_calibrated_pipeline.joblib"
    if not oof_pipeline_path.exists():
        raise FileNotFoundError(f"OOF calibrated pipeline not found at {oof_pipeline_path}")

    logger.info(f"Evaluating New OOF-Calibrated Pipeline: {oof_pipeline_path}...")
    oof_pipeline = joblib.load(oof_pipeline_path)
    p_oof_test = oof_pipeline.predict_proba(X_test)[:, 1]

    oof_auc = roc_auc_score(y_test_arr, p_oof_test)
    oof_pr_auc = average_precision_score(y_test_arr, p_oof_test)
    oof_brier = brier_score_loss(y_test_arr, p_oof_test)
    oof_cal = compute_calibration_metrics(y_test_arr, p_oof_test, n_bins=10)

    # Load frozen policy thresholds
    frozen_config_path = MODELS_DIR / "frozen_policy_config.json"
    with open(frozen_config_path, "r") as f:
        frozen_cfg = json.load(f)

    threshs = frozen_cfg["frozen_thresholds"]
    t_f1 = threshs["f1_optimal"]["threshold"]
    t_bacc = threshs["balanced_accuracy"]["threshold"]
    t_cost10 = threshs["cost_sensitive_10_1"]["threshold"]
    t_cost5 = threshs["cost_sensitive_5_1"]["threshold"]
    t_app = threshs["three_tier_economic"]["approve_threshold"]
    t_rej = threshs["three_tier_economic"]["reject_threshold"]

    # Evaluate all OOF frozen policies on test set
    oof_pol_f1 = evaluate_binary_policy(y_test_arr, p_oof_test, t_f1, f"1. OOF F1-Optimal (theta={t_f1:.3f})", cost_model)
    oof_pol_bacc = evaluate_binary_policy(y_test_arr, p_oof_test, t_bacc, f"2. OOF Balanced Acc (theta={t_bacc:.3f})", cost_model)
    oof_pol_c10 = evaluate_binary_policy(y_test_arr, p_oof_test, t_cost10, f"3. OOF Cost-Sensitive 10:1 (theta={t_cost10:.3f})", cost_model)
    oof_pol_c5 = evaluate_binary_policy(y_test_arr, p_oof_test, t_cost5, f"4. OOF Cost-Sensitive 5:1 (theta={t_cost5:.3f})", cost_model)
    oof_pol_3tier = evaluate_3tier_policy(y_test_arr, p_oof_test, t_app, t_rej, f"5. OOF 3-Tier ({t_app:.3f}/{t_rej:.3f})", cost_model)

    cost_reduction_pct = round(
        float((oof_pol_f1["total_expected_cost"] - oof_pol_3tier["total_expected_cost"]) / oof_pol_f1["total_expected_cost"] * 100),
        2,
    )

    comparison_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "test_sample_size": len(X_test),
            "test_default_prevalence": round(float(y_test_arr.mean()), 4),
        },
        "cost_model": {
            "cost_fn": cost_model.cost_fn,
            "cost_fp": cost_model.cost_fp,
            "cost_manual_review": cost_model.cost_manual_review,
            "is_demonstration": True,
        },
        "baseline_system": {
            "name": "LightGBM + Val-Fitted Isotonic Calibrator + Val-Optimized Policy",
            "calibration_sample_size": 21398,
            "test_metrics": {
                "roc_auc": round(float(base_auc), 4),
                "pr_auc": round(float(base_pr_auc), 4),
                "brier_score": round(float(base_brier), 4),
                "ece_weighted": base_cal["ece"],
                "macro_ece": base_cal["macro_ece"],
            },
            "test_policies": {
                "f1_optimal": base_f1_pol,
                "three_tier_economic": base_3tier_pol,
            },
        },
        "oof_calibrated_system": {
            "name": "LightGBM + OOF-Fitted Isotonic Calibrator + Val-Optimized Policy",
            "calibration_sample_size": 99856,
            "test_metrics": {
                "roc_auc": round(float(oof_auc), 4),
                "pr_auc": round(float(oof_pr_auc), 4),
                "brier_score": round(float(oof_brier), 4),
                "ece_weighted": oof_cal["ece"],
                "macro_ece": oof_cal["macro_ece"],
            },
            "test_policies": {
                "f1_optimal": oof_pol_f1,
                "balanced_accuracy": oof_pol_bacc,
                "cost_sensitive_10_1": oof_pol_c10,
                "cost_sensitive_5_1": oof_pol_c5,
                "three_tier_economic": oof_pol_3tier,
            },
            "economic_summary": {
                "f1_optimal_cost": oof_pol_f1["total_expected_cost"],
                "three_tier_cost": oof_pol_3tier["total_expected_cost"],
                "total_savings": round(oof_pol_f1["total_expected_cost"] - oof_pol_3tier["total_expected_cost"], 2),
                "cost_reduction_pct": cost_reduction_pct,
            },
        },
    }

    report_path = REPORTS_DIR / "oof_vs_baseline_comparison.json"
    with open(report_path, "w") as f:
        json.dump(comparison_report, f, indent=2)

    logger.info(f"Saved comparative evaluation report to {report_path}")

    # Print clean benchmark comparison tables
    print("\n" + "=" * 105)
    print("  OUT-OF-SAMPLE TEST BENCHMARK: BASELINE vs. OOF-CALIBRATED SYSTEM (N=21,398)")
    print("=" * 105)
    header = f"{'System':<40} {'ROC-AUC':>8} {'PR-AUC':>8} {'Brier':>7} {'ECE (10b)':>10} {'Macro ECE':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'Baseline (Val-Calibrated)':<40} {base_auc:>8.4f} {base_pr_auc:>8.4f} {base_brier:>7.4f} {base_cal['ece']:>10.4f} {base_cal['macro_ece']:>10.4f}")
    print(f"{'New Proposed (OOF-Calibrated)':<40} {oof_auc:>8.4f} {oof_pr_auc:>8.4f} {oof_brier:>7.4f} {oof_cal['ece']:>10.4f} {oof_cal['macro_ece']:>10.4f}")
    print("=" * 105)

    print("\n" + "=" * 115)
    print("  OOF-CALIBRATED POLICIES ON UNTOUCHED TEST SET")
    print("=" * 115)
    p_header = f"{'Policy':<30} {'Thresholds':<28} {'Prec':>6} {'Rec':>6} {'Spec':>6} {'F1':>6} {'Cost/App':>9} {'Total Cost':>12}"
    print(p_header)
    print("-" * len(p_header))
    for pol in [oof_pol_f1, oof_pol_bacc, oof_pol_c10, oof_pol_c5, oof_pol_3tier]:
        print(
            f"{pol['policy_name']:<30} {pol['thresholds']:<28} "
            f"{pol['precision']:>6.4f} {pol['recall']:>6.4f} {pol['specificity']:>6.4f} {pol['f1']:>6.4f} "
            f"${pol['cost_per_applicant']:>8.2f} ${pol['total_expected_cost']:>11,.0f}"
        )
    print("=" * 115)

    return comparison_report


if __name__ == "__main__":
    run_test_evaluation_benchmark()
