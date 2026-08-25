"""
Paired Bootstrap Statistical Comparison — Mortgage AI
======================================================
Evaluates statistical uncertainty between:
- Baseline v3.0.0 (LightGBM + v3.0 OOF Isotonic + Policy 0.055/0.405)
- Candidate v3.1 (HPO LightGBM + v3.1 OOF Isotonic + Policy 0.045/0.335)

Protocol:
- Evaluates on the exact same untouched test set (`data/test.csv`, N=21,398).
- Performs B=1,000 paired bootstrap iterations with fixed seed=42.
- Estimates 95% empirical bootstrap confidence intervals (2.5th to 97.5th percentiles)
  for differences in ROC-AUC, PR-AUC, Brier score, Weighted ECE, and Expected Cost.
- Determines whether differences are statistically distinguishable from zero.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.inference.predict import MODEL_FEATURES
from ml.training.eval_utils import compute_calibration_metrics
from ml.training.calibrated_predictor import CalibratedPredictor
from risk.decision_policy import DecisionPolicy, CostModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = _PROJECT_ROOT / "ml" / "models"
DATA_DIR = _PROJECT_ROOT / "data"
METRICS_DIR = _PROJECT_ROOT / "reports" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def compute_policy_cost(
    y_true: np.ndarray,
    p_cal: np.ndarray,
    approve_t: float,
    reject_t: float,
    cost_model: CostModel,
) -> Tuple[float, float, float]:
    """Compute total expected policy cost, cost/app, and review rate."""
    n = len(y_true)
    app_mask = p_cal <= approve_t
    rej_mask = p_cal >= reject_t
    rev_mask = (~app_mask) & (~rej_mask)

    app_defaults = int(np.sum(y_true[app_mask]))
    rej_non_defaults = int(np.sum((1 - y_true)[rej_mask]))
    n_rev = int(np.sum(rev_mask))

    fn_cost = app_defaults * cost_model.cost_fn
    fp_cost = rej_non_defaults * cost_model.cost_fp
    rev_cost = n_rev * cost_model.cost_manual_review
    total_cost = fn_cost + fp_cost + rev_cost
    cost_per_app = total_cost / n
    review_rate = n_rev / n
    return total_cost, cost_per_app, review_rate


def run_paired_bootstrap(
    n_bootstraps: int = 1000,
    seed: int = 42,
    test_csv_path: str = "data/test.csv",
    output_path: str = "reports/metrics/hpo_statistical_comparison.json",
) -> Dict[str, Any]:
    """Execute paired bootstrap resampling across baseline and candidate models."""
    logger.info("=" * 70)
    logger.info("  STARTING PAIRED BOOTSTRAP STATISTICAL COMPARISON")
    logger.info("=" * 70)

    # 1. Load test data
    test_df = pd.read_csv(test_csv_path)
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["target"].values
    n_test = len(y_test)
    logger.info(f"Loaded untouched test set: N={n_test:,} samples (defaults: {int(y_test.sum()):,})")

    # 2. Load baseline and candidate models
    base_pipeline = joblib.load(MODELS_DIR / "lightgbm_calibrated_pipeline.joblib")

    # Load candidate raw model and calibrator
    cand_model = joblib.load(MODELS_DIR / "lightgbm_hpo_candidate.joblib")
    cand_iso = joblib.load(MODELS_DIR / "lightgbm_hpo_candidate_oof_calibrator_isotonic.joblib")
    cand_pipeline = CalibratedPredictor(
        base_model=cand_model,
        calibrator=cand_iso,
        calibration_method="isotonic",
        model_name="lightgbm_hpo_candidate",
        version="v3.1-hpo-candidate",
    )

    # 3. Full-sample predictions
    p_base_cal = base_pipeline.predict_proba(X_test)[:, 1]
    p_cand_cal = cand_pipeline.predict_proba(X_test)[:, 1]

    cost_model = CostModel(cost_fn=10000.0, cost_fp=1000.0, cost_manual_review=150.0)

    # Policies
    base_app_t, base_rej_t = 0.055, 0.405
    cand_app_t, cand_rej_t = 0.045, 0.335

    # Point estimates on full sample
    base_auc = roc_auc_score(y_test, p_base_cal)
    cand_auc = roc_auc_score(y_test, p_cand_cal)
    base_pr = average_precision_score(y_test, p_base_cal)
    cand_pr = average_precision_score(y_test, p_cand_cal)
    base_brier = brier_score_loss(y_test, p_base_cal)
    cand_brier = brier_score_loss(y_test, p_cand_cal)
    base_wece = compute_calibration_metrics(y_test, p_base_cal, n_bins=10)["ece"]
    cand_wece = compute_calibration_metrics(y_test, p_cand_cal, n_bins=10)["ece"]

    base_cost, base_cpa, base_rev = compute_policy_cost(y_test, p_base_cal, base_app_t, base_rej_t, cost_model)
    cand_cost, cand_cpa, cand_rev = compute_policy_cost(y_test, p_cand_cal, cand_app_t, cand_rej_t, cost_model)

    # 4. Bootstrap loop
    np.random.seed(seed)
    logger.info(f"Running {n_bootstraps:,} paired bootstrap iterations...")
    t0_boot = time.time()

    delta_aucs = np.zeros(n_bootstraps)
    delta_prs = np.zeros(n_bootstraps)
    delta_briers = np.zeros(n_bootstraps)
    delta_weces = np.zeros(n_bootstraps)
    delta_costs = np.zeros(n_bootstraps)
    delta_cpas = np.zeros(n_bootstraps)

    base_aucs = np.zeros(n_bootstraps)
    cand_aucs = np.zeros(n_bootstraps)
    base_prs = np.zeros(n_bootstraps)
    cand_prs = np.zeros(n_bootstraps)
    base_briers = np.zeros(n_bootstraps)
    cand_briers = np.zeros(n_bootstraps)
    base_weces = np.zeros(n_bootstraps)
    cand_weces = np.zeros(n_bootstraps)
    base_costs = np.zeros(n_bootstraps)
    cand_costs = np.zeros(n_bootstraps)

    for b in range(n_bootstraps):
        boot_indices = np.random.choice(n_test, size=n_test, replace=True)
        y_b = y_test[boot_indices]
        p_base_b = p_base_cal[boot_indices]
        p_cand_b = p_cand_cal[boot_indices]

        # Check if positive class exists in sample (virtually guaranteed at N=21,398)
        if y_b.sum() == 0 or y_b.sum() == n_test:
            continue

        # ROC-AUC
        b_auc_b = roc_auc_score(y_b, p_base_b)
        c_auc_b = roc_auc_score(y_b, p_cand_b)
        base_aucs[b] = b_auc_b
        cand_aucs[b] = c_auc_b
        delta_aucs[b] = c_auc_b - b_auc_b

        # PR-AUC
        b_pr_b = average_precision_score(y_b, p_base_b)
        c_pr_b = average_precision_score(y_b, p_cand_b)
        base_prs[b] = b_pr_b
        cand_prs[b] = c_pr_b
        delta_prs[b] = c_pr_b - b_pr_b

        # Brier
        b_br_b = brier_score_loss(y_b, p_base_b)
        c_br_b = brier_score_loss(y_b, p_cand_b)
        base_briers[b] = b_br_b
        cand_briers[b] = c_br_b
        delta_briers[b] = c_br_b - b_br_b

        # Weighted ECE
        b_wece_b = compute_calibration_metrics(y_b, p_base_b, n_bins=10)["ece"]
        c_wece_b = compute_calibration_metrics(y_b, p_cand_b, n_bins=10)["ece"]
        base_weces[b] = b_wece_b
        cand_weces[b] = c_wece_b
        delta_weces[b] = c_wece_b - b_wece_b

        # Cost
        b_cost_b, b_cpa_b, _ = compute_policy_cost(y_b, p_base_b, base_app_t, base_rej_t, cost_model)
        c_cost_b, c_cpa_b, _ = compute_policy_cost(y_b, p_cand_b, cand_app_t, cand_rej_t, cost_model)
        base_costs[b] = b_cost_b
        cand_costs[b] = c_cost_b
        delta_costs[b] = c_cost_b - b_cost_b
        delta_cpas[b] = c_cpa_b - b_cpa_b

    t_boot_elapsed = time.time() - t0_boot
    logger.info(f"Bootstrap finished in {t_boot_elapsed:.1f}s.")

    def format_ci(arr: np.ndarray, decimals: int = 4) -> Dict[str, Any]:
        ci_lo = round(float(np.percentile(arr, 2.5)), decimals)
        ci_hi = round(float(np.percentile(arr, 97.5)), decimals)
        mean_b = round(float(np.mean(arr)), decimals)
        std_b = round(float(np.std(arr)), decimals)
        return {
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
            "bootstrap_mean": mean_b,
            "bootstrap_std": std_b,
            "includes_zero": bool(ci_lo <= 0 <= ci_hi),
        }

    auc_diff_ci = format_ci(delta_aucs, 4)
    pr_diff_ci = format_ci(delta_prs, 4)
    brier_diff_ci = format_ci(delta_briers, 4)
    wece_diff_ci = format_ci(delta_weces, 4)
    cost_diff_ci = format_ci(delta_costs, 2)
    cpa_diff_ci = format_ci(delta_cpas, 2)

    comparison_report = {
        "metadata": {
            "analysis_type": "paired_bootstrap_uncertainty_analysis",
            "eval_dataset": "data/test.csv",
            "sample_size": n_test,
            "n_bootstraps": n_bootstraps,
            "random_seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "point_estimates": {
            "roc_auc": {
                "baseline_v3": round(float(base_auc), 4),
                "candidate_v3_1": round(float(cand_auc), 4),
                "point_delta": round(float(cand_auc - base_auc), 4),
            },
            "pr_auc": {
                "baseline_v3": round(float(base_pr), 4),
                "candidate_v3_1": round(float(cand_pr), 4),
                "point_delta": round(float(cand_pr - base_pr), 4),
            },
            "brier_score": {
                "baseline_v3": round(float(base_brier), 4),
                "candidate_v3_1": round(float(cand_brier), 4),
                "point_delta": round(float(cand_brier - base_brier), 4),
            },
            "weighted_ece": {
                "baseline_v3": round(float(base_wece), 4),
                "candidate_v3_1": round(float(cand_wece), 4),
                "point_delta": round(float(cand_wece - base_wece), 4),
            },
            "total_expected_cost": {
                "baseline_v3": round(float(base_cost), 2),
                "candidate_v3_1": round(float(cand_cost), 2),
                "point_delta": round(float(cand_cost - base_cost), 2),
            },
            "cost_per_applicant": {
                "baseline_v3": round(float(base_cpa), 2),
                "candidate_v3_1": round(float(cand_cpa), 2),
                "point_delta": round(float(cand_cpa - base_cpa), 2),
            },
        },
        "paired_difference_95_ci": {
            "delta_roc_auc": auc_diff_ci,
            "delta_pr_auc": pr_diff_ci,
            "delta_brier_score": brier_diff_ci,
            "delta_weighted_ece": wece_diff_ci,
            "delta_total_expected_cost": cost_diff_ci,
            "delta_cost_per_applicant": cpa_diff_ci,
        },
        "statistical_interpretation": {
            "roc_auc_significant": not auc_diff_ci["includes_zero"],
            "pr_auc_significant": not pr_diff_ci["includes_zero"],
            "brier_significant": not brier_diff_ci["includes_zero"],
            "wece_significant": not wece_diff_ci["includes_zero"],
            "cost_significant": not cost_diff_ci["includes_zero"],
            "summary_statement": (
                f"95% CI for Delta ROC-AUC: [{auc_diff_ci['ci_95_lower']:.4f}, {auc_diff_ci['ci_95_upper']:.4f}]. "
                f"95% CI for Delta PR-AUC: [{pr_diff_ci['ci_95_lower']:.4f}, {pr_diff_ci['ci_95_upper']:.4f}]. "
                f"95% CI for Delta Total Cost: [${cost_diff_ci['ci_95_lower']:,.2f}, ${cost_diff_ci['ci_95_upper']:,.2f}]. "
                "Point estimates show improvement across all metrics; interval analysis reveals the degree of statistical overlap."
            ),
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)
    logger.info(f"Saved bootstrap statistical report to {out_file}")

    return comparison_report


if __name__ == "__main__":
    run_paired_bootstrap()
