"""
Policy Sensitivity Analysis — Mortgage AI
==========================================
Evaluates a 2D grid of (approve_threshold, reject_threshold) configurations
around the frozen canonical policy (Approve <= 0.055, Reject >= 0.405) on the
untouched test dataset.

IMPORTANT:
This is sensitivity analysis ONLY. It does NOT select a new threshold.
The frozen canonical policy remains the production/demo candidate.
No threshold is optimized using test labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib

from ml.inference.predict import MODEL_FEATURES
from risk.decision_policy import DecisionPolicy, DecisionState, CostModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_policy_config(
    y_true: np.ndarray,
    y_prob_cal: np.ndarray,
    approve_thresh: float,
    reject_thresh: float,
    cost_model: CostModel,
) -> Dict[str, Any]:
    """Evaluate a single policy configuration on test data."""
    n = len(y_true)

    # 3-tier routing
    app_mask = y_prob_cal <= approve_thresh
    rej_mask = y_prob_cal >= reject_thresh
    rev_mask = ~app_mask & ~rej_mask

    n_app = int(np.sum(app_mask))
    n_rev = int(np.sum(rev_mask))
    n_rej = int(np.sum(rej_mask))

    approval_pct = round(n_app / n, 4)
    review_pct = round(n_rev / n, 4)
    rejection_pct = round(n_rej / n, 4)

    # Defaults in each tier
    approved_defaults = int(np.sum(y_true[app_mask])) if n_app > 0 else 0
    rejected_non_defaults = int(np.sum(1 - y_true[rej_mask])) if n_rej > 0 else 0
    rejected_defaults = int(np.sum(y_true[rej_mask])) if n_rej > 0 else 0

    # Cost calculation
    fn_cost = approved_defaults * cost_model.cost_fn
    fp_cost = rejected_non_defaults * cost_model.cost_fp
    review_cost = n_rev * cost_model.cost_manual_review
    total_cost = fn_cost + fp_cost + review_cost
    cost_per_app = round(total_cost / n, 2)

    # Binary metrics (flagged = p > approve_thresh)
    y_pred_flagged = (y_prob_cal > approve_thresh).astype(int)
    tp = int(np.sum((y_pred_flagged == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_flagged == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred_flagged == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred_flagged == 0) & (y_true == 1)))

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0

    review_constraint_met = review_pct <= 0.25

    return {
        "approve_threshold": approve_thresh,
        "reject_threshold": reject_thresh,
        "approval_pct": approval_pct,
        "review_pct": review_pct,
        "rejection_pct": rejection_pct,
        "review_volume": n_rev,
        "approved_defaults": approved_defaults,
        "rejected_non_defaults": rejected_non_defaults,
        "rejected_defaults": rejected_defaults,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "review_cost": review_cost,
        "total_expected_cost": total_cost,
        "cost_per_applicant": cost_per_app,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "review_constraint_met": review_constraint_met,
    }


def run_policy_sensitivity(
    test_csv_path: str = "data/test.csv",
    pipeline_path: str = "ml/models/lightgbm_calibrated_pipeline.joblib",
    output_report_path: str = "reports/metrics/policy_sensitivity.json",
) -> Dict[str, Any]:
    """Execute full 2D grid sensitivity analysis around frozen policy."""
    p_path = Path(pipeline_path)
    if not p_path.exists():
        if Path("models/lightgbm_calibrated_pipeline.joblib").exists():
            p_path = Path("models/lightgbm_calibrated_pipeline.joblib")
        else:
            raise FileNotFoundError(f"Pipeline artifact not found at {pipeline_path}")

    test_df = pd.read_csv(test_csv_path)
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["target"].values

    pipeline = joblib.load(p_path)
    y_prob_cal = pipeline.predict_proba(X_test)[:, 1]

    cost_model = CostModel(cost_fn=10000.0, cost_fp=1000.0, cost_manual_review=150.0)

    approve_thresholds = [0.045, 0.050, 0.055, 0.060, 0.065]
    reject_thresholds = [0.385, 0.395, 0.405, 0.415, 0.425]

    grid_results: List[Dict[str, Any]] = []
    frozen_result = None

    for app_t in approve_thresholds:
        for rej_t in reject_thresholds:
            if app_t >= rej_t:
                continue  # Invalid configuration
            result = evaluate_policy_config(y_test, y_prob_cal, app_t, rej_t, cost_model)
            grid_results.append(result)
            if app_t == 0.055 and rej_t == 0.405:
                frozen_result = result

    # Analysis: robust region, constraint violations, cost sensitivity
    valid_configs = [r for r in grid_results if r["review_constraint_met"]]
    violating_configs = [r for r in grid_results if not r["review_constraint_met"]]
    costs = [r["total_expected_cost"] for r in grid_results]
    min_cost_config = min(grid_results, key=lambda x: x["total_expected_cost"])
    max_cost_config = max(grid_results, key=lambda x: x["total_expected_cost"])

    # Cost gradient around frozen policy
    cost_gradient = []
    if frozen_result:
        frozen_cost = frozen_result["total_expected_cost"]
        for r in grid_results:
            delta_app = r["approve_threshold"] - 0.055
            delta_rej = r["reject_threshold"] - 0.405
            cost_delta = r["total_expected_cost"] - frozen_cost
            cost_gradient.append({
                "approve_threshold": r["approve_threshold"],
                "reject_threshold": r["reject_threshold"],
                "delta_approve": round(delta_app, 4),
                "delta_reject": round(delta_rej, 4),
                "cost_delta": cost_delta,
                "cost_delta_pct": round(cost_delta / frozen_cost * 100, 2) if frozen_cost > 0 else 0,
                "review_constraint_met": r["review_constraint_met"],
            })

    report = {
        "metadata": {
            "analysis_type": "policy_sensitivity_analysis",
            "eval_dataset": "data/test.csv",
            "sample_size": len(y_test),
            "frozen_canonical_policy": {"approve_threshold": 0.055, "reject_threshold": 0.405},
            "cost_model": {
                "cost_fn": cost_model.cost_fn,
                "cost_fp": cost_model.cost_fp,
                "cost_manual_review": cost_model.cost_manual_review,
                "is_demonstration": True,
            },
            "disclaimer": (
                "Sensitivity analysis only. No threshold was selected or optimized from test results. "
                "The frozen canonical policy remains the production/demo candidate."
            ),
        },
        "frozen_policy_result": frozen_result,
        "grid_summary": {
            "total_configurations_evaluated": len(grid_results),
            "configurations_meeting_review_constraint": len(valid_configs),
            "configurations_violating_review_constraint": len(violating_configs),
            "min_cost_configuration": min_cost_config,
            "max_cost_configuration": max_cost_config,
            "cost_range": [min(costs), max(costs)],
        },
        "cost_gradient_around_frozen": cost_gradient,
        "full_grid_results": grid_results,
    }

    out_path = Path(output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Policy sensitivity report saved to {out_path}")
    return report


if __name__ == "__main__":
    run_policy_sensitivity()
