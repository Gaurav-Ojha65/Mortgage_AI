"""
Subgroup Calibration Audit Module — Mortgage AI
================================================
Audits probability calibration reliability across demographic and proxy subgroups
using the frozen OOF-calibrated LightGBM model on untouched test data.

Calculates:
- Brier score
- Weighted ECE
- Macro ECE
- Mean predicted probability
- Actual default prevalence (observed rate)
- Absolute calibration gap (mean_pred - actual)
- 10-bin reliability curve data (bin accuracy, confidence, sample count)

IMPORTANT:
Does NOT fit any subgroup-specific calibrators. Evaluates the single global
OOF isotonic calibrator across cohorts to diagnose calibration transferability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
import joblib

from ml.inference.predict import MODEL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_subgroup_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute 10-bin reliability diagram metrics and ECE for a subgroup.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    total_samples = len(y_true)
    weighted_ece = 0.0
    macro_ece_sum = 0.0
    valid_bins = 0

    bins_data: List[Dict[str, Any]] = []

    for b in range(n_bins):
        bin_low = float(bin_edges[b])
        bin_high = float(bin_edges[b + 1])
        mask = (bin_indices == b)
        count = int(np.sum(mask))

        if count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            err = abs(bin_acc - bin_conf)
            weighted_ece += (count / total_samples) * err
            macro_ece_sum += err
            valid_bins += 1

            bins_data.append({
                "bin_index": b,
                "bin_range": [round(bin_low, 2), round(bin_high, 2)],
                "sample_count": count,
                "fraction_of_total": round(count / total_samples, 4),
                "mean_predicted_prob": round(bin_conf, 4),
                "actual_default_rate": round(bin_acc, 4),
                "calibration_error": round(err, 4),
            })
        else:
            bins_data.append({
                "bin_index": b,
                "bin_range": [round(bin_low, 2), round(bin_high, 2)],
                "sample_count": 0,
                "fraction_of_total": 0.0,
                "mean_predicted_prob": None,
                "actual_default_rate": None,
                "calibration_error": None,
            })

    macro_ece = macro_ece_sum / valid_bins if valid_bins > 0 else 0.0
    mean_pred = float(np.mean(y_prob))
    actual_prev = float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    global_cal_gap = mean_pred - actual_prev

    return {
        "sample_size": total_samples,
        "default_count": int(np.sum(y_true)),
        "actual_default_prevalence": round(actual_prev, 4),
        "mean_predicted_probability": round(mean_pred, 4),
        "calibration_gap": round(global_cal_gap, 4),
        "brier_score": round(brier, 4),
        "weighted_ece": round(float(weighted_ece), 4),
        "macro_ece": round(float(macro_ece), 4),
        "bins": bins_data,
    }


def run_subgroup_calibration_audit(
    test_csv_path: str = "data/test.csv",
    pipeline_path: str = "ml/models/lightgbm_calibrated_pipeline.joblib",
    output_report_path: str = "reports/metrics/subgroup_calibration.json",
) -> Dict[str, Any]:
    """Execute complete subgroup calibration audit on untouched test data."""
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

    # Setup categorical slices
    home_labels = {0: "Rent", 1: "Own", 2: "Mortgage"}
    test_df["home_ownership_cat"] = test_df["home_ownership"].map(home_labels).fillna("Unknown")

    test_df["career_tenure_proxy"] = pd.cut(
        test_df["employment_years"],
        bins=[-1, 10, 25, 100],
        labels=["Early Career (<10y)", "Mid Career (10-25y)", "Senior (25y+)"],
    ).astype(str)

    test_df["income_bracket"] = pd.cut(
        test_df["annual_income"],
        bins=[-1, 40000, 80000, 10000000],
        labels=["Low (<$40k)", "Middle ($40k-$80k)", "High (>$80k)"],
    ).astype(str)

    purpose_labels = {0: "Home Purchase", 1: "Refinance", 2: "Debt Consolidation", 3: "Home Improvement", 4: "Other"}
    test_df["purpose_cat"] = test_df["purpose_encoded"].map(purpose_labels).fillna("Category Other")

    sensitive_attributes = {
        "home_ownership": "home_ownership_cat",
        "career_tenure_proxy": "career_tenure_proxy",
        "income_bracket": "income_bracket",
        "loan_purpose": "purpose_cat",
    }

    # Overall portfolio calibration
    overall_cal = compute_subgroup_reliability_curve(y_test, y_prob_cal, n_bins=10)

    subgroup_calibration_results: Dict[str, Any] = {}
    subgroup_rankings: List[Dict[str, Any]] = []

    for attr_key, col_name in sensitive_attributes.items():
        groups = test_df[col_name].unique()
        group_res: Dict[str, Any] = {}

        for grp in sorted(groups):
            mask = (test_df[col_name] == grp).values
            y_sub = y_test[mask]
            p_sub = y_prob_cal[mask]
            cal_res = compute_subgroup_reliability_curve(y_sub, p_sub, n_bins=10)
            group_res[str(grp)] = cal_res

            subgroup_rankings.append({
                "attribute": attr_key,
                "subgroup": str(grp),
                "sample_size": cal_res["sample_size"],
                "weighted_ece": cal_res["weighted_ece"],
                "macro_ece": cal_res["macro_ece"],
                "brier_score": cal_res["brier_score"],
                "calibration_gap": cal_res["calibration_gap"],
                "sample_size_adequate": bool(cal_res["sample_size"] >= 1000),
            })

        subgroup_calibration_results[attr_key] = group_res

    # Rank by weighted ECE
    sorted_by_ece = sorted(subgroup_rankings, key=lambda x: x["weighted_ece"])
    best_calibrated = sorted_by_ece[0]
    worst_calibrated = sorted_by_ece[-1]

    audit_summary = {
        "metadata": {
            "audit_type": "subgroup_calibration_audit",
            "eval_dataset": "data/test.csv",
            "total_test_samples": len(y_test),
            "model_type": "LightGBM + OOF Isotonic Calibrator",
            "binning_strategy": "10 uniform bins [0.0, 1.0]",
        },
        "overall_portfolio": overall_cal,
        "best_calibrated_subgroup": best_calibrated,
        "worst_calibrated_subgroup": worst_calibrated,
        "subgroup_rankings_by_weighted_ece": sorted_by_ece,
        "subgroup_details": subgroup_calibration_results,
    }

    out_path = Path(output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info(f"Subgroup calibration audit successfully saved to {out_path}")
    return audit_summary


if __name__ == "__main__":
    run_subgroup_calibration_audit()
