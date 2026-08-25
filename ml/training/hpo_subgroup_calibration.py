"""
Subgroup Calibration Audit for HPO Candidate — Mortgage AI
===========================================================
Audits the calibration performance (Brier score, Weighted ECE, Macro ECE, 10-bin breakdown)
of the global OOF Isotonic calibrator bundled with the HPO candidate pipeline
across all demographic/proxy cohorts on the untouched test dataset.
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

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.inference.predict import MODEL_FEATURES
from ml.training.calibrated_predictor import CalibratedPredictor
from ml.training.eval_utils import compute_calibration_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = _PROJECT_ROOT / "ml" / "models"
METRICS_DIR = _PROJECT_ROOT / "reports" / "metrics"


def run_hpo_subgroup_calibration(
    test_csv_path: str = "data/test.csv",
    output_report_path: str = "reports/metrics/hpo_subgroup_calibration.json",
) -> Dict[str, Any]:
    """Audit subgroup calibration reliability for HPO candidate."""
    test_df = pd.read_csv(test_csv_path)
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["target"].values

    cand_model = joblib.load(MODELS_DIR / "lightgbm_hpo_candidate.joblib")
    cand_iso = joblib.load(MODELS_DIR / "lightgbm_hpo_candidate_oof_calibrator_isotonic.joblib")
    pipeline = CalibratedPredictor(
        base_model=cand_model,
        calibrator=cand_iso,
        calibration_method="isotonic",
        model_name="lightgbm_hpo_candidate",
        version="v3.1-hpo-candidate",
    )

    y_prob_cal = pipeline.predict_proba(X_test)[:, 1]

    def audit_group(mask: np.ndarray, label: str) -> Dict[str, Any]:
        y_sub = y_test[mask]
        p_sub = y_prob_cal[mask]
        n = len(y_sub)
        if n == 0:
            return {}
        brier = float(brier_score_loss(y_sub, p_sub))
        cal_res = compute_calibration_metrics(y_sub, p_sub, n_bins=10)
        return {
            "group_name": label,
            "sample_size": n,
            "sample_pct": round(n / len(y_test), 4),
            "observed_default_rate": round(float(np.mean(y_sub)), 4),
            "mean_predicted_prob": round(float(np.mean(p_sub)), 4),
            "brier_score": round(brier, 4),
            "weighted_ece": round(cal_res["ece"], 4),
            "macro_ece": round(cal_res["macro_ece"], 4),
            "calibration_gap": round(abs(float(np.mean(p_sub)) - float(np.mean(y_sub))), 4),
            "bins": cal_res["bin_details"],
        }

    # Overall portfolio
    overall = audit_group(np.ones(len(y_test), dtype=bool), "Overall Portfolio")

    # Subgroups
    subgroups = {}

    # Home Ownership
    home_map = {0: "Rent", 1: "Own", 2: "Mortgage"}
    subgroups["home_ownership"] = {
        label: audit_group((test_df["home_ownership"] == code).values, label)
        for code, label in home_map.items()
    }

    # Career Tenure
    tenure_masks = {
        "Early Career (< 10 yrs)": (test_df["employment_years"] < 10).values,
        "Mid Career (10-25 yrs)": ((test_df["employment_years"] >= 10) & (test_df["employment_years"] <= 25)).values,
        "Senior Career (> 25 yrs)": (test_df["employment_years"] > 25).values,
    }
    subgroups["career_tenure"] = {
        label: audit_group(mask, label) for label, mask in tenure_masks.items()
    }

    # Income Tiers
    inc_masks = {
        "Low Income (< $40k)": (test_df["annual_income"] < 40000).values,
        "Middle Income ($40k-$80k)": ((test_df["annual_income"] >= 40000) & (test_df["annual_income"] <= 80000)).values,
        "High Income (> $80k)": (test_df["annual_income"] > 80000).values,
    }
    subgroups["income_bracket"] = {
        label: audit_group(mask, label) for label, mask in inc_masks.items()
    }

    # Loan Purpose
    purpose_map = {
        0: "Debt Consolidation",
        1: "Home Improvement",
        2: "Credit Card",
        3: "Major Purchase",
        4: "Other / Small Business",
    }
    subgroups["loan_purpose"] = {
        label: audit_group((test_df["purpose_encoded"] == code).values, label)
        for code, label in purpose_map.items()
        if np.sum((test_df["purpose_encoded"] == code).values) > 0
    }

    report = {
        "metadata": {
            "analysis_type": "hpo_candidate_subgroup_calibration_audit",
            "model_version": "v3.1-hpo-candidate",
            "eval_dataset": "data/test.csv",
            "sample_size": len(y_test),
        },
        "overall_portfolio": overall,
        "subgroups": subgroups,
    }

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved HPO subgroup calibration report to {out_file}")
    return report


if __name__ == "__main__":
    run_hpo_subgroup_calibration()
