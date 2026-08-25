"""
Fairness Re-Validation Module for HPO Candidate — Mortgage AI
=============================================================
Evaluates the HPO Candidate LightGBM pipeline and frozen HPO 3-tier policy
(Approve <= 0.045, Reject >= 0.335) on the untouched test dataset.

Evaluates demographic/proxy attributes:
- Home Ownership (0: Rent, 1: Own, 2: Mortgage)
- Career Tenure / Age Proxy (< 10 yrs, 10-25 yrs, 25+ yrs)
- Income Tiers (< $40k Low, $40k-$80k Middle, > $80k High)
- Loan Purpose (0: Debt Cons, 1: Home Imp, 2: Credit Card, 3: Major Purp, 4: Other)

IMPORTANT DISCLAIMER:
Fairness analysis is provided for internal model governance and fair-lending decision support.
It does NOT constitute legal certification of compliance with ECOA, Fair Housing Act, or other statutes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
import joblib

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.inference.predict import MODEL_FEATURES
from ml.training.calibrated_predictor import CalibratedPredictor
from ml.training.eval_utils import compute_calibration_metrics
from risk.decision_policy import DecisionPolicy, DecisionState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = _PROJECT_ROOT / "ml" / "models"
METRICS_DIR = _PROJECT_ROOT / "reports" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_subgroup_metrics(
    y_true: np.ndarray,
    y_prob_cal: np.ndarray,
    policy: DecisionPolicy,
) -> Dict[str, Any]:
    """Calculate model-level and policy-level metrics for a specific subgroup."""
    n = len(y_true)
    if n == 0:
        return {}

    n_pos = int(np.sum(y_true))
    n_neg = n - n_pos
    obs_default_rate = float(np.mean(y_true))
    mean_pred_prob = float(np.mean(y_prob_cal))
    brier = float(brier_score_loss(y_true, y_prob_cal))
    cal_res = compute_calibration_metrics(y_true, y_prob_cal, n_bins=10)

    # ROC-AUC (only valid if both classes present)
    roc_auc = float(roc_auc_score(y_true, y_prob_cal)) if (n_pos > 0 and n_neg > 0) else None

    # Binary metrics using policy approve threshold as decision boundary (p > approve_threshold -> flagged)
    y_pred_flagged = (y_prob_cal > policy.approve_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_flagged, labels=[0, 1]).ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tpr = recall
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Policy 3-tier routing
    decisions = [policy.decide(float(p))["decision"] for p in y_prob_cal]
    app_mask = np.array([d == DecisionState.APPROVE.value for d in decisions])
    rev_mask = np.array([d == DecisionState.MANUAL_REVIEW.value for d in decisions])
    rej_mask = np.array([d == DecisionState.REJECT.value for d in decisions])

    n_app = int(np.sum(app_mask))
    n_rev = int(np.sum(rev_mask))
    n_rej = int(np.sum(rej_mask))

    app_rate = float(n_app / n)
    rev_rate = float(n_rev / n)
    rej_rate = float(n_rej / n)

    # Defaults within approved group (leakage default rate)
    app_defaults = int(np.sum(y_true[app_mask]))
    app_default_rate = float(app_defaults / n_app) if n_app > 0 else 0.0

    return {
        "sample_size": n,
        "n_defaults": n_pos,
        "n_non_defaults": n_neg,
        "observed_default_rate": round(obs_default_rate, 4),
        "mean_predicted_prob": round(mean_pred_prob, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "brier_score": round(brier, 4),
        "weighted_ece": round(cal_res["ece"], 4),
        "macro_ece": round(cal_res["macro_ece"], 4),
        "approval_rate": round(app_rate, 4),
        "review_rate": round(rev_rate, 4),
        "rejection_rate": round(rej_rate, 4),
        "approved_volume": n_app,
        "review_volume": n_rev,
        "rejected_volume": n_rej,
        "approved_defaults": app_defaults,
        "approved_default_rate": round(app_default_rate, 4),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def compute_disparities(subgroups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute demographic parity, equal opportunity, equalized odds, and disparate impact."""
    app_rates = [m["approval_rate"] for m in subgroups.values() if "approval_rate" in m]
    tprs = [m["tpr"] for m in subgroups.values() if "tpr" in m]
    fprs = [m["fpr"] for m in subgroups.values() if "fpr" in m]

    if not app_rates:
        return {}

    max_app = max(app_rates)
    min_app = min(app_rates)
    dp_diff = round(max_app - min_app, 4)
    di_ratio = round(min_app / max_app, 4) if max_app > 0 else 1.0

    max_tpr = max(tprs)
    min_tpr = min(tprs)
    eq_opp_diff = round(max_tpr - min_tpr, 4)

    max_fpr = max(fprs)
    min_fpr = min(fprs)
    eq_odds_diff = round(max(max_tpr - min_tpr, max_fpr - min_fpr), 4)

    return {
        "demographic_parity_difference": dp_diff,
        "disparate_impact_ratio": di_ratio,
        "equal_opportunity_difference": eq_opp_diff,
        "equalized_odds_difference": eq_odds_diff,
    }


def run_hpo_fairness_audit(
    test_csv_path: str = "data/test.csv",
    output_report_path: str = "reports/metrics/hpo_fairness_report.json",
) -> Dict[str, Any]:
    """Run fairness audit for HPO candidate pipeline."""
    # 1. Load test data
    test_df = pd.read_csv(test_csv_path)
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["target"].values

    # 2. Load HPO candidate pipeline
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

    # 3. HPO Decision Policy (Approve <= 0.045, Reject >= 0.335)
    policy = DecisionPolicy(
        policy_name="hpo_candidate_policy",
        policy_version="v3.1-hpo",
        approve_threshold=0.045,
        reject_threshold=0.335,
    )

    # 4. Overall portfolio metrics
    overall_metrics = evaluate_subgroup_metrics(y_test, y_prob_cal, policy)

    # 5. Subgroup evaluations
    subgroups: Dict[str, Dict[str, Any]] = {}

    # A. Home Ownership (0: Rent, 1: Own, 2: Mortgage)
    home_map = {0: "Rent", 1: "Own", 2: "Mortgage"}
    home_groups = {}
    for code, label in home_map.items():
        mask = (test_df["home_ownership"] == code).values
        home_groups[label] = evaluate_subgroup_metrics(y_test[mask], y_prob_cal[mask], policy)
    subgroups["home_ownership"] = home_groups

    # B. Career Tenure (Employment Years proxy)
    tenure_groups = {}
    tenure_masks = {
        "Early Career (< 10 yrs)": (test_df["employment_years"] < 10).values,
        "Mid Career (10-25 yrs)": ((test_df["employment_years"] >= 10) & (test_df["employment_years"] <= 25)).values,
        "Senior Career (> 25 yrs)": (test_df["employment_years"] > 25).values,
    }
    for label, mask in tenure_masks.items():
        tenure_groups[label] = evaluate_subgroup_metrics(y_test[mask], y_prob_cal[mask], policy)
    subgroups["career_tenure"] = tenure_groups

    # C. Income Tiers (<$40k Low, $40k-$80k Middle, >$80k High)
    inc_groups = {}
    inc_masks = {
        "Low Income (< $40k)": (test_df["annual_income"] < 40000).values,
        "Middle Income ($40k-$80k)": ((test_df["annual_income"] >= 40000) & (test_df["annual_income"] <= 80000)).values,
        "High Income (> $80k)": (test_df["annual_income"] > 80000).values,
    }
    for label, mask in inc_masks.items():
        inc_groups[label] = evaluate_subgroup_metrics(y_test[mask], y_prob_cal[mask], policy)
    subgroups["income_bracket"] = inc_groups

    # D. Loan Purpose (0: Debt Cons, 1: Home Imp, 2: Credit Card, 3: Major Purp, 4: Other)
    purpose_map = {
        0: "Debt Consolidation",
        1: "Home Improvement",
        2: "Credit Card",
        3: "Major Purchase",
        4: "Other / Small Business",
    }
    purp_groups = {}
    for code, label in purpose_map.items():
        mask = (test_df["purpose_encoded"] == code).values
        if np.sum(mask) > 0:
            purp_groups[label] = evaluate_subgroup_metrics(y_test[mask], y_prob_cal[mask], policy)
    subgroups["loan_purpose"] = purp_groups

    # 6. Compute Disparity Summaries
    disparities = {}
    for dim_name, dim_dict in subgroups.items():
        disparities[dim_name] = compute_disparities(dim_dict)

    report = {
        "metadata": {
            "analysis_type": "hpo_candidate_fairness_revalidation",
            "model_version": "v3.1-hpo-candidate",
            "eval_dataset": "data/test.csv",
            "sample_size": len(y_test),
            "policy": {
                "approve_threshold": policy.approve_threshold,
                "reject_threshold": policy.reject_threshold,
            },
            "disclaimer": (
                "Fairness analysis for model risk management and fair-lending decision support. "
                "Does NOT constitute a legal certification of compliance with ECOA, Fair Housing Act, or statutory rules."
            ),
        },
        "overall_portfolio": overall_metrics,
        "subgroups": subgroups,
        "disparities": disparities,
    }

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved HPO fairness report to {out_file}")
    return report


if __name__ == "__main__":
    run_hpo_fairness_audit()
