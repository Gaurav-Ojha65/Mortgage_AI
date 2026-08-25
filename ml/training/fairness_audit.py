"""
Fairness Audit Module — Mortgage AI
===================================
Fair-lending-oriented fairness audit evaluating the frozen OOF-calibrated LightGBM
and frozen 3-tier economic decision policy on the untouched test dataset.

Evaluates demographic/proxy attributes:
- Home Ownership (0: Rent, 1: Own, 2: Mortgage)
- Age/Career Tenure Proxy via Employment Years (< 10 yrs, 10-25 yrs, 25+ yrs)
- Income Bracket (< $40k Low, $40k-$80k Middle, > $80k High)
- Loan Purpose (0, 1, 2, 3, 4)

IMPORTANT DISCLAIMER:
This is an analytical fairness audit designed for risk governance and fair-lending
decision support. It does NOT constitute a legal certification of compliance with
the Equal Credit Opportunity Act (ECOA), Fair Housing Act, or other regulatory frameworks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
import joblib

from ml.inference.predict import MODEL_FEATURES
from risk.decision_policy import DecisionPolicy, DecisionState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Calculate weighted and macro Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    weighted_ece = 0.0
    macro_ece_sum = 0.0
    valid_bins = 0
    total_samples = len(y_true)

    for b in range(n_bins):
        mask = (bin_indices == b)
        count = np.sum(mask)
        if count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            err = np.abs(bin_acc - bin_conf)
            weighted_ece += (count / total_samples) * err
            macro_ece_sum += err
            valid_bins += 1

    macro_ece = macro_ece_sum / valid_bins if valid_bins > 0 else 0.0
    return {"weighted_ece": float(weighted_ece), "macro_ece": float(macro_ece)}


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
    ece_res = compute_ece(y_true, y_prob_cal, n_bins=10)

    # ROC-AUC (only valid if both classes present)
    if n_pos > 0 and n_neg > 0:
        roc_auc = float(roc_auc_score(y_true, y_prob_cal))
    else:
        roc_auc = None

    # Binary metrics using policy approve threshold as decision boundary (p > approve_threshold -> flagged)
    y_pred_flagged = (y_prob_cal > policy.approve_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_flagged, labels=[0, 1]).ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # TPR
    tpr = recall
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # TNR

    # Policy 3-tier routing
    decisions = [policy.decide(float(p))["decision"] for p in y_prob_cal]
    app_mask = np.array([d == DecisionState.APPROVE.value for d in decisions])
    rev_mask = np.array([d == DecisionState.MANUAL_REVIEW.value for d in decisions])
    rej_mask = np.array([d == DecisionState.REJECT.value for d in decisions])

    n_app = int(np.sum(app_mask))
    n_rev = int(np.sum(rev_mask))
    n_rej = int(np.sum(rej_mask))

    approval_rate = float(n_app / n)
    review_rate = float(n_rev / n)
    rejection_rate = float(n_rej / n)

    # Risk within tiers
    approved_defaults = int(np.sum(y_true[app_mask])) if n_app > 0 else 0
    approved_default_rate = float(approved_defaults / n_app) if n_app > 0 else 0.0

    rejected_non_defaults = int(np.sum(1 - y_true[rej_mask])) if n_rej > 0 else 0
    rejected_non_default_rate = float(rejected_non_defaults / n_rej) if n_rej > 0 else 0.0

    return {
        "sample_size": n,
        "default_count": n_pos,
        "non_default_count": n_neg,
        "observed_default_rate": round(obs_default_rate, 4),
        "mean_predicted_probability": round(mean_pred_prob, 4),
        "brier_score": round(brier, 4),
        "weighted_ece": round(ece_res["weighted_ece"], 4),
        "macro_ece": round(ece_res["macro_ece"], 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "precision": round(precision, 4),
        "recall_tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "specificity": round(specificity, 4),
        "approval_count": n_app,
        "approval_rate": round(approval_rate, 4),
        "manual_review_count": n_rev,
        "manual_review_rate": round(review_rate, 4),
        "rejection_count": n_rej,
        "rejection_rate": round(rejection_rate, 4),
        "approved_defaults": approved_defaults,
        "approved_default_rate": round(approved_default_rate, 4),
        "rejected_non_defaults": rejected_non_defaults,
        "rejected_non_default_rate": round(rejected_non_default_rate, 4),
    }


def run_fairness_audit(
    test_csv_path: str = "data/test.csv",
    pipeline_path: str = "ml/models/lightgbm_calibrated_pipeline.joblib",
    output_report_path: str = "reports/metrics/fairness_report.json",
) -> Dict[str, Any]:
    """Execute complete fairness audit on untouched test set."""
    # Resolve pipeline path if fallback needed
    p_path = Path(pipeline_path)
    if not p_path.exists():
        if Path("models/lightgbm_calibrated_pipeline.joblib").exists():
            p_path = Path("models/lightgbm_calibrated_pipeline.joblib")
        else:
            raise FileNotFoundError(f"Pipeline artifact not found at {pipeline_path} or models/")
    test_df = pd.read_csv(test_csv_path)
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["target"].values

    pipeline = joblib.load(p_path)
    y_prob_cal = pipeline.predict_proba(X_test)[:, 1]

    # Frozen 3-tier policy
    frozen_policy = DecisionPolicy(
        policy_name="frozen_oof_3tier_policy",
        policy_version="v3.0-frozen",
        approve_threshold=0.055,
        reject_threshold=0.405,
        description="Frozen 3-tier policy (Approve <= 0.055, Reject >= 0.405)",
    )

    # 1. Overall Portfolio Benchmark
    overall_metrics = evaluate_subgroup_metrics(y_test, y_prob_cal, frozen_policy)

    # 2. Define Subgroup Categories
    # Attribute 1: Home Ownership (0: Rent, 1: Own, 2: Mortgage)
    home_labels = {0: "Rent", 1: "Own", 2: "Mortgage"}
    test_df["home_ownership_cat"] = test_df["home_ownership"].map(home_labels).fillna("Unknown")

    # Attribute 2: Employment / Career Tenure Proxy (< 10, 10-25, 25+)
    test_df["career_tenure_proxy"] = pd.cut(
        test_df["employment_years"],
        bins=[-1, 10, 25, 100],
        labels=["Early Career (<10y)", "Mid Career (10-25y)", "Senior (25y+)"],
    ).astype(str)

    # Attribute 3: Income Bracket (< $40k Low, $40k-$80k Middle, > $80k High)
    test_df["income_bracket"] = pd.cut(
        test_df["annual_income"],
        bins=[-1, 40000, 80000, 10000000],
        labels=["Low (<$40k)", "Middle ($40k-$80k)", "High (>$80k)"],
    ).astype(str)

    # Attribute 4: Purpose Encoded (0, 1, 2, 3, 4)
    purpose_labels = {0: "Home Purchase", 1: "Refinance", 2: "Debt Consolidation", 3: "Home Improvement", 4: "Other"}
    test_df["purpose_cat"] = test_df["purpose_encoded"].map(purpose_labels).fillna("Category Other")

    sensitive_attributes = {
        "home_ownership": "home_ownership_cat",
        "career_tenure_proxy": "career_tenure_proxy",
        "income_bracket": "income_bracket",
        "loan_purpose": "purpose_cat",
    }

    subgroup_reports: Dict[str, Any] = {}
    fairness_disparities: Dict[str, Any] = {}

    for attr_key, col_name in sensitive_attributes.items():
        groups = test_df[col_name].unique()
        group_metrics: Dict[str, Any] = {}
        approval_rates: Dict[str, float] = {}
        tpr_rates: Dict[str, float] = {}
        fpr_rates: Dict[str, float] = {}

        for grp in sorted(groups):
            mask = (test_df[col_name] == grp).values
            y_sub = y_test[mask]
            p_sub = y_prob_cal[mask]
            metrics = evaluate_subgroup_metrics(y_sub, p_sub, frozen_policy)
            group_metrics[str(grp)] = metrics
            approval_rates[str(grp)] = metrics["approval_rate"]
            tpr_rates[str(grp)] = metrics["recall_tpr"]
            fpr_rates[str(grp)] = metrics["fpr"]

        # Disparity Calculations
        app_vals = list(approval_rates.values())
        tpr_vals = list(tpr_rates.values())
        fpr_vals = list(fpr_rates.values())

        demographic_parity_diff = float(max(app_vals) - min(app_vals))
        disparate_impact_ratio = float(min(app_vals) / max(app_vals)) if max(app_vals) > 0 else 1.0
        equal_opportunity_diff = float(max(tpr_vals) - min(tpr_vals))
        equalized_odds_diff = float(max(equal_opportunity_diff, max(fpr_vals) - min(fpr_vals)))

        subgroup_reports[attr_key] = group_metrics
        fairness_disparities[attr_key] = {
            "demographic_parity_difference": round(demographic_parity_diff, 4),
            "disparate_impact_ratio": round(disparate_impact_ratio, 4),
            "equal_opportunity_difference": round(equal_opportunity_diff, 4),
            "equalized_odds_difference": round(equalized_odds_diff, 4),
            "highest_approval_group": max(approval_rates, key=approval_rates.get),
            "lowest_approval_group": min(approval_rates, key=approval_rates.get),
        }

    full_report = {
        "metadata": {
            "audit_type": "fairness_audit",
            "eval_dataset": "data/test.csv",
            "sample_size": len(y_test),
            "model": "LightGBM",
            "calibration": "OOF Isotonic",
            "policy": "Frozen 3-Tier Policy (Approve <= 0.055, Reject >= 0.405)",
            "disclaimer": (
                "Fair-lending fairness analysis for model risk management and governance. "
                "Does not constitute legal certification under ECOA or the Fair Housing Act."
            ),
        },
        "overall_portfolio": overall_metrics,
        "disparity_summary": fairness_disparities,
        "subgroups": subgroup_reports,
    }

    # Ensure output dir exists and save
    out_path = Path(output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Fairness audit report successfully saved to {out_path}")
    return full_report


if __name__ == "__main__":
    run_fairness_audit()
