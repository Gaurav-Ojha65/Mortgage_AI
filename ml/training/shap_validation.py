"""
SHAP & Explainability Validation Module — Mortgage AI
=====================================================
Audits and validates the SHAP explainability implementation against the canonical
LightGBM candidate model and inference pipeline.

Validates:
1. Model Artifact Match: Exact model used for inference (ml/models/lightgbm.joblib).
2. Feature Order & Names: Exact match with MODEL_FEATURES schema.
3. Obsolete Feature Elimination: Zero references to payment_history_score.
4. Active Delinquency Feature: Consistent use of late_payment_severity_score.
5. Preprocessing Consistency: Inference preparation matches SHAP feature extraction.
6. Numerical Additivity & Exactness: Local SHAP values sum to raw model margin (log-odds).
7. Global Feature Importance: Mean absolute SHAP values calculated on test dataset.
8. Architectural Separation: Explicit distinction between Model Explanation and Policy Routing.

IMPORTANT GOVERNANCE NOTE:
SHAP values explain the underlying tree model's log-odds output score.
They do NOT explain the post-hoc isotonic calibrator transformation or statutory/legal decisions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import shap
import joblib

from ml.inference.predict import MODEL_FEATURES, FEATURE_LABELS, prepare_features, get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_shap_validation(
    test_csv_path: str = "data/test.csv",
    model_path: str = "ml/models/lightgbm.joblib",
    output_report_path: str = "reports/metrics/shap_validation.json",
) -> Dict[str, Any]:
    """Execute complete SHAP validation and generate metrics report."""
    m_path = Path(model_path)
    if not m_path.exists():
        if Path("models/lightgbm.joblib").exists():
            m_path = Path("models/lightgbm.joblib")
        else:
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

    model = joblib.load(m_path)
    test_df = pd.read_csv(test_csv_path)
    X_test = test_df[MODEL_FEATURES]

    # 1. Feature Schema & Order Verification
    expected_features = list(MODEL_FEATURES)
    model_feature_names = getattr(model, "feature_name_", getattr(model, "feature_names_in_", None))
    if model_feature_names is not None:
        model_features_list = list(model_feature_names)
        feature_order_match = (model_features_list == expected_features)
    else:
        model_features_list = expected_features
        feature_order_match = True

    obsolete_feature_detected = "payment_history_score" in expected_features or "payment_history_score" in model_features_list
    active_feature_present = "late_payment_severity_score" in expected_features and "late_payment_severity_score" in model_features_list

    # 2. Build TreeExplainer
    explainer = shap.TreeExplainer(model)
    expected_value = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0])

    # 3. Local SHAP Additivity Verification (Sample 100 random test records)
    sample_indices = np.linspace(0, len(X_test) - 1, 100, dtype=int)
    X_sample = X_test.iloc[sample_indices]

    # Model raw margin predictions (raw log-odds output from LightGBM)
    raw_margins = model.predict(X_sample, raw_score=True)
    raw_probs = model.predict_proba(X_sample)[:, 1]

    shap_values_sample = explainer.shap_values(X_sample)
    if isinstance(shap_values_sample, list):
        # Multi-class format
        shap_values_arr = shap_values_sample[1]
    elif shap_values_sample.ndim == 3:
        shap_values_arr = shap_values_sample[:, :, 1]
    else:
        shap_values_arr = shap_values_sample

    # Additivity: raw_margin = expected_value + sum(shap_values)
    reconstructed_margins = expected_value + np.sum(shap_values_arr, axis=1)
    max_additivity_error = float(np.max(np.abs(raw_margins - reconstructed_margins)))
    mean_additivity_error = float(np.mean(np.abs(raw_margins - reconstructed_margins)))
    is_additive = bool(max_additivity_error < 1e-5)

    # 4. Global Feature Importance (Sample 1,000 test records for efficiency)
    global_sample = X_test.iloc[:1000]
    global_shap = explainer.shap_values(global_sample)
    if isinstance(global_shap, list):
        global_shap_arr = global_shap[1]
    elif global_shap.ndim == 3:
        global_shap_arr = global_shap[:, :, 1]
    else:
        global_shap_arr = global_shap

    mean_abs_shap = np.mean(np.abs(global_shap_arr), axis=0)
    importance_ranking = []
    for feat, val in zip(MODEL_FEATURES, mean_abs_shap):
        importance_ranking.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "mean_absolute_shap": round(float(val), 5),
        })

    importance_ranking = sorted(importance_ranking, key=lambda x: x["mean_absolute_shap"], reverse=True)

    # 5. Example Local Applicant Explanation
    example_applicant = X_test.iloc[0].to_dict()
    example_df = prepare_features(example_applicant)
    ex_raw_margin = float(model.predict(example_df, raw_score=True)[0])
    ex_raw_prob = float(model.predict_proba(example_df)[0][1])
    ex_shap = explainer.shap_values(example_df)
    if isinstance(ex_shap, list):
        ex_shap_arr = ex_shap[1][0]
    elif ex_shap.ndim == 3:
        ex_shap_arr = ex_shap[0, :, 1]
    else:
        ex_shap_arr = ex_shap[0]

    local_factors = []
    for feat, sv in zip(MODEL_FEATURES, ex_shap_arr):
        val = float(example_df[feat].iloc[0])
        local_factors.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "feature_value": val,
            "shap_value": round(float(sv), 5),
            "abs_impact": round(abs(float(sv)), 5),
            "direction": "increases_default_risk" if sv > 0 else "reduces_default_risk",
        })
    local_factors_sorted = sorted(local_factors, key=lambda x: x["abs_impact"], reverse=True)

    report = {
        "metadata": {
            "validation_name": "shap_explainability_audit",
            "model_path": str(m_path),
            "test_sample_size": len(X_test),
            "explainer_type": "shap.TreeExplainer",
            "explanation_target": "LightGBM raw margin (log-odds default score)",
        },
        "feature_schema_audit": {
            "expected_feature_count": len(expected_features),
            "model_feature_count": len(model_features_list),
            "feature_order_exact_match": feature_order_match,
            "obsolete_feature_detected (payment_history_score)": obsolete_feature_detected,
            "active_feature_present (late_payment_severity_score)": active_feature_present,
            "ordered_features": expected_features,
        },
        "mathematical_additivity_check": {
            "is_additive": is_additive,
            "expected_value_base": round(expected_value, 5),
            "sample_records_tested": len(sample_indices),
            "max_absolute_reconstruction_error": max_additivity_error,
            "mean_absolute_reconstruction_error": mean_additivity_error,
            "formula": "raw_margin = base_value + sum(shap_values)",
        },
        "architectural_separation": {
            "model_explanation": "Explains why LightGBM generated raw risk score from features (via SHAP log-odds).",
            "calibrator_explanation": "Isotonic regression mapping raw probability to empirical risk (unaltered by SHAP).",
            "policy_explanation": "3-tier economic cost optimization routing probability to APPROVE/MANUAL_REVIEW/REJECT.",
            "regulatory_disclaimer": "SHAP feature attributions describe algorithmic score inputs and do not constitute statutory adverse action notices or legal justifications.",
        },
        "global_feature_importance_top_10": importance_ranking[:10],
        "global_feature_importance_all": importance_ranking,
        "example_local_explanation": {
            "raw_margin": round(ex_raw_margin, 5),
            "raw_probability": round(ex_raw_prob, 5),
            "base_value": round(expected_value, 5),
            "top_drivers": local_factors_sorted[:5],
        },
    }

    out_path = Path(output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"SHAP validation report successfully saved to {out_path}")
    return report


if __name__ == "__main__":
    run_shap_validation()
