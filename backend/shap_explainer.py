"""
SHAP Explainability Engine
Computes feature contributions for XGBoost, LightGBM, and LogisticRegression.
Cached per model so first call is slow (~2s), subsequent calls are instant.
"""

import json, warnings
import numpy  as np
import pandas as pd
import shap
import joblib

warnings.filterwarnings("ignore")

from ml.inference.predict import MODEL_FEATURES, FEATURE_LABELS

FEATURE_NAMES = list(MODEL_FEATURES)

# ─── Explainer cache ──────────────────────────────────────────────────────────

_explainer_cache: dict = {}


def _get_explainer(model, model_name: str):
    """Build or return cached SHAP explainer for the given model."""
    key = model_name.lower()
    if key in _explainer_cache:
        return _explainer_cache[key]

    print(f"[SHAP] Building explainer for {model_name}...")

    # XGBoost / LightGBM → TreeExplainer (fast, exact)
    if "xgboost" in key or "lightgbm" in key:
        explainer = shap.TreeExplainer(model)

    # Logistic Regression → LinearExplainer
    elif key == "logisticregression":
        inner = model.named_steps["clf"]
        explainer = shap.LinearExplainer(inner, masker=shap.maskers.Independent(
            np.zeros((1, len(FEATURE_NAMES)))
        ))
    else:
        # Fallback: KernelExplainer (slow but universal)
        explainer = shap.KernelExplainer(
            model.predict_proba,
            shap.sample(np.zeros((100, len(FEATURE_NAMES))), 50),
        )

    _explainer_cache[key] = explainer
    return explainer


# ─── Main explain function ────────────────────────────────────────────────────

def explain_decision(
    applicant_dict: dict,
    model,
    model_name: str,
) -> dict:
    """
    Given one applicant's features and a trained model, return:
      - shap_values: signed contribution of each feature to the approval probability
      - base_value:  model's average prediction (starting point)
      - approval_probability: final predicted probability
      - decision: 'approved' | 'rejected'
      - top_factors: sorted list of (feature, value, shap, direction, human_label)
      - plain_english: list of plain-English reason strings

    All values are for the POSITIVE class (approval = 1).
    """
    # Build feature array
    X = pd.DataFrame([{k: applicant_dict.get(k, 0) for k in FEATURE_NAMES}]).astype(float)

    explainer = _get_explainer(model, model_name)

    # ── Compute SHAP values ──────────────────────────────────────────────────
    if "xgboost" in model_name.lower() or "lightgbm" in model_name.lower():
        shap_out = explainer(X)
        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        shap_values = sv[0]
        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base_value = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
        else:
            base_value = float(exp_val)

    elif model_name.lower() == "logisticregression":
        inner     = model.named_steps["clf"]
        scaler    = model.named_steps["scaler"]
        X_scaled  = scaler.transform(X)
        shap_out  = explainer.shap_values(X_scaled)
        shap_values = shap_out[0]
        base_value  = float(explainer.expected_value)

    else:
        shap_out    = explainer.shap_values(X)
        shap_values = shap_out[0]
        base_value  = float(explainer.expected_value[1])

    # ── Final predicted probability ──────────────────────────────────────────
    prob = float(model.predict_proba(X)[0][1])

    # ── Build factor list ────────────────────────────────────────────────────
    factors = []
    for i, feat in enumerate(FEATURE_NAMES):
        sv_i      = float(shap_values[i])
        raw_val   = float(X[feat].iloc[0])
        direction = "positive" if sv_i > 0 else "negative"
        factors.append({
            "feature":     feat,
            "label":       FEATURE_LABELS.get(feat, feat),
            "raw_value":   round(raw_val, 4),
            "shap_value":  round(sv_i, 4),
            "abs_impact":  round(abs(sv_i), 4),
            "direction":   direction,
        })

    factors_sorted = sorted(factors, key=lambda x: x["abs_impact"], reverse=True)

    # ── Plain-English reasons ────────────────────────────────────────────────
    plain = _plain_english(factors_sorted[:5], prob)

    return {
        "decision":            "approved" if prob >= 0.5 else "rejected",
        "approval_probability": round(prob, 4),
        "base_value":           round(base_value, 4),
        "top_factors":          factors_sorted[:10],
        "all_factors":          factors_sorted,
        "plain_english":        plain,
        "model_used":           model_name,
    }


# ─── Plain-English generator ─────────────────────────────────────────────────

def _plain_english(top_factors: list, prob: float) -> list:
    """Turn top SHAP factors into human-readable sentences."""
    reasons = []
    outcome = "approval" if prob >= 0.5 else "rejection"

    templates = {
        "credit_score": {
            "positive": "Your credit score of {v:.0f} is strong and significantly supports your {o}.",
            "negative": "Your credit score of {v:.0f} is below our threshold and is the main factor in your {o}.",
        },
        "dti_ratio": {
            "positive": "Your debt-to-income ratio of {v:.0%} is healthy, which helps your application.",
            "negative": "Your debt-to-income ratio of {v:.0%} is high — lenders prefer under 36%.",
        },
        "annual_income": {
            "positive": "Your annual income of ${v:,.0f} demonstrates strong repayment capacity.",
            "negative": "Your annual income of ${v:,.0f} is low relative to the requested loan amount.",
        },
        "late_payment_severity_score": {
            "positive": "Your payment severity score of {v:.0%} reflects low delinquency history.",
            "negative": "Your payment severity score of {v:.0%} reflects severe past-due delinquency events.",
        },
        "credit_utilization": {
            "positive": "Credit utilization of {v:.0%} is well-managed.",
            "negative": "Credit utilization of {v:.0%} is high — aim to keep it below 30%.",
        },
        "num_derogatory_marks": {
            "positive": "No derogatory marks on your credit record is a strong positive signal.",
            "negative": "{v:.0f} derogatory mark(s) on your record negatively affect your application.",
        },
        "employment_years": {
            "positive": "{v:.0f} years of employment history demonstrates financial stability.",
            "negative": "Limited employment history ({v:.0f} years) adds risk to your application.",
        },
        "num_late_payments": {
            "positive": "No recent late payments strengthens your profile.",
            "negative": "{v:.0f} late payment(s) in the last 12 months is a red flag for lenders.",
        },
        "savings_balance": {
            "positive": "Savings balance of ${v:,.0f} provides a good financial cushion.",
            "negative": "Low savings balance of ${v:,.0f} reduces your safety margin.",
        },
        "loan_amount": {
            "positive": "The requested amount of ${v:,.0f} is proportionate to your profile.",
            "negative": "The requested amount of ${v:,.0f} is high relative to your financial profile.",
        },
    }

    for f in top_factors:
        feat = f["feature"]
        if feat in templates:
            tpl = templates[feat][f["direction"]]
            try:
                reasons.append(tpl.format(v=f["raw_value"], o=outcome))
            except Exception:
                pass

    if not reasons:
        reasons.append(
            f"Based on your overall financial profile, the model {'approved' if prob >= 0.5 else 'rejected'} your application."
        )

    return reasons