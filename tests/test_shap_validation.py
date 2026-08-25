"""
Unit Tests for SHAP Explainability Engine & Validation — Mortgage AI
====================================================================
Tests:
1. Exact feature names and schema matching MODEL_FEATURES.
2. Complete absence of obsolete feature 'payment_history_score'.
3. Correct presence of active feature 'late_payment_severity_score'.
4. Feature order consistency across inference, explainer, and model artifact.
5. SHAP output dimensions and shape.
6. Mathematical additivity: f(x) == base_value + sum(shap_values).
7. Explanation consistency with prediction pipeline.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from ml.inference.predict import MODEL_FEATURES, FEATURE_LABELS, prepare_features
from backend.shap_explainer import explain_decision, _get_explainer, FEATURE_NAMES
import shap


@pytest.fixture(scope="module")
def lightgbm_model():
    """Load canonical LightGBM base model artifact."""
    p = Path("ml/models/lightgbm.joblib")
    if not p.exists():
        p = Path("models/lightgbm.joblib")
    if not p.exists():
        pytest.skip("LightGBM model artifact not found")
    return joblib.load(p)


@pytest.fixture(scope="module")
def sample_applicant():
    """Standard applicant feature dictionary."""
    return {
        "credit_score": 680,
        "annual_income": 75000,
        "loan_amount": 250000,
        "loan_term": 360,
        "dti_ratio": 0.28,
        "employment_years": 8.0,
        "num_credit_lines": 12,
        "num_derogatory_marks": 0,
        "credit_utilization": 0.22,
        "late_payment_severity_score": 0.98,
        "home_ownership": 2,
        "purpose_encoded": 0,
        "num_late_payments": 0,
        "savings_balance": 18000,
        "monthly_expenses": 2400,
    }


class TestSHAPSchemaAndFeatures:
    """Test feature names, ordering, and absence of obsolete columns."""

    def test_feature_names_match_model_features(self):
        assert FEATURE_NAMES == list(MODEL_FEATURES), "SHAP FEATURE_NAMES must exactly match MODEL_FEATURES"

    def test_no_obsolete_payment_history_score(self):
        assert "payment_history_score" not in FEATURE_NAMES, "Obsolete 'payment_history_score' must not exist in SHAP"
        assert "payment_history_score" not in MODEL_FEATURES, "Obsolete 'payment_history_score' must not exist in MODEL_FEATURES"

    def test_active_late_payment_severity_score_present(self):
        assert "late_payment_severity_score" in FEATURE_NAMES, "'late_payment_severity_score' must be present in SHAP"
        assert "late_payment_severity_score" in MODEL_FEATURES, "'late_payment_severity_score' must be present in MODEL_FEATURES"

    def test_feature_labels_completeness(self):
        for feat in MODEL_FEATURES:
            assert feat in FEATURE_LABELS, f"Missing human label for feature {feat}"

    def test_model_artifact_feature_names(self, lightgbm_model):
        model_names = getattr(lightgbm_model, "feature_name_", getattr(lightgbm_model, "feature_names_in_", None))
        if model_names is not None:
            assert list(model_names) == list(MODEL_FEATURES), "Model artifact feature names must match MODEL_FEATURES exactly"


class TestSHAPExecutionAndAdditivity:
    """Test SHAP computation, dimensions, and numerical additivity."""

    def test_shap_output_shape(self, lightgbm_model, sample_applicant):
        X = prepare_features(sample_applicant)
        explainer = shap.TreeExplainer(lightgbm_model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        elif shap_vals.ndim == 3:
            sv = shap_vals[:, :, 1]
        else:
            sv = shap_vals
        assert sv.shape == (1, 15), f"Expected SHAP shape (1, 15), got {sv.shape}"

    def test_mathematical_additivity_single_prediction(self, lightgbm_model, sample_applicant):
        X = prepare_features(sample_applicant)
        explainer = shap.TreeExplainer(lightgbm_model)
        raw_margin = float(lightgbm_model.predict(X, raw_score=True)[0])

        base_val = explainer.expected_value
        if not np.isscalar(base_val):
            base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])
        else:
            base_val = float(base_val)

        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        elif shap_vals.ndim == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]

        reconstructed_margin = base_val + np.sum(sv)
        abs_err = abs(raw_margin - reconstructed_margin)
        assert abs_err < 1e-5, f"SHAP additivity violated: raw={raw_margin}, reconstructed={reconstructed_margin}, err={abs_err}"

    def test_explain_decision_function(self, lightgbm_model, sample_applicant):
        result = explain_decision(sample_applicant, lightgbm_model, "lightgbm")
        assert "approval_probability" in result
        assert "base_value" in result
        assert "top_factors" in result
        assert "all_factors" in result
        assert "plain_english" in result
        assert len(result["all_factors"]) == 15
        assert len(result["plain_english"]) > 0

        # Check factor features are all valid
        for factor in result["all_factors"]:
            assert factor["feature"] in MODEL_FEATURES
            assert factor["direction"] in ("positive", "negative")
            assert "raw_value" in factor
            assert "shap_value" in factor
