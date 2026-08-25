"""
System Integration Tests — Fairness, SHAP, Calibrated Inference & Policy
========================================================================
Comprehensive end-to-end integration test suite verifying:
1. Full prediction pipeline:
   Applicant Input -> Feature Preprocessing -> LightGBM Base Model ->
   OOF Isotonic Calibrator -> Calibrated Default Probability ->
   Frozen 3-Tier Policy -> SHAP Explainability -> Audit Metadata.
2. Verification of all required contract fields.
3. Determinism of predictions and feature attributions across repeated invocations.
4. High-risk, borderline, and low-risk routing verification.
5. FastAPI integration via /analyze and /policy/evaluate endpoints.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from ml.inference.predict import (
    predict_single,
    get_model,
    get_calibrated_model,
    prepare_features,
    MODEL_FEATURES,
)
from backend.shap_explainer import explain_decision
from risk.decision_policy import DecisionPolicy, CostModel, get_active_policy
from backend.api import app


@pytest.fixture(scope="module")
def low_risk_applicant():
    """Prime applicant profile (should route to APPROVE)."""
    return {
        "credit_score": 780,
        "annual_income": 120000,
        "loan_amount": 180000,
        "loan_term": 360,
        "dti_ratio": 0.18,
        "employment_years": 12.0,
        "num_credit_lines": 14,
        "num_derogatory_marks": 0,
        "credit_utilization": 0.08,
        "late_payment_severity_score": 1.0,
        "home_ownership": 2,
        "purpose_encoded": 0,
        "num_late_payments": 0,
        "savings_balance": 55000,
        "monthly_expenses": 2200,
    }


@pytest.fixture(scope="module")
def borderline_applicant():
    """Borderline risk profile (should route to MANUAL_REVIEW)."""
    return {
        "credit_score": 680,
        "annual_income": 65000,
        "loan_amount": 160000,
        "loan_term": 36,
        "dti_ratio": 0.32,
        "employment_years": 4.0,
        "num_credit_lines": 6,
        "num_derogatory_marks": 0,
        "credit_utilization": 0.35,
        "late_payment_severity_score": 0.90,
        "home_ownership": 0,
        "purpose_encoded": 1,
        "num_late_payments": 0,
        "savings_balance": 8000,
        "monthly_expenses": 2000,
    }


@pytest.fixture(scope="module")
def distressed_applicant():
    """Severe risk profile (should route to REJECT)."""
    return {
        "credit_score": 490,
        "annual_income": 24000,
        "loan_amount": 320000,
        "loan_term": 180,
        "dti_ratio": 0.88,
        "employment_years": 0.5,
        "num_credit_lines": 2,
        "num_derogatory_marks": 3,
        "credit_utilization": 0.96,
        "late_payment_severity_score": 0.20,
        "home_ownership": 0,
        "purpose_encoded": 2,
        "num_late_payments": 4,
        "savings_balance": 150,
        "monthly_expenses": 1900,
    }


class TestEndToEndPredictionPipeline:
    """Validate the complete data -> model -> calibrator -> policy -> SHAP pipeline."""

    def test_full_pipeline_response_schema(self, low_risk_applicant):
        # 1. Prediction & Policy
        pred_res = predict_single(
            low_risk_applicant,
            model_name="lightgbm",
            calibration_method="isotonic",
        )

        # Verify all mandatory contract keys are present
        expected_keys = [
            "model_name",
            "model_version",
            "calibration_method",
            "calibration_version",
            "raw_default_probability",
            "calibrated_default_probability",
            "approval_probability",
            "decision",
            "risk_tier",
            "expected_economic_cost",
            "policy_version",
            "policy_name",
            "policy_metadata",
        ]
        for key in expected_keys:
            assert key in pred_res, f"Missing contract key: {key}"

        # 2. SHAP Explainability on the exact same base model
        model = get_model("lightgbm")
        shap_res = explain_decision(low_risk_applicant, model, "lightgbm")

        assert "top_factors" in shap_res
        assert "base_value" in shap_res
        assert "all_factors" in shap_res
        assert len(shap_res["all_factors"]) == len(MODEL_FEATURES)

    def test_deterministic_pipeline_execution(self, borderline_applicant):
        """Ensure repeated pipeline calls on identical input produce exact bit-for-bit outputs."""
        model = get_model("lightgbm")
        first_pred = predict_single(borderline_applicant, model_name="lightgbm", calibration_method="isotonic")
        first_shap = explain_decision(borderline_applicant, model, "lightgbm")

        for _ in range(10):
            pred = predict_single(borderline_applicant, model_name="lightgbm", calibration_method="isotonic")
            shap_out = explain_decision(borderline_applicant, model, "lightgbm")

            assert pred["raw_default_probability"] == first_pred["raw_default_probability"]
            assert pred["calibrated_default_probability"] == first_pred["calibrated_default_probability"]
            assert pred["decision"] == first_pred["decision"]
            assert pred["risk_tier"] == first_pred["risk_tier"]
            assert pred["expected_economic_cost"] == first_pred["expected_economic_cost"]

            assert shap_out["base_value"] == first_shap["base_value"]
            for f1, f2 in zip(first_shap["top_factors"], shap_out["top_factors"]):
                assert f1["feature"] == f2["feature"]
                assert f1["shap_value"] == f2["shap_value"]

    def test_three_tier_routing_profiles(self, low_risk_applicant, borderline_applicant, distressed_applicant):
        """Verify routing behavior conforms to frozen 3-tier bounds."""
        policy = DecisionPolicy(approve_threshold=0.045, reject_threshold=0.335)

        res_low = predict_single(low_risk_applicant, model_name="lightgbm", calibration_method="isotonic", policy=policy)
        assert res_low["calibrated_default_probability"] <= 0.045
        assert res_low["decision"] == "APPROVE"

        res_border = predict_single(borderline_applicant, model_name="lightgbm", calibration_method="isotonic", policy=policy)
        assert 0.045 < res_border["calibrated_default_probability"] < 0.335
        assert res_border["decision"] == "MANUAL_REVIEW"

        res_distressed = predict_single(distressed_applicant, model_name="lightgbm", calibration_method="isotonic", policy=policy)
        assert res_distressed["calibrated_default_probability"] >= 0.335
        assert res_distressed["decision"] == "REJECT"


class TestAPIEndToEndIntegration:
    """Validate FastAPI integration with calibrated decision engine."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_api_analyze_with_audit_metadata(self, client):
        payload = {
            "income": 85000,
            "loan_amount": 250000,
            "credit_score": 720,
            "interest_rate": 6.25,
            "loan_term": 30,
            "property_value": 350000,
            "existing_loans": 0,
        }
        resp = client.post("/analyze", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]

        # Check essential fields
        assert "model_name" in data
        assert "calibration_method" in data
        assert "calibrated_default_probability" in data
        assert "raw_default_probability" in data
        assert "decision" in data
        assert "risk_tier" in data
        assert "expected_economic_cost" in data
        assert "policy_version" in data
        assert data["decision"] in ("APPROVE", "CONDITIONAL", "MANUAL_REVIEW", "REJECT")
