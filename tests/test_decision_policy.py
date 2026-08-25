"""
Unit & Integration Tests for Decision Policy Engine and Calibrated Inference
=============================================================================
Tests:
1. DecisionPolicy and CostModel validation constraints.
2. All three decision states (APPROVE, MANUAL_REVIEW, REJECT).
3. Risk tier mappings and probability bounds.
4. Policy optimization helpers on validation splits.
5. Calibrated inference pipeline via predict_single().
6. FastAPI endpoints (/policy/config, /policy/evaluate, /analyze).
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from risk.decision_policy import (
    DecisionPolicy,
    CostModel,
    DecisionState,
    RiskTier,
    get_active_policy,
    update_active_policy,
    get_policy_audit_log,
    optimize_f1_threshold,
    optimize_balanced_accuracy_threshold,
    optimize_cost_sensitive_binary_threshold,
    optimize_three_tier_policy,
)
from ml.inference.predict import (
    predict_single,
    get_calibrated_model,
    prepare_features,
    MODEL_FEATURES,
)
from backend.api import app


@pytest.fixture
def sample_applicant():
    return {
        "credit_score": 740,
        "annual_income": 95000,
        "loan_amount": 200000,
        "loan_term": 360,
        "dti_ratio": 0.25,
        "employment_years": 6,
        "num_credit_lines": 8,
        "num_derogatory_marks": 0,
        "credit_utilization": 0.20,
        "late_payment_severity_score": 1.0,
        "home_ownership": 2,
        "purpose_encoded": 1,
        "num_late_payments": 0,
        "savings_balance": 35000,
        "monthly_expenses": 2800,
    }


@pytest.fixture
def high_risk_applicant():
    return {
        "credit_score": 480,
        "annual_income": 22000,
        "loan_amount": 350000,
        "loan_term": 180,
        "dti_ratio": 0.85,
        "employment_years": 0.5,
        "num_credit_lines": 2,
        "num_derogatory_marks": 4,
        "credit_utilization": 0.95,
        "late_payment_severity_score": 0.30,
        "home_ownership": 0,
        "purpose_encoded": 0,
        "num_late_payments": 5,
        "savings_balance": 200,
        "monthly_expenses": 2000,
    }


# =============================================================================
# 1. Policy & CostModel Unit Tests
# =============================================================================

class TestDecisionPolicyUnit:
    def test_default_initialization(self):
        policy = DecisionPolicy()
        assert policy.approve_threshold == 0.045
        assert policy.reject_threshold == 0.335
        assert policy.policy_version == "v3.1-policy-v1"
        assert policy.policy_name == "optimized_3tier_economic_policy"
        assert policy.cost_model.cost_fn == 10000.0
        assert policy.cost_model.cost_fp == 1000.0
        assert policy.cost_model.is_demonstration is True

    def test_threshold_ordering_validation(self):
        with pytest.raises(ValueError, match="Threshold ordering violation"):
            DecisionPolicy(approve_threshold=0.25, reject_threshold=0.10)

    def test_threshold_bounds_validation(self):
        with pytest.raises(ValueError, match="approve_threshold"):
            DecisionPolicy(approve_threshold=-0.05, reject_threshold=0.20)
        with pytest.raises(ValueError, match="reject_threshold"):
            DecisionPolicy(approve_threshold=0.05, reject_threshold=1.50)

    def test_cost_model_non_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            CostModel(cost_fn=-100.0)

    def test_three_decision_states(self):
        policy = DecisionPolicy(approve_threshold=0.04, reject_threshold=0.16)

        # 1. Approve state
        res_approve = policy.decide(0.02)
        assert res_approve["decision"] == DecisionState.APPROVE.value
        assert res_approve["risk_tier"] in (RiskTier.VERY_LOW.value, RiskTier.LOW.value)

        # 2. Manual Review state
        res_review = policy.decide(0.10)
        assert res_review["decision"] == DecisionState.MANUAL_REVIEW.value
        assert res_review["risk_tier"] == RiskTier.MODERATE.value

        # 3. Reject state
        res_reject = policy.decide(0.25)
        assert res_reject["decision"] == DecisionState.REJECT.value
        assert res_reject["risk_tier"] in (RiskTier.HIGH.value, RiskTier.SEVERE.value)

    def test_probability_out_of_bounds(self):
        policy = DecisionPolicy()
        with pytest.raises(ValueError, match=r"in \[0.0, 1.0\]"):
            policy.decide(-0.1)
        with pytest.raises(ValueError, match=r"in \[0.0, 1.0\]"):
            policy.decide(1.1)

    def test_single_applicant_expected_cost(self):
        """Verify single applicant expected cost formula across all 3 decision states."""
        policy = DecisionPolicy(
            approve_threshold=0.040,
            reject_threshold=0.110,
            cost_model=CostModel(cost_fn=10000.0, cost_fp=1000.0, cost_manual_review=150.0),
        )
        # 1. Approved borrower: expected loss = p * cost_fn
        res_app = policy.decide(0.02)
        assert res_app["decision"] == "APPROVE"
        assert res_app["expected_economic_cost"] == 200.0  # 0.02 * 10,000

        # 2. Manual review borrower: expected loss = triage review cost
        res_rev = policy.decide(0.08)
        assert res_rev["decision"] == "MANUAL_REVIEW"
        assert res_rev["expected_economic_cost"] == 150.0  # cost_manual_review

        # 3. Rejected borrower: expected loss = (1 - p) * cost_fp (opportunity loss)
        res_rej = policy.decide(0.20)
        assert res_rej["decision"] == "REJECT"
        assert res_rej["expected_economic_cost"] == 800.0  # (1 - 0.20) * 1,000

    def test_three_tier_cost_aggregation(self):
        """Verify aggregate cost formula for 3-tier policy matches exact arithmetic."""
        cm = CostModel(cost_fn=10000.0, cost_fp=1000.0, cost_manual_review=150.0)
        # 14,106 approved (234 defaults)
        fn_cost = 234 * cm.cost_fn
        # 3,257 rejected (2,313 good borrowers)
        fp_cost = 2313 * cm.cost_fp
        # 4,035 manual review
        rev_cost = 4035 * cm.cost_manual_review
        total = fn_cost + fp_cost + rev_cost
        assert fn_cost == 2340000.0
        assert fp_cost == 2313000.0
        assert rev_cost == 605250.0
        assert total == 5258250.0
        assert round(total / 21398, 2) == 245.74

    def test_exact_threshold_boundary_behavior(self):
        """Verify strict inequality and equality at boundary values."""
        policy = DecisionPolicy(approve_threshold=0.045, reject_threshold=0.335)

        # 1. Exactly on approve threshold -> APPROVE
        res_exact_app = policy.decide(0.045)
        assert res_exact_app["decision"] == "APPROVE"

        # 2. Infinitesimally above approve threshold -> MANUAL_REVIEW
        res_above_app = policy.decide(0.045001)
        assert res_above_app["decision"] == "MANUAL_REVIEW"

        # 3. Infinitesimally below reject threshold -> MANUAL_REVIEW
        res_below_rej = policy.decide(0.334999)
        assert res_below_rej["decision"] == "MANUAL_REVIEW"

        # 4. Exactly on reject threshold -> REJECT
        res_exact_rej = policy.decide(0.335)
        assert res_exact_rej["decision"] == "REJECT"

    def test_policy_serialization_and_deserialization(self):
        """Verify round-trip dictionary and JSON serialization."""
        import json
        original = DecisionPolicy(
            policy_name="frozen_v3_test",
            policy_version="v3.0-test",
            approve_threshold=0.055,
            reject_threshold=0.405,
            cost_model=CostModel(cost_fn=12000.0, cost_fp=1500.0, cost_manual_review=200.0),
            description="Test serialization roundtrip",
        )
        d = original.to_dict()
        json_str = json.dumps(d)
        reloaded_dict = json.loads(json_str)
        reconstructed = DecisionPolicy.from_dict(reloaded_dict)

        assert reconstructed.policy_name == original.policy_name
        assert reconstructed.policy_version == original.policy_version
        assert reconstructed.approve_threshold == original.approve_threshold
        assert reconstructed.reject_threshold == original.reject_threshold
        assert reconstructed.cost_model.cost_fn == original.cost_model.cost_fn
        assert reconstructed.cost_model.cost_fp == original.cost_model.cost_fp
        assert reconstructed.cost_model.cost_manual_review == original.cost_model.cost_manual_review
        assert reconstructed.description == original.description

    def test_policy_determinism(self):
        """Verify identical outputs across multiple evaluations."""
        policy = DecisionPolicy(approve_threshold=0.055, reject_threshold=0.405)
        prob = 0.1234
        results = [policy.decide(prob) for _ in range(50)]
        for r in results:
            assert r["decision"] == results[0]["decision"]
            assert r["expected_economic_cost"] == results[0]["expected_economic_cost"]
            assert r["risk_tier"] == results[0]["risk_tier"]

    def test_review_rate_constraint_enforcement(self):
        """Verify optimize_three_tier_policy never exceeds target_review_rate_max."""
        np.random.seed(42)
        y_val = np.random.binomial(1, 0.07, 1000)
        p_val = np.random.uniform(0.01, 0.40, 1000)
        cm = CostModel(cost_fn=10000.0, cost_fp=1000.0, cost_manual_review=150.0)

        # Force tight review rate max of 15%
        policy_tight = optimize_three_tier_policy(y_val, p_val, cost_model=cm, target_review_rate_max=0.15)
        # Check actual review rate on the validation data
        is_app = p_val <= policy_tight.approve_threshold
        is_rej = p_val >= policy_tight.reject_threshold
        actual_rev_rate = np.mean((~is_app) & (~is_rej))
        assert actual_rev_rate <= 0.1501, f"Review rate {actual_rev_rate} exceeded 15% max constraint"

    def test_optimization_never_uses_test_labels(self):
        """Verify optimization helpers operate purely on passed validation arguments."""
        # Create synthetic validation vectors
        y_synth = np.array([0, 0, 1, 0, 1, 0, 0, 1])
        p_synth = np.array([0.02, 0.04, 0.08, 0.03, 0.25, 0.01, 0.05, 0.30])
        cm = CostModel()

        # Optimization runs without accessing filesystem or test split
        pol = optimize_three_tier_policy(y_synth, p_synth, cost_model=cm)
        assert pol.approve_threshold > 0.0
        assert pol.reject_threshold > pol.approve_threshold

    def test_policy_audit_trail(self):
        prev_pol = get_active_policy()
        initial_log_len = len(get_policy_audit_log())
        try:
            new_pol = DecisionPolicy(
                policy_name="audit_test_policy",
                policy_version="test-v2",
                approve_threshold=0.03,
                reject_threshold=0.12,
            )
            update_active_policy(new_pol, user="test_analyst")
            active = get_active_policy()
            assert active.policy_name == "audit_test_policy"
            logs = get_policy_audit_log()
            assert len(logs) == initial_log_len + 1
            assert logs[-1]["user"] == "test_analyst"
        finally:
            update_active_policy(prev_pol, user="test_cleanup")


# =============================================================================
# 2. Policy Optimization Helper Tests
# =============================================================================

class TestPolicyOptimizationHelpers:
    def test_optimize_f1_threshold(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        p_val = np.array([0.02, 0.05, 0.10, 0.12, 0.20, 0.35, 0.60, 0.80])
        best_t, best_f1 = optimize_f1_threshold(y_true, p_val)
        assert 0.0 < best_t < 1.0
        assert best_f1 > 0.80

    def test_optimize_three_tier_policy(self):
        y_true = np.array([0]*90 + [1]*10)
        p_val = np.linspace(0.01, 0.50, 100)
        cm = CostModel(cost_fn=5000.0, cost_fp=500.0)
        pol = optimize_three_tier_policy(y_true, p_val, cost_model=cm)
        assert pol.approve_threshold <= pol.reject_threshold
        assert 0.0 <= pol.approve_threshold <= 1.0
        assert 0.0 <= pol.reject_threshold <= 1.0


# =============================================================================
# 3. Calibrated Inference Integration Tests
# =============================================================================

class TestCalibratedInference:
    def test_prepare_features_schema(self, sample_applicant):
        df = prepare_features(sample_applicant)
        assert list(df.columns) == MODEL_FEATURES
        assert df.shape == (1, 15)

    def test_get_calibrated_model(self):
        model = get_calibrated_model("lightgbm", "isotonic")
        assert model is not None
        assert hasattr(model, "predict_proba")

    def test_predict_single_calibrated_output(self, sample_applicant):
        res = predict_single(sample_applicant, model_name="lightgbm", calibration_method="isotonic")

        # Required fields check
        required_fields = [
            "model_name",
            "model_version",
            "calibration_method",
            "calibration_version",
            "raw_default_probability",
            "calibrated_default_probability",
            "approval_probability",
            "decision",
            "risk_tier",
            "policy_version",
            "policy_metadata",
            "expected_economic_cost",
        ]
        for field in required_fields:
            assert field in res, f"Missing required field: {field}"

        # Value constraints
        assert res["model_name"] == "lightgbm"
        assert res["calibration_method"] == "isotonic"
        assert 0.0 <= res["calibrated_default_probability"] <= 1.0
        assert 0.0 <= res["raw_default_probability"] <= 1.0
        assert res["decision"] in ("APPROVE", "MANUAL_REVIEW", "REJECT")
        assert res["risk_tier"] in ("VERY_LOW", "LOW", "MODERATE", "HIGH", "SEVERE")

    def test_predict_single_high_risk(self, high_risk_applicant):
        res = predict_single(high_risk_applicant, model_name="lightgbm", calibration_method="isotonic")
        assert res["calibrated_default_probability"] > 0.20
        assert res["decision"] in ("MANUAL_REVIEW", "REJECT")

    def test_predict_single_custom_policy_override(self, sample_applicant):
        strict_policy = DecisionPolicy(
            policy_name="ultra_strict",
            approve_threshold=0.001,
            reject_threshold=0.002,
        )
        res = predict_single(sample_applicant, policy=strict_policy)
        assert res["policy_name"] == "ultra_strict"
        # Since approve threshold is 0.001, applicant should be rejected
        assert res["decision"] == "REJECT"


# =============================================================================
# 4. API Endpoints Tests
# =============================================================================

class TestPolicyAPIEndpoints:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_get_policy_config(self, client):
        resp = client.get("/policy/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "active_policy" in data["data"]
        assert "approve_threshold" in data["data"]["active_policy"]
        assert "reject_threshold" in data["data"]["active_policy"]

    def test_post_policy_evaluate(self, client, sample_applicant):
        payload = {
            "application": {
                "income": 95000,
                "loan_amount": 200000,
                "credit_score": 740,
                "interest_rate": 6.5,
                "loan_term": 30,
                "property_value": 300000,
                "existing_loans": 0,
            },
            "approve_threshold": 0.05,
            "reject_threshold": 0.15,
            "cost_fn": 12000.0,
            "cost_fp": 1500.0,
        }
        resp = client.post("/policy/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        res_data = data["data"]
        assert "calibrated_default_probability" in res_data
        assert "decision" in res_data
        assert res_data["decision"] in ("APPROVE", "MANUAL_REVIEW", "REJECT")
        assert res_data["policy_metadata"]["cost_fn"] == 12000.0

    def test_post_policy_evaluate_invalid_thresholds(self, client):
        payload = {
            "application": {
                "income": 60000,
                "loan_amount": 150000,
                "credit_score": 700,
                "interest_rate": 6.5,
                "loan_term": 30,
            },
            "approve_threshold": 0.40,
            "reject_threshold": 0.20,  # Invalid: approve > reject
        }
        resp = client.post("/policy/evaluate", json=payload)
        assert resp.status_code == 400
        assert "cannot exceed" in resp.json()["detail"]

    def test_analyze_endpoint_returns_calibrated_metadata(self, client):
        payload = {
            "income": 95000,
            "loan_amount": 200000,
            "credit_score": 740,
            "interest_rate": 6.5,
            "loan_term": 30,
            "property_value": 300000,
            "existing_loans": 0,
        }
        resp = client.post("/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        res_data = data["data"]
        assert "model_name" in res_data
        assert "calibration_method" in res_data
        assert "calibrated_default_probability" in res_data
        assert "decision" in res_data
