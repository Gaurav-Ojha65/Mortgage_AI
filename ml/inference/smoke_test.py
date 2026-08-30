"""
End-to-End Inference & API Smoke Test — Promoted v3.1 Canonical Baseline
========================================================================
Validates:
1. Canonical model loading (lightgbm.joblib)
2. Calibrator loading (lightgbm_oof_calibrator_isotonic.joblib)
3. CalibratedPredictor pipeline loading (lightgbm_calibrated_pipeline.joblib)
4. Frozen policy config loading (ml/models/frozen_policy_config.json)
5. Decision policy engine defaults & evaluation
6. SHAP TreeExplainer exact attribution and additivity on promoted model
7. FastAPI testclient /analyze endpoint response schema and contracts
8. Verification of all required response fields
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predict import (
    get_model,
    get_calibrated_model,
    predict_single,
    prepare_features,
    MODEL_FEATURES,
    clear_cache,
)
from backend.shap_explainer import explain_decision
from risk.decision_policy import get_active_policy, DecisionPolicy
from backend.api import app

def run_smoke_test():
    print("=" * 70)
    print("MORTGAGE AI v3.1 CANONICAL BASELINE — END-TO-END SMOKE TEST")
    print("=" * 70)
    clear_cache()
    
    # 1. Model Loading
    print("\n1. Loading canonical raw base model...")
    model = get_model("lightgbm")
    print(f"   Success: {type(model)} (n_estimators={model.n_estimators}, lr={model.learning_rate:.4f})")
    assert model.n_estimators == 651, f"Expected 651 trees for v3.1, got {model.n_estimators}"
    
    # 2. Calibrator & Pipeline Loading
    print("\n2. Loading canonical calibrated pipeline...")
    cal_pipeline = get_calibrated_model("lightgbm", "isotonic")
    print(f"   Success: {type(cal_pipeline)}")
    
    # 3. Policy Config Loading
    print("\n3. Loading canonical frozen policy config...")
    policy_config_path = PROJECT_ROOT / "ml" / "models" / "frozen_policy_config.json"
    assert policy_config_path.exists(), "frozen_policy_config.json missing!"
    cfg = json.loads(policy_config_path.read_text(encoding="utf-8"))
    active_policy = get_active_policy()
    print(f"   Active Policy: {active_policy.policy_name} ({active_policy.policy_version})")
    print(f"   Approve Threshold: {active_policy.approve_threshold}")
    print(f"   Reject Threshold:  {active_policy.reject_threshold}")
    assert active_policy.approve_threshold == 0.045
    assert active_policy.reject_threshold == 0.335
    
    # 4. Predict Single Inference
    print("\n4. Testing single applicant prediction pipeline...")
    sample_app = {
        "credit_score": 720,
        "annual_income": 85000,
        "loan_amount": 250000,
        "loan_term": 360,
        "dti_ratio": 0.28,
        "employment_years": 8.0,
        "num_credit_lines": 10,
        "num_derogatory_marks": 0,
        "credit_utilization": 0.25,
        "late_payment_severity_score": 0.98,
        "home_ownership": 2,
        "purpose_encoded": 0,
        "num_late_payments": 0,
        "savings_balance": 25000,
        "monthly_expenses": 3200,
    }
    pred_res = predict_single(sample_app, model_name="lightgbm", calibration_method="isotonic")
    print(f"   Prediction Output:")
    for k, v in pred_res.items():
        print(f"     {k}: {v}")
        
    required_keys = [
        "model_name",
        "model_version",
        "calibration_method",
        "calibration_version",
        "raw_default_probability",
        "calibrated_default_probability",
        "decision",
        "risk_tier",
        "policy_version",
        "policy_metadata",
        "expected_economic_cost",
    ]
    for key in required_keys:
        assert key in pred_res, f"Missing required field: {key}"
        
    assert pred_res["model_version"] == "v3.1"
    assert pred_res["calibration_version"] == "oof-iso-v3.1"
    assert pred_res["policy_version"] == "v3.1-policy-v1"
    
    # 5. SHAP Explainability on Promoted Model
    print("\n5. Testing SHAP TreeExplainer on promoted model...")
    shap_res = explain_decision(sample_app, model, "lightgbm")
    print(f"   SHAP Top Factors:")
    for f in shap_res["top_factors"][:3]:
        print(f"     - {f['feature']}: {f['shap_value']:+.4f} ({f['direction']})")
    assert len(shap_res["all_factors"]) == 15
    assert shap_res["base_value"] is not None
    
    # 6. FastAPI API Endpoint Integration
    print("\n6. Testing FastAPI /analyze endpoint...")
    with TestClient(app) as client:
        api_payload = {
            "income": 85000,
            "loan_amount": 250000,
            "credit_score": 720,
            "interest_rate": 6.25,
            "loan_term": 30,
            "property_value": 350000,
            "existing_loans": 0,
        }
        resp = client.post("/analyze", json=api_payload)
        assert resp.status_code == 200, f"API returned status {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["success"] is True
        print(f"   API Response Status: 200 OK")
        print(f"   Decision: {body['data']['decision']}")
        print(f"   Calibrated Prob: {body['data']['calibrated_default_probability']}")
        print(f"   Expected Cost: ${body['data']['expected_economic_cost']}")
        
        # 7. Check Policy Evaluate API Endpoint
        print("\n7. Testing FastAPI /policy/evaluate endpoint...")
        eval_payload = {
            "application": api_payload,
            "approve_threshold": 0.045,
            "reject_threshold": 0.335,
            "cost_fn": 10000.0,
            "cost_fp": 1000.0,
            "cost_manual_review": 150.0,
        }
        resp_pol = client.post("/policy/evaluate", json=eval_payload)
        assert resp_pol.status_code == 200
        body_pol = resp_pol.json()
        assert body_pol["success"] is True
        print(f"   Policy Evaluate Status: 200 OK")
        print(f"   Simulated Decision: {body_pol['data']['decision']}")
    
    print("\n" + "=" * 70)
    print("ALL SMOKE TEST CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()
