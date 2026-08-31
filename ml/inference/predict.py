"""
Unified Prediction Module for Mortgage AI
==========================================
Single entry point for ALL model inference — used by api.py, model_router.py,
and shap_router.py. Ensures consistent feature ordering and model loading.

Usage:
    from ml.inference.predict import predict_single, get_model, MODEL_FEATURES

    result = predict_single(applicant_dict)
    result = predict_single(applicant_dict, model_name="lightgbm")
"""

import os
import logging
from typing import Dict, Optional, List, Any
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

# ─── Single source of truth: feature schema ──────────────────────────────────
# These 15 features are produced by data/pipeline.py and consumed by all models.
# Order matters — models were trained with this exact column order.

MODEL_FEATURES: List[str] = [
    "credit_score",
    "annual_income",
    "loan_amount",
    "loan_term",
    "dti_ratio",
    "employment_years",
    "num_credit_lines",
    "num_derogatory_marks",
    "credit_utilization",
    "late_payment_severity_score",
    "home_ownership",
    "purpose_encoded",
    "num_late_payments",
    "savings_balance",
    "monthly_expenses",
]

FEATURE_LABELS: Dict[str, str] = {
    "credit_score":                 "Credit Score",
    "annual_income":                "Annual Income",
    "loan_amount":                  "Loan Amount",
    "loan_term":                    "Loan Term (months)",
    "dti_ratio":                    "Debt-to-Income Ratio",
    "employment_years":             "Employment History (years)",
    "num_credit_lines":             "Open Credit Lines",
    "num_derogatory_marks":         "Severe Delinquency (90+ days, count)",
    "credit_utilization":           "Credit Utilization",
    "late_payment_severity_score":  "Late Payment Severity Score (0=worst, 1=perfect)",
    "home_ownership":               "Home Ownership",
    "purpose_encoded":              "Loan Purpose",
    "num_late_payments":            "Mild Late Payments (30-59 days, count)",
    "savings_balance":              "Savings Balance",
    "monthly_expenses":             "Monthly Expenses",
}

# Available model names (lowercase, no spaces)
AVAILABLE_MODELS = ["logisticregression", "xgboost", "lightgbm", "ensemble"]

# ─── Project root detection ──────────────────────────────────────────────────
# Works whether called from project root or from ml/ subdirectory

def _project_root() -> Path:
    """Resolve the project root directory."""
    # ml/predict.py is at <root>/ml/predict.py
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "models").is_dir():
        return candidate
    # Fallback: current working directory
    return Path.cwd()


MODELS_DIR = _project_root() / "models"


# ─── Model cache ─────────────────────────────────────────────────────────────

_model_cache: Dict[str, object] = {}
_calibrator_cache: Dict[str, object] = {}


def get_model(name: str = "auto") -> object:
    """
    Load a raw trained base model by name. Uses cache after first load.

    Args:
        name: Model name ('xgboost', 'lightgbm', 'logisticregression', or 'auto')
              'auto' loads the active/best model.

    Returns:
        Trained scikit-learn compatible model
    """
    if name == "auto":
        name = get_active_model_name()

    name = name.lower().replace(" ", "")
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {AVAILABLE_MODELS}")

    if name not in _model_cache:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Run 'python -m ml.training.train' to train models first."
            )
        _model_cache[name] = joblib.load(path)
        logger.info(f"Loaded base model: {name} from {path}")

    return _model_cache[name]


def get_calibrated_model(name: str = "lightgbm", method: str = "isotonic") -> object:
    """
    Load a calibrated classifier model (fitted on validation data).

    Args:
        name: Base model name (default 'lightgbm')
        method: Calibration method ('isotonic' or 'sigmoid')

    Returns:
        CalibratedClassifierCV estimator or CalibratedPredictor pipeline
    """
    name = name.lower().replace(" ", "")
    method = method.lower().replace(" ", "")
    cache_key = f"{name}_{method}"

    if cache_key not in _calibrator_cache:
        pipeline_path = MODELS_DIR / f"{name}_calibrated_pipeline.joblib"
        if pipeline_path.exists() and method == "isotonic":
            _calibrator_cache[cache_key] = joblib.load(pipeline_path)
            logger.info(f"Loaded calibrated pipeline: {cache_key} from {pipeline_path}")
            return _calibrator_cache[cache_key]

        filename = f"{name}_calibrated_{method}.joblib"
        path = MODELS_DIR / filename
        if not path.exists():
            # Fallback: if calibrated artifact doesn't exist yet, load raw model
            logger.warning(
                f"Calibrated artifact '{filename}' not found at {path}. "
                f"Falling back to raw model."
            )
            return get_model(name)
        _calibrator_cache[cache_key] = joblib.load(path)
        logger.info(f"Loaded calibrated model: {cache_key} from {path}")

    return _calibrator_cache[cache_key]


def get_active_model_name() -> str:
    """Read the active model name from models/best_model_name.txt."""
    txt_path = MODELS_DIR / "best_model_name.txt"
    if txt_path.exists():
        return txt_path.read_text().strip().lower().replace(" ", "")
    return "lightgbm"  # sensible default post-fixes


def clear_cache():
    """Clear both raw model and calibrator caches (e.g. after retraining)."""
    _model_cache.clear()
    _calibrator_cache.clear()
    logger.info("Model and calibrator caches cleared")


# ─── Feature preparation ────────────────────────────────────────────────────

def prepare_features(applicant: Dict) -> pd.DataFrame:
    """
    Convert raw applicant dict into a model-ready DataFrame.

    Fills missing features with sensible defaults. Guarantees exact column
    ordering expected by trained models.

    Args:
        applicant: Dictionary with applicant data. Keys should match MODEL_FEATURES.

    Returns:
        DataFrame with shape (1, 15) in exact MODEL_FEATURES order
    """
    defaults = {
        "credit_score": 650,
        "annual_income": 50000,
        "loan_amount": 100000,
        "loan_term": 36,
        "dti_ratio": 0.3,
        "employment_years": 2.0,
        "num_credit_lines": 3,
        "num_derogatory_marks": 0,
        "credit_utilization": 0.3,
        "late_payment_severity_score": 0.95,
        "home_ownership": 1,
        "purpose_encoded": 0,
        "num_late_payments": 0,
        "savings_balance": 5000,
        "monthly_expenses": 2000,
    }

    row = {}
    for feat in MODEL_FEATURES:
        val = applicant.get(feat, defaults.get(feat, 0))
        row[feat] = float(val)

    return pd.DataFrame([row], columns=MODEL_FEATURES)


# ─── Calibrated Prediction Pipeline ──────────────────────────────────────────

def predict_single(
    applicant: Dict,
    model_name: str = "auto",
    calibration_method: str = "isotonic",
    threshold: Optional[float] = None,
    policy: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Score a single applicant through the complete calibrated prediction pipeline:

    Features -> Model (LightGBM) -> Isotonic Calibrator -> Calibrated Probability -> Decision Policy

    Args:
        applicant: Dictionary with applicant features
        model_name: Model to use ('lightgbm', 'xgboost', 'auto', etc.)
        calibration_method: 'isotonic' (recommended), 'sigmoid', or 'none'
        threshold: Optional binary threshold override (if policy is not provided)
        policy: Optional custom DecisionPolicy instance

    Returns:
        Dict with comprehensive calibrated outputs and decision policy details.
    """
    from risk.decision_policy import get_active_policy, DecisionPolicy

    actual_name = model_name if model_name != "auto" else get_active_model_name()
    X = prepare_features(applicant)

    # 1. Raw base model prediction
    raw_model = get_model(actual_name)
    raw_default_prob = float(raw_model.predict_proba(X)[0][1])

    # 2. Probability calibration step
    if calibration_method.lower() in ("isotonic", "sigmoid", "platt"):
        try:
            calibrated_model = get_calibrated_model(actual_name, method=calibration_method)
            calibrated_default_prob = float(calibrated_model.predict_proba(X)[0][1])
            cal_version = "oof-iso-v3.1" if calibration_method == "isotonic" else "oof-sig-v3.1"
        except Exception as exc:
            logger.warning(f"Calibration failed: {exc}. Using raw probability.")
            calibrated_default_prob = raw_default_prob
            cal_version = "fallback-none"
    else:
        calibrated_default_prob = raw_default_prob
        cal_version = "none"
        calibration_method = "none"

    # Clamp probability to strict [0.0, 1.0] range
    calibrated_default_prob = max(0.0, min(1.0, calibrated_default_prob))
    raw_default_prob = max(0.0, min(1.0, raw_default_prob))

    # 3. Decision Policy execution
    active_policy = policy or get_active_policy()
    if threshold is not None:
        # Binary threshold override requested
        decision_val = "APPROVE" if calibrated_default_prob <= threshold else "REJECT"
        risk_val = "LOW" if calibrated_default_prob <= 0.05 else ("HIGH" if calibrated_default_prob >= 0.20 else "MODERATE")
        policy_res = {
            "decision": decision_val,
            "risk_tier": risk_val,
            "calibrated_default_probability": round(calibrated_default_prob, 4),
            "expected_economic_cost": round(calibrated_default_prob * 10000.0, 2),
            "policy_version": "manual-override-threshold",
            "policy_name": "binary_override",
            "policy_metadata": {"binary_threshold": threshold},
        }
    else:
        policy_res = active_policy.decide(calibrated_default_prob)

    approval_prob = round(1.0 - calibrated_default_prob, 4)

    return {
        # Model identification
        "model_name": actual_name,
        "model_version": "v3.1",
        "calibration_method": calibration_method,
        "calibration_version": cal_version,

        # Probabilities
        "raw_default_probability": round(raw_default_prob, 4),
        "calibrated_default_probability": round(calibrated_default_prob, 4),
        "approval_probability": approval_prob,

        # Policy & Decision
        "decision": policy_res["decision"],
        "risk_tier": policy_res["risk_tier"],
        "expected_economic_cost": policy_res["expected_economic_cost"],
        "policy_version": policy_res["policy_version"],
        "policy_name": policy_res["policy_name"],
        "policy_metadata": policy_res["policy_metadata"],

        # Legacy compatibility aliases
        "risk_level": policy_res["risk_tier"].lower(),
        "model_used": actual_name,
    }


def predict_all_models(
    applicant: Dict,
    threshold: float = 0.5,
) -> Dict:
    """
    Score one applicant through ALL available models.

    Returns:
        Dict with per-model results and consensus info
    """
    results = {}
    for name in AVAILABLE_MODELS:
        try:
            results[name] = predict_single(applicant, model_name=name, threshold=threshold)
        except FileNotFoundError:
            logger.warning(f"Model '{name}' not available, skipping")
            continue

    if not results:
        raise FileNotFoundError("No trained models found. Run 'python -m ml.train' first.")

    # Consensus
    decisions = [r["decision"] for r in results.values()]
    probs = [r["approval_probability"] for r in results.values()]
    avg_prob = round(float(np.mean(probs)), 4)

    return {
        "models": results,
        "consensus": {
            "all_agree": len(set(decisions)) == 1,
            "avg_probability": avg_prob,
            "final_decision": "approved" if avg_prob >= threshold else "rejected",
            "models_agreeing_approve": sum(1 for d in decisions if d == "approved"),
            "models_agreeing_reject": sum(1 for d in decisions if d == "rejected"),
        },
    }


# ─── EMI calculation ────────────────────────────────────────────────────────

def calculate_emi(principal: float, annual_rate: float, term_months: int) -> float:
    """
    Calculate Equated Monthly Installment.

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate as percentage (e.g., 8.5)
        term_months: Loan term in months

    Returns:
        Monthly EMI amount
    """
    if principal <= 0 or term_months <= 0:
        return 0.0

    monthly_rate = annual_rate / 12 / 100
    if monthly_rate == 0:
        return round(principal / term_months, 2)

    emi = principal * monthly_rate * (1 + monthly_rate) ** term_months / (
        (1 + monthly_rate) ** term_months - 1
    )
    return round(emi, 2)


if __name__ == "__main__":
    # Quick self-test
    test_applicant = {
        "credit_score": 720,
        "annual_income": 85000,
        "loan_amount": 250000,
        "loan_term": 360,
        "dti_ratio": 0.28,
        "employment_years": 8,
        "num_credit_lines": 5,
        "num_derogatory_marks": 0,
        "credit_utilization": 0.25,
        "late_payment_severity_score": 0.95,
        "home_ownership": 2,
        "purpose_encoded": 1,
        "num_late_payments": 0,
        "savings_balance": 25000,
        "monthly_expenses": 3500,
    }

    print("=" * 60)
    print("  ML Predict — Self Test")
    print("=" * 60)
    print(f"  Models dir: {MODELS_DIR}")
    print(f"  Active model: {get_active_model_name()}")
    print(f"  Features: {len(MODEL_FEATURES)}")

    try:
        result = predict_single(test_applicant)
        print(f"\n  Single prediction:")
        for k, v in result.items():
            print(f"    {k}: {v}")

        all_results = predict_all_models(test_applicant)
        print(f"\n  All models:")
        for name, r in all_results["models"].items():
            print(f"    {name}: {r['decision']} ({r['approval_probability']:.2%})")
        print(f"  Consensus: {all_results['consensus']}")
    except FileNotFoundError as e:
        print(f"\n  [!] {e}")
        print("  Train models first: python -m ml.train")
