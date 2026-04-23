"""
Model Router — Multi-Model Comparison Endpoints
Supports: single prediction, model comparison, feature importance, model switching.
"""

import os, json
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy  as np
import joblib

router = APIRouter()

# ─── Load all models at startup ──────────────────────────────────────────────

MODELS_DIR   = "models"
MODEL_NAMES  = ["logisticregression", "xgboost", "lightgbm"]
_model_cache = {}

def _load(name: str):
    if name not in _model_cache:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if not os.path.exists(path):
            raise HTTPException(404, f"Model '{name}' not found. Run train_models.py first.")
        _model_cache[name] = joblib.load(path)
    return _model_cache[name]

def _load_report():
    path = os.path.join(MODELS_DIR, "comparison_report.json")
    if not os.path.exists(path):
        raise HTTPException(404, "comparison_report.json not found. Run train_models.py first.")
    with open(path) as f:
        return json.load(f)

def _active_model_name() -> str:
    path = os.path.join(MODELS_DIR, "best_model_name.txt")
    if os.path.exists(path):
        return open(path).read().strip().lower().replace(" ", "")
    return "xgboost"

# ─── Request / response schemas ──────────────────────────────────────────────

class ApplicantInput(BaseModel):
    credit_score:          float = Field(..., ge=300, le=850,   description="FICO score")
    annual_income:         float = Field(..., ge=0,             description="Annual income USD")
    loan_amount:           float = Field(..., ge=1000,          description="Requested loan amount")
    loan_term:             int   = Field(36,  ge=12, le=360,    description="Loan term in months")
    dti_ratio:             float = Field(..., ge=0,   le=1,     description="Debt-to-income ratio 0-1")
    employment_years:      float = Field(2.0, ge=0,             description="Years at current employer")
    num_credit_lines:      int   = Field(3,   ge=0,             description="Number of open credit lines")
    num_derogatory_marks:  int   = Field(0,   ge=0,             description="Derogatory marks on record")
    credit_utilization:    float = Field(0.3, ge=0,   le=1,     description="Credit utilization ratio 0-1")
    payment_history_score: float = Field(0.9, ge=0,   le=1,     description="On-time payment rate 0-1")
    home_ownership:        int   = Field(1,   ge=0,   le=2,     description="0=rent 1=own 2=mortgage")
    purpose_encoded:       int   = Field(0,   ge=0,   le=9,     description="Loan purpose category")
    num_late_payments:     int   = Field(0,   ge=0,             description="Number of late payments (12mo)")
    savings_balance:       float = Field(5000, ge=0,            description="Savings account balance")
    monthly_expenses:      float = Field(2000, ge=0,            description="Monthly expenses USD")

    def to_feature_array(self):
        return np.array([[
            self.credit_score, self.annual_income, self.loan_amount,
            self.loan_term, self.dti_ratio, self.employment_years,
            self.num_credit_lines, self.num_derogatory_marks,
            self.credit_utilization, self.payment_history_score,
            self.home_ownership, self.purpose_encoded,
            self.num_late_payments, self.savings_balance,
            self.monthly_expenses,
        ]])

# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/models")
def list_models():
    """List available models and which is currently active."""
    report = _load_report()
    active = _active_model_name()
    return {
        "active_model": active,
        "available":    MODEL_NAMES,
        "winner":       report["winner"],
        "metrics":      report["metrics"],
    }


@router.get("/models/comparison")
def get_comparison():
    """Full comparison report: metrics, confusion matrices, feature importances."""
    return _load_report()


@router.post("/analyze")
def analyze(applicant: ApplicantInput, model: Optional[str] = None):
    """
    Score a single applicant with the active (or specified) model.
    ?model=xgboost | lightgbm | logisticregression
    """
    model_name = (model or _active_model_name()).lower().replace(" ", "")
    clf        = _load(model_name)
    X          = applicant.to_feature_array()

    prob       = float(clf.predict_proba(X)[0][1])
    decision   = "approved" if prob >= 0.5 else "rejected"
    risk_level = (
        "low"      if prob >= 0.75 else
        "medium"   if prob >= 0.50 else
        "high"     if prob >= 0.25 else
        "critical"
    )

    # Monthly EMI (simple amortisation)
    r   = 0.08 / 12  # assume 8% APR, replace with dynamic rate
    n   = applicant.loan_term
    emi = applicant.loan_amount * r / (1 - (1 + r) ** -n) if prob >= 0.5 else 0

    return {
        "decision":            decision,
        "approval_probability": round(prob, 4),
        "default_probability":  round(1 - prob, 4),
        "risk_level":           risk_level,
        "emi":                  round(emi, 2),
        "model_used":           model_name,
    }


@router.post("/analyze/compare")
def compare_all_models(applicant: ApplicantInput):
    """
    Run same applicant through ALL models and return side-by-side results.
    Used by the Compare UI component.
    """
    X       = applicant.to_feature_array()
    results = {}

    for name in MODEL_NAMES:
        clf  = _load(name)
        prob = float(clf.predict_proba(X)[0][1])
        results[name] = {
            "approval_probability": round(prob, 4),
            "default_probability":  round(1 - prob, 4),
            "decision":  "approved" if prob >= 0.5 else "rejected",
            "risk_level": (
                "low"    if prob >= 0.75 else
                "medium" if prob >= 0.50 else
                "high"   if prob >= 0.25 else
                "critical"
            ),
        }

    # Agreement check
    decisions   = [r["decision"] for r in results.values()]
    all_agree   = len(set(decisions)) == 1
    avg_prob    = round(np.mean([r["approval_probability"] for r in results.values()]), 4)
    final_call  = "approved" if avg_prob >= 0.5 else "rejected"

    return {
        "models":       results,
        "consensus": {
            "all_agree":          all_agree,
            "avg_probability":    avg_prob,
            "final_decision":     final_call,
            "disagreement_count": decisions.count("approved"),
        },
    }


@router.post("/models/switch/{model_name}")
def switch_active_model(model_name: str):
    """Switch which model handles /analyze requests."""
    name = model_name.lower().replace(" ", "")
    if name not in MODEL_NAMES:
        raise HTTPException(400, f"Unknown model '{model_name}'. Choose from: {MODEL_NAMES}")

    # Verify model file exists
    _load(name)

    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(name)

    return {"switched_to": name, "message": f"{name} is now the active model for /analyze"}


@router.get("/models/feature-importance/{model_name}")
def feature_importance(model_name: str):
    """Return feature importance for a specific model."""
    report = _load_report()
    name   = model_name  # keep original capitalisation for report lookup

    # Try case-insensitive match
    match  = next((k for k in report["importance"] if k.lower() == name.lower()), None)
    if not match:
        raise HTTPException(404, f"No importance data for '{name}'")

    importance = report["importance"][match]
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return {
        "model":    match,
        "features": [{"name": k, "importance": v} for k, v in sorted_imp],
    }