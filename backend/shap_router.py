"""
SHAP explanation endpoints.
Uses ml.predict for consistent feature engineering.
"""

from pathlib import Path
from typing import Optional
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from ml.inference.predict import get_model, get_active_model_name, prepare_features, MODEL_FEATURES
from shap_explainer import explain_decision

shap_router = APIRouter(tags=["explain"])

MODEL_NAMES = ["logisticregression", "xgboost", "lightgbm"]


# ─── Schema ────────────────────────────────────────────────────────────────────

class ApplicantInput(BaseModel):
    credit_score:          float = Field(..., ge=300,  le=850)
    annual_income:         float = Field(..., ge=0)
    loan_amount:           float = Field(..., ge=1000)
    loan_term:             int   = Field(36,  ge=12,   le=360)
    dti_ratio:             float = Field(..., ge=0,    le=1)
    employment_years:      float = Field(2.0, ge=0)
    num_credit_lines:      int   = Field(3,   ge=0)
    num_derogatory_marks:  int   = Field(0,   ge=0)
    credit_utilization:          float = Field(0.3, ge=0,    le=1)
    late_payment_severity_score: float = Field(0.95, ge=0,   le=1)
    home_ownership:              int   = Field(1,   ge=0,    le=2)
    purpose_encoded:       int   = Field(0,   ge=0,    le=9)
    num_late_payments:     int   = Field(0,   ge=0)
    savings_balance:       float = Field(5000, ge=0)
    monthly_expenses:      float = Field(2000, ge=0)


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@shap_router.post("/explain")
def explain(applicant: ApplicantInput, model: Optional[str] = None):
    """Full SHAP explanation for one applicant."""
    name = (model or get_active_model_name()).lower().replace(" ", "")
    if name not in MODEL_NAMES:
        raise HTTPException(400, f"Unknown model. Choose from: {MODEL_NAMES}")

    try:
        clf = get_model(name)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))

    result = explain_decision(applicant.model_dump(), clf, name)
    return result


@shap_router.post("/explain/compare")
def explain_compare(applicant: ApplicantInput):
    """Run SHAP on ALL models and return top factors from each."""
    results = {}
    for name in MODEL_NAMES:
        try:
            clf = get_model(name)
            r = explain_decision(applicant.model_dump(), clf, name)
            results[name] = {
                "decision":             r["decision"],
                "approval_probability": r["approval_probability"],
                "top_factors":          r["top_factors"][:5],
                "plain_english":        r["plain_english"][:3],
            }
        except FileNotFoundError:
            results[name] = {"error": "model not trained"}
        except Exception as e:
            results[name] = {"error": str(e)}
    return {"models": results}


@shap_router.post("/explain/what-if")
def what_if(
    applicant: ApplicantInput,
    changes: dict,
    model: Optional[str] = None,
):
    """Score original applicant AND a modified version."""
    name = (model or get_active_model_name()).lower().replace(" ", "")

    try:
        clf = get_model(name)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))

    original = applicant.model_dump()
    modified = {**original, **changes}

    r_before = explain_decision(original, clf, name)
    r_after  = explain_decision(modified, clf, name)

    return {
        "original":          r_before,
        "modified":          r_after,
        "changes_applied":   changes,
        "probability_delta": round(
            r_after["approval_probability"] - r_before["approval_probability"], 4
        ),
        "decision_changed": r_before["decision"] != r_after["decision"],
    }