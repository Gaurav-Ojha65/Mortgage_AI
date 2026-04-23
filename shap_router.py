"""
SHAP explanation endpoints.
Add to your FastAPI app:

    from shap_router import shap_router
    app.include_router(shap_router, prefix="/api")
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing   import Optional
import joblib, os

from shap_explainer import explain_decision

shap_router = APIRouter()

MODELS_DIR  = "models"
MODEL_NAMES = ["logisticregression", "xgboost", "lightgbm"]
_cache      = {}

def _load(name: str):
    if name not in _cache:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if not os.path.exists(path):
            raise HTTPException(404, f"Model '{name}' not found — run train_models.py")
        _cache[name] = joblib.load(path)
    return _cache[name]

def _active() -> str:
    p = os.path.join(MODELS_DIR, "best_model_name.txt")
    return open(p).read().strip().lower().replace(" ", "") if os.path.exists(p) else "xgboost"


# ── Request schema ────────────────────────────────────────────────────────────

class ApplicantInput(BaseModel):
    credit_score:          float = Field(..., ge=300,  le=850)
    annual_income:         float = Field(..., ge=0)
    loan_amount:           float = Field(..., ge=1000)
    loan_term:             int   = Field(36,  ge=12,   le=360)
    dti_ratio:             float = Field(..., ge=0,    le=1)
    employment_years:      float = Field(2.0, ge=0)
    num_credit_lines:      int   = Field(3,   ge=0)
    num_derogatory_marks:  int   = Field(0,   ge=0)
    credit_utilization:    float = Field(0.3, ge=0,    le=1)
    payment_history_score: float = Field(0.9, ge=0,    le=1)
    home_ownership:        int   = Field(1,   ge=0,    le=2)
    purpose_encoded:       int   = Field(0,   ge=0,    le=9)
    num_late_payments:     int   = Field(0,   ge=0)
    savings_balance:       float = Field(5000, ge=0)
    monthly_expenses:      float = Field(2000, ge=0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@shap_router.post("/explain")
def explain(applicant: ApplicantInput, model: Optional[str] = None):
    """
    Full SHAP explanation for one applicant.
    Returns waterfall data + plain-English reasons.
    Optional ?model=xgboost | lightgbm | logisticregression
    """
    name = (model or _active()).lower().replace(" ", "")
    if name not in MODEL_NAMES:
        raise HTTPException(400, f"Unknown model. Choose from: {MODEL_NAMES}")

    clf    = _load(name)
    result = explain_decision(applicant.model_dump(), clf, name)
    return result


@shap_router.post("/explain/compare")
def explain_compare(applicant: ApplicantInput):
    """
    Run SHAP on ALL models and return top factors from each.
    Useful to see if models disagree on WHY, not just whether to approve.
    """
    results = {}
    for name in MODEL_NAMES:
        clf = _load(name)
        r   = explain_decision(applicant.model_dump(), clf, name)
        results[name] = {
            "decision":            r["decision"],
            "approval_probability": r["approval_probability"],
            "top_factors":         r["top_factors"][:5],
            "plain_english":       r["plain_english"][:3],
        }
    return {"models": results}


@shap_router.post("/explain/what-if")
def what_if(
    applicant:     ApplicantInput,
    changes:       dict,
    model: Optional[str] = None,
):
    """
    Score original applicant AND a modified version.
    Useful for: 'what if credit score was 750 instead of 650?'

    Body:
        { "applicant": {...}, "changes": {"credit_score": 750} }

    Returns before + after SHAP with probability delta.
    """
    name = (model or _active()).lower().replace(" ", "")
    clf  = _load(name)

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
        "decision_changed":  r_before["decision"] != r_after["decision"],
    }