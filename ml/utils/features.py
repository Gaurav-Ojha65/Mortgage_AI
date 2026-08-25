"""
Feature engineering for Mortgage AI inference pipeline.

NOTE: The canonical 15-feature schema lives in ml/predict.py (MODEL_FEATURES).
This module validates and prepares raw applicant data for model inference,
mapping the simple /analyze endpoint inputs into the full 15-feature space.
"""

import sys
from pathlib import Path

# Ensure project root is in path regardless of working directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ─── Credit score bounds (local constants to avoid circular import) ────────────
CREDIT_SCORE_MIN = 300
CREDIT_SCORE_MAX = 850
LOAN_TERM_YEARS_MIN = 1
LOAN_TERM_YEARS_MAX = 30

# ─── Inference features for the simple /analyze endpoint (8-feature model) ────
# These are the fields that come in through the /analyze API form.
INFERENCE_FEATURES = [
    "income",
    "loan_amount",
    "credit_score",
    "existing_loans",
    "loan_term",
    "debt_to_income_ratio",
    "emi_to_income_ratio",
    "credit_utilization_score",
]


def engineer_features(data: dict) -> dict:
    """
    Engineer features from raw loan application data for the /analyze endpoint.

    Maps the simple 6-field form input into derived features for the
    basic prediction model (best_model.pkl).

    Args:
        data: dict with keys:
            - income: Monthly income (must be positive)
            - loan_amount: Total loan principal (must be positive)
            - credit_score: Credit score 300-850
            - existing_loans: Number of existing loans
            - loan_term: Loan tenure in years
            - interest_rate: Annual interest rate % (default 8.5%)
            - emi: Monthly EMI (calculated if not provided)

    Returns:
        Enriched dict with original fields plus engineered features.

    Raises:
        ValueError: For invalid inputs.
    """
    income = float(data.get("income", 0))
    loan_amount = float(data.get("loan_amount", 0))
    credit_score = int(data.get("credit_score", 650))
    existing_loans = int(data.get("existing_loans", 0))
    loan_term = int(data.get("loan_term", 5))
    interest_rate = float(data.get("interest_rate", 8.5))
    emi = data.get("emi")

    # Validate
    if income <= 0:
        raise ValueError(f"Income must be positive, got {income}")
    if loan_amount <= 0:
        raise ValueError(f"Loan amount must be positive, got {loan_amount}")
    if loan_term < LOAN_TERM_YEARS_MIN or loan_term > LOAN_TERM_YEARS_MAX:
        raise ValueError(
            f"Loan term must be {LOAN_TERM_YEARS_MIN}-{LOAN_TERM_YEARS_MAX} years, got {loan_term}"
        )
    if credit_score < CREDIT_SCORE_MIN or credit_score > CREDIT_SCORE_MAX:
        raise ValueError(
            f"Credit score must be {CREDIT_SCORE_MIN}-{CREDIT_SCORE_MAX}, got {credit_score}"
        )

    # Calculate EMI if not provided
    if emi is None:
        monthly_rate = interest_rate / 12 / 100
        n_months = loan_term * 12
        if monthly_rate == 0:
            emi = loan_amount / n_months
        else:
            emi = (
                loan_amount
                * monthly_rate
                * (1 + monthly_rate) ** n_months
                / ((1 + monthly_rate) ** n_months - 1)
            )
    emi = float(emi)

    # 1. Debt-to-income ratio
    debt_to_income_ratio = loan_amount / (income * loan_term)

    # 2. EMI-to-income ratio (monthly EMI as % of monthly income)
    emi_to_income_ratio = (emi / income) * 100

    # 3. Credit utilization score (normalized 0-1)
    credit_utilization_score = (credit_score - CREDIT_SCORE_MIN) / (
        CREDIT_SCORE_MAX - CREDIT_SCORE_MIN
    )
    credit_utilization_score = max(0.0, min(1.0, credit_utilization_score))

    # 4. Loan burden index
    loan_burden_index = existing_loans * 0.15 + debt_to_income_ratio

    # 5. Affordability score (1 = no burden, 0 = all income to EMI)
    affordability_score = max(0.0, min(1.0, 1 - (emi / income)))

    return {
        **data,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "loan_term": loan_term,
        "emi": round(emi, 2),
        "debt_to_income_ratio": round(debt_to_income_ratio, 4),
        "emi_to_income_ratio": round(emi_to_income_ratio, 2),
        "credit_utilization_score": round(credit_utilization_score, 4),
        "loan_burden_index": round(loan_burden_index, 4),
        "affordability_score": round(affordability_score, 4),
    }


def get_feature_names() -> list:
    """Returns feature columns for basic model inference."""
    return INFERENCE_FEATURES.copy()


def applicant_to_15_features(data: dict) -> dict:
    """
    Map a simple /analyze applicant dict to the full 15-feature schema
    used by the multi-model system (ml/predict.py MODEL_FEATURES).

    This bridges the simple 6-field form and the 15-feature trained models.

    Args:
        data: Raw applicant dict from /analyze endpoint

    Returns:
        Dict matching ml.predict.MODEL_FEATURES schema
    """
    enriched = engineer_features(data)

    income = enriched["income"]
    loan_amount = enriched["loan_amount"]
    interest_rate = float(data.get("interest_rate", 8.5))
    loan_term_years = enriched["loan_term"]
    loan_term_months = loan_term_years * 12

    return {
        "credit_score":          enriched["credit_score"],
        "annual_income":         income * 12,                    # monthly → annual
        "loan_amount":           loan_amount,
        "loan_term":             loan_term_months,               # years → months
        "dti_ratio":             float(data.get("dti_ratio")) if data.get("dti_ratio") is not None else min(enriched["debt_to_income_ratio"], 1.0),
        "employment_years":      float(data.get("employment_years") if data.get("employment_years") is not None else 3.0),
        "num_credit_lines":      int(data.get("num_credit_lines") if data.get("num_credit_lines") is not None else 3),
        "num_derogatory_marks":  int(data.get("num_derogatory_marks") if data.get("num_derogatory_marks") is not None else enriched["existing_loans"]),
        "credit_utilization":    float(data.get("credit_utilization")) if data.get("credit_utilization") is not None else max(0.05, min(0.95, 1.0 - enriched["credit_utilization_score"])),
        "late_payment_severity_score": float(data.get("late_payment_severity_score")) if data.get("late_payment_severity_score") is not None else min(1.0, max(0.0, 1 - enriched["emi_to_income_ratio"] / 100)),
        "home_ownership":        int(data.get("home_ownership") if data.get("home_ownership") is not None else 1),
        "purpose_encoded":       int(data.get("purpose_encoded") if data.get("purpose_encoded") is not None else 0),
        "num_late_payments":     int(data.get("num_late_payments") if data.get("num_late_payments") is not None else 0),
        "savings_balance":       float(data.get("savings_balance") if data.get("savings_balance") is not None else income * 3),
        "monthly_expenses":      float(data.get("monthly_expenses") if data.get("monthly_expenses") is not None else income * 0.4),
    }


if __name__ == "__main__":
    test_data = {
        "income": 5000,
        "loan_amount": 200000,
        "credit_score": 720,
        "existing_loans": 1,
        "loan_term": 20,
        "interest_rate": 8.5,
    }

    enriched = engineer_features(test_data)
    print("Basic enriched features:", {k: v for k, v in enriched.items()
                                        if k not in test_data})

    full = applicant_to_15_features(test_data)
    print("\n15-feature mapping:")
    for k, v in full.items():
        print(f"  {k}: {v}")
