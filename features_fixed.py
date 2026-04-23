"""
Feature engineering - Fixed version with validation and consistent schema.
"""

# Single source of truth for feature columns
INFERENCE_FEATURES = [
    "income",
    "loan_amount",
    "credit_score",
    "existing_loans",
    "loan_term",
    "debt_to_income_ratio",
    "emi_to_income_ratio",
    "credit_utilization_score"
]


def engineer_features(data: dict) -> dict:
    """
    Engineer features from raw loan application data.
    Raises ValueError for invalid inputs (no silent failures).
    """
    income = float(data.get("income", 0))
    loan_amount = float(data.get("loan_amount", 0))
    credit_score = int(data.get("credit_score", 650))
    existing_loans = int(data.get("existing_loans", 0))
    loan_term = int(data.get("loan_term", 5))

    # Calculate EMI from interest rate if provided, else use default 8.5%
    interest_rate = float(data.get("interest_rate", 8.5))
    emi = data.get("emi")

    # Validate inputs - raise clear errors (no None returns!)
    if income <= 0:
        raise ValueError(f"Income must be positive, got {income}")
    if loan_term <= 0 or loan_term > 30:
        raise ValueError(f"Loan term must be 1-30 years, got {loan_term}")
    if credit_score < 300 or credit_score > 850:
        raise ValueError(f"Credit score must be 300-850, got {credit_score}")

    # Calculate EMI if not provided
    if emi is None:
        monthly_rate = interest_rate / 12 / 100
        n_months = loan_term * 12
        if monthly_rate == 0:
            emi = loan_amount / n_months
        else:
            emi = loan_amount * monthly_rate * (1 + monthly_rate) ** n_months / \
                  ((1 + monthly_rate) ** n_months - 1)

    emi = float(emi)

    # Feature engineering
    debt_to_income_ratio = loan_amount / (income * loan_term)
    emi_to_income_ratio = (emi / income) * 100
    credit_utilization_score = (credit_score - 300) / (850 - 300)

    # Return all features in consistent order
    return {
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "loan_term": loan_term,
        "debt_to_income_ratio": round(debt_to_income_ratio, 4),
        "emi_to_income_ratio": round(emi_to_income_ratio, 2),
        "credit_utilization_score": round(credit_utilization_score, 4),
    }


def get_feature_names() -> list:
    """Returns feature columns in exact order for model inference."""
    return INFERENCE_FEATURES.copy()


if __name__ == "__main__":
    # Test
    test_data = {
        "income": 50000,
        "loan_amount": 200000,
        "credit_score": 650,
        "existing_loans": 2,
        "loan_term": 60,
    }

    result = engineer_features(test_data)
    print("Engineered features:", result)
    print("Feature order:", get_feature_names())
