"""
Advanced feature engineering for mortgage loan applications.
Transforms raw input into model-ready features for ML-based mortgage decisions.
"""


def engineer_features(data: dict) -> dict:
    """
    Engineer features from raw loan application data.

    Args:
        data: Dictionary containing raw loan application fields:
            - income: Monthly income
            - loan_amount: Total loan principal
            - credit_score: Credit score (300-850)
            - existing_loans: Number of existing loan obligations
            - loan_term: Loan tenure in months
            - emi: Monthly EMI amount

    Returns:
        Enriched dictionary with all original fields plus engineered features
    """
    income = data.get("income", 0)
    loan_amount = data.get("loan_amount", 0)
    credit_score = data.get("credit_score", 0)
    existing_loans = data.get("existing_loans", 0)
    loan_term = data.get("loan_term", 0)
    emi = data.get("emi", 0)

    # Handle edge cases
    if income <= 0 or loan_term <= 0:
        return {
            **data,
            "debt_to_income_ratio": None,
            "emi_to_income_ratio": None,
            "credit_utilization_score": None,
            "loan_burden_index": None,
            "affordability_score": None,
        }

    # 1. debt_to_income_ratio
    # Measures how much loan is taken relative to total income over loan term
    # Higher ratio = more debt burden, higher default risk
    debt_to_income_ratio = loan_amount / (income * loan_term / 12)

    # 2. emi_to_income_ratio
    # Monthly payment as percentage of monthly income
    # Industry threshold: should not exceed 40% for mortgage approval
    emi_to_income_ratio = (emi / income) * 100

    # 3. credit_utilization_score
    # Normalized credit score to 0-1 range for ML model compatibility
    # 850 = perfect credit, 300 = worst, normalize to 0-1 scale
    # Higher score = better borrower, lower credit risk
    credit_utilization_score = (credit_score - 300) / (850 - 300)
    credit_utilization_score = max(0.0, min(1.0, credit_utilization_score))

    # 4. loan_burden_index
    # Composite index combining existing obligations and new debt burden
    # Weights existing loans (0.15 per loan) plus debt-to-income ratio
    # Higher index = greater financial burden = higher default risk
    loan_burden_index = existing_loans * 0.15 + debt_to_income_ratio

    # 5. affordability_score
    # Inverse of EMI burden - how much income is left after EMI
    # 1.0 = no EMI burden, 0.0 = entire income goes to EMI
    # Clipped to 0-1 range for model stability
    affordability_score = 1 - (emi / income)
    affordability_score = max(0.0, min(1.0, affordability_score))

    return {
        **data,
        "debt_to_income_ratio": round(debt_to_income_ratio, 4),
        "emi_to_income_ratio": round(emi_to_income_ratio, 2),
        "credit_utilization_score": round(credit_utilization_score, 4),
        "loan_burden_index": round(loan_burden_index, 4),
        "affordability_score": round(affordability_score, 4),
    }


def get_feature_names() -> list:
    """
    Returns all feature column names in order for model training/inference.

    Returns:
        List of feature names: original input fields followed by engineered features
    """
    return [
        "income",
        "loan_amount",
        "credit_score",
        "existing_loans",
        "loan_term",
        "emi",
        "debt_to_income_ratio",
        "emi_to_income_ratio",
        "credit_utilization_score",
        "loan_burden_index",
        "affordability_score",
    ]


if __name__ == "__main__":
    raw_application = {
        "income": 50000,
        "loan_amount": 200000,
        "credit_score": 650,
        "existing_loans": 2,
        "loan_term": 60,  # months
        "emi": 4103.31,
    }

    enriched = engineer_features(raw_application)

    print("=" * 50)
    print("FEATURE ENGINEERING SAMPLE RUN")
    print("=" * 50)
    print("\nRaw Input:")
    for k, v in raw_application.items():
        print(f"  {k}: {v}")

    print("\nEngineered Features:")
    for k, v in enriched.items():
        if k not in raw_application:
            print(f"  {k}: {v}")

    print("\nAll Features (get_feature_names order):")
    print(f"  {get_feature_names()}")

    print("\nFull Enriched Record:")
    print(f"  {enriched}")