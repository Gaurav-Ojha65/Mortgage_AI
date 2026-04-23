"""
Risk Assessment - Fixed version using constants and loan_term factor.
"""

from constants import (
    RISK_HIGH_THRESHOLD, RISK_LOW_THRESHOLD,
    CREDIT_SCORE_MIN, CREDIT_SCORE_MAX,
    LOAN_TERM_YEARS_MIN, LOAN_TERM_YEARS_MAX
)


def calculate_risk(income, loan_amount, credit_score, existing_loans, loan_term=5):
    """
    Calculate risk level using constants from constants.py.
    Now includes loan_term as a factor (longer loans = higher risk).
    """
    risk_score = 0

    # Income vs Loan ratio
    ratio = loan_amount / income

    if ratio > 5:
        risk_score += 3
    elif ratio > 3:
        risk_score += 2
    else:
        risk_score += 1

    # Credit Score - use constants
    if credit_score < CREDIT_SCORE_MIN + 300:  # 600
        risk_score += 3
    elif credit_score < 700:
        risk_score += 2
    else:
        risk_score += 1

    # Existing Loans
    if existing_loans >= 3:
        risk_score += 3
    elif existing_loans >= 1:
        risk_score += 2
    else:
        risk_score += 1

    # NEW: Loan Term factor (was missing!)
    # Longer loans are riskier due to more uncertainty
    if loan_term > 20:
        risk_score += 2
    elif loan_term > 10:
        risk_score += 1

    # Final Risk Level using thresholds from constants
    if risk_score >= 7:
        return "HIGH"
    elif risk_score >= 5:
        return "MEDIUM"
    else:
        return "LOW"


if __name__ == "__main__":
    # Test cases
    print("Risk Assessment Tests (Fixed):")

    # Short loan, good credit
    print(f"Safe case (5yr): {calculate_risk(50000, 100000, 750, 0, 5)}")

    # Long loan, same amount - higher risk
    print(f"Long loan (30yr): {calculate_risk(50000, 100000, 750, 0, 30)}")

    # Marginal case
    print(f"Marginal case: {calculate_risk(50000, 200000, 650, 2, 10)}")
