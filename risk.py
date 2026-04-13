def calculate_risk(income, loan_amount, credit_score, existing_loans):
    risk_score = 0

    # Income vs Loan
    ratio = loan_amount / income

    if ratio > 5:
        risk_score += 3
    elif ratio > 3:
        risk_score += 2
    else:
        risk_score += 1

    # Credit Score
    if credit_score < 600:
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

    # Final Risk Level
    if risk_score >= 7:
        return "HIGH"
    elif risk_score >= 5:
        return "MEDIUM"
    else:
        return "LOW"


# Test
if __name__ == "__main__":
    risk = calculate_risk(50000, 200000, 650, 2)
    print("Risk Level:", risk)