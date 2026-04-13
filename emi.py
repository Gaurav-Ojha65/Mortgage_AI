"""EMI (Equated Monthly Installment) calculator."""


def calculate_emi(P: float, annual_rate: float, years: int) -> float:
    """
    Calculate EMI for a loan.

    Args:
        P: Principal loan amount (must be positive)
        annual_rate: Annual interest rate as percentage (e.g., 8.5 for 8.5%)
        years: Loan tenure in years (must be positive integer)

    Returns:
        Monthly EMI amount

    Raises:
        ValueError: If inputs are invalid
    """
    if P <= 0:
        raise ValueError("Principal must be positive")
    if annual_rate < 0:
        raise ValueError("Annual rate cannot be negative")
    if years <= 0:
        raise ValueError("Years must be positive")

    # Convert annual rate to monthly rate
    monthly_rate = annual_rate / 12 / 100

    # Handle zero interest rate case
    if monthly_rate == 0:
        return P / years / 12

    n = years * 12  # total number of months

    # EMI formula: P * r * (1+r)^n / ((1+r)^n - 1)
    emi = P * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)

    return round(emi, 2)


if __name__ == "__main__":
    # Example usage
    principal = 100000
    annual_rate = 8.5
    years = 5

    emi = calculate_emi(principal, annual_rate, years)
    print(f"Loan Amount: Rs.{principal}")
    print(f"Annual Rate: {annual_rate}%")
    print(f"Tenure: {years} years")
    print(f"EMI: Rs.{emi}")

    # Zero interest rate example
    print(f"\nZero interest EMI: Rs.{calculate_emi(100000, 0, 5)}")
