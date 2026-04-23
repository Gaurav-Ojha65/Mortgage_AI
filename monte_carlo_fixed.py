"""
Monte Carlo Simulation - Fixed version with proper seeding.
"""

import numpy as np
from emi import calculate_emi


def simulate(loan_input: dict, n_simulations: int = 10000, seed: int = None) -> dict:
    """
    Run Monte Carlo simulation to estimate loan default probability.

    Fixed: seed is optional - production uses None for true randomness.
    """
    income = loan_input["income"]
    loan_amount = loan_input["loan_amount"]
    interest_rate = loan_input["interest_rate"]
    loan_term = loan_input["loan_term"]
    credit_score = loan_input.get("credit_score", 700)

    # Only seed for testing/reproducibility - production uses None
    if seed is not None:
        np.random.seed(seed)

    # Vectorized simulation setup
    income_variation = np.random.normal(1.0, 0.10, n_simulations)
    simulated_incomes = income * income_variation
    simulated_incomes = np.maximum(simulated_incomes, income * 0.3)

    rate_variation = np.random.uniform(-0.02, 0.02, n_simulations)
    simulated_rates = interest_rate * (1 + rate_variation)
    simulated_rates = np.maximum(simulated_rates, 1.0)

    job_loss_mask = np.random.random(n_simulations) < 0.08
    n_job_loss = np.sum(job_loss_mask)

    loan_term_months = int(loan_term * 12)
    n_years = loan_term

    total_expense_shocks = np.random.poisson(0.5, size=n_simulations) * n_years
    total_unexpected_expenses = total_expense_shocks * np.random.uniform(5000, 50000, n_simulations)

    # Calculate EMI for all simulations
    emi_values = np.array([
        calculate_emi(loan_amount, rate, loan_term)
        for rate in simulated_rates
    ])

    # Scenario Classification
    stressed_income = simulated_incomes.copy()
    stressed_income[job_loss_mask] *= 0.30

    monthly_expenses = total_unexpected_expenses / loan_term_months
    effective_income = simulated_incomes - monthly_expenses

    emi_ratio_normal = emi_values / simulated_incomes
    emi_ratio_stressed = emi_values / stressed_income

    default_normal = emi_ratio_normal > 0.50
    default_stressed = emi_ratio_stressed > 0.50
    default_expense = (emi_values / effective_income) > 0.50

    defaults = default_normal | default_stressed | default_expense

    stable = (~job_loss_mask) & (emi_ratio_normal < 0.40)
    stressed = ((~job_loss_mask) & (emi_ratio_normal >= 0.40) & (emi_ratio_normal <= 0.50)) | \
              (job_loss_mask & (emi_ratio_stressed <= 0.50))
    crisis = (job_loss_mask) & (emi_ratio_stressed > 0.50) | default_expense

    scenario_breakdown = {
        "stable": int(np.sum(stable)),
        "stressed": int(np.sum(stressed)),
        "crisis": int(np.sum(crisis))
    }

    default_prob = np.mean(defaults)

    if default_prob < 0.15:
        risk_label = "LOW"
    elif default_prob < 0.35:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    worst_case_emi = np.percentile(emi_values, 95)

    # Safe income threshold - use separate RNG for independence
    income_sweep = np.linspace(income * 0.6, income * 1.4, 100)
    default_probs = []

    for inc in income_sweep:
        # Use separate random number generator for threshold calc
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        test_incomes = inc * rng.normal(1.0, 0.10, min(n_simulations, 5000))
        test_incomes = np.maximum(test_incomes, inc * 0.3)
        test_emi_ratio = emi_values[:min(n_simulations, 5000)] / test_incomes
        test_defaults = test_emi_ratio > 0.50
        default_probs.append(np.mean(test_defaults))

    default_probs = np.array(default_probs)
    safe_indices = np.where(default_probs < 0.15)[0]
    safe_income_threshold = income_sweep[safe_indices[0]] if len(safe_indices) > 0 else income_sweep[-1]

    return {
        "default_probability": float(default_prob),
        "risk_label": risk_label,
        "worst_case_emi": float(worst_case_emi),
        "safe_income_threshold": float(safe_income_threshold),
        "scenario_breakdown": scenario_breakdown,
        "mean_emi_ratio": float(np.mean(emi_ratio_normal)),
        "median_emi_ratio": float(np.median(emi_ratio_normal)),
        "percentile_95_emi_ratio": float(np.percentile(emi_ratio_normal, 95)),
        "job_loss_count": int(n_job_loss),
        "n_simulations": n_simulations
    }


if __name__ == "__main__":
    loan_input = {
        "income": 50000,
        "loan_amount": 200000,
        "interest_rate": 8.5,
        "loan_term": 5,
        "credit_score": 650
    }

    # Test with seed for reproducibility
    results1 = simulate(loan_input, n_simulations=1000, seed=42)
    results2 = simulate(loan_input, n_simulations=1000, seed=42)

    print(f"With same seed: prob1={results1['default_probability']:.4f}, prob2={results2['default_probability']:.4f}")

    # Production run without seed (will vary)
    results_prod = simulate(loan_input, n_simulations=1000)
    print(f"Production run (no seed): prob={results_prod['default_probability']:.4f}")
    print(f"Risk: {results_prod['risk_label']}")
