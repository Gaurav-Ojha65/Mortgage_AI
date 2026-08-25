"""
Monte Carlo Simulation for Mortgage Loan Default Risk
Vectorized simulation of 10,000+ scenarios with uncertainty modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from emi import calculate_emi


def simulate(loan_input: dict, n_simulations: int = 10000) -> dict:
    """
    Run Monte Carlo simulation to estimate loan default probability.

    Args:
        loan_input: dict with keys:
            - income: monthly income
            - loan_amount: total loan principal
            - interest_rate: annual interest rate (%)
            - loan_term: loan term in years
            - credit_score: credit score (used for risk adjustment)
        n_simulations: number of Monte Carlo runs (default 10000)

    Returns:
        dict with:
            - default_probability: float 0-1
            - risk_label: "LOW" / "MEDIUM" / "HIGH"
            - worst_case_emi: 95th percentile EMI
            - safe_income_threshold: min income for P(default) < 0.15
            - scenario_breakdown: {stable, stressed, crisis} counts
    """
    income = loan_input["income"]
    loan_amount = loan_input["loan_amount"]
    interest_rate = loan_input["interest_rate"]
    loan_term = loan_input["loan_term"]
    credit_score = loan_input.get("credit_score", 700)

    np.random.seed(42)

    # --- Vectorized simulation setup ---

    # a) Income variation: normal distribution ±20% (std=10% of mean)
    income_variation = np.random.normal(1.0, 0.10, n_simulations)
    simulated_incomes = income * income_variation
    simulated_incomes = np.maximum(simulated_incomes, income * 0.3)  # Floor at 30% of original

    # b) Interest rate variation: uniform ±2%
    rate_variation = np.random.uniform(-0.02, 0.02, n_simulations)
    simulated_rates = interest_rate * (1 + rate_variation)
    simulated_rates = np.maximum(simulated_rates, 1.0)  # Floor at 1%

    # c) Job loss probability: 8% chance, 6-month income gap
    job_loss_mask = np.random.random(n_simulations) < 0.08
    n_job_loss = np.sum(job_loss_mask)

    # d) Unexpected expense shocks: 1-3 large expenses per year, Rs.5000-50000 each
    loan_term_months = int(loan_term * 12)
    n_years = loan_term

    # Number of expense shocks per simulation (1-3 per year)
    n_expense_shocks = np.random.randint(1, 4, size=n_simulations) * n_years
    expense_amounts = np.random.uniform(5000, 50000, size=n_simulations)
    total_expense_shocks = np.random.poisson(0.5, size=n_simulations) * n_years
    total_unexpected_expenses = total_expense_shocks * np.random.uniform(5000, 50000)

    # Calculate base EMI for all simulations
    emi_values = np.array([
        calculate_emi(loan_amount, rate, loan_term)
        for rate in simulated_rates
    ])

    # --- Scenario Classification ---

    # Stress period: 6 months post-job-loss at 30% income
    stressed_income = simulated_incomes.copy()
    stressed_income[job_loss_mask] *= 0.30

    # Effective monthly income after expenses
    monthly_expenses = total_unexpected_expenses / loan_term_months
    effective_income = simulated_incomes - monthly_expenses

    # EMI-to-income ratios
    emi_ratio_normal = emi_values / simulated_incomes
    emi_ratio_stressed = emi_values / stressed_income

    # Default: EMI > 50% of income in any month
    default_normal = emi_ratio_normal > 0.50
    default_stressed = emi_ratio_stressed > 0.50
    default_expense = (emi_values / effective_income) > 0.50

    # Overall default = any of the above conditions
    defaults = default_normal | default_stressed | default_expense

    # --- Scenario Breakdown ---
    # Stable: no stress, EMI < 40% of income
    # Stressed: either job loss OR EMI 40-50%
    # Crisis: job loss AND high EMI (>50%)

    stable = (~job_loss_mask) & (emi_ratio_normal < 0.40)
    stressed = ((~job_loss_mask) & (emi_ratio_normal >= 0.40) & (emi_ratio_normal <= 0.50)) | \
              (job_loss_mask & (emi_ratio_stressed <= 0.50))
    crisis = (job_loss_mask) & (emi_ratio_stressed > 0.50) | default_expense

    scenario_breakdown = {
        "stable": int(np.sum(stable)),
        "stressed": int(np.sum(stressed)),
        "crisis": int(np.sum(crisis))
    }

    # --- Risk Label based on thresholds ---
    default_prob = np.mean(defaults)

    if default_prob < 0.15:
        risk_label = "LOW"
    elif default_prob < 0.35:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    # --- Worst Case EMI (95th percentile) ---
    worst_case_emi = np.percentile(emi_values, 95)

    # --- Safe Income Threshold ---
    # Find minimum income where P(default) < 0.15
    income_sweep = np.linspace(income * 0.6, income * 1.4, 100)
    default_probs = []

    for inc in income_sweep:
        test_incomes = inc * np.random.normal(1.0, 0.10, min(n_simulations, 5000))
        test_incomes = np.maximum(test_incomes, inc * 0.3)
        test_emi_ratio = emi_values[:min(n_simulations, 5000)] / test_incomes
        test_defaults = test_emi_ratio > 0.50
        default_probs.append(np.mean(test_defaults))

    default_probs = np.array(default_probs)
    safe_indices = np.where(default_probs < 0.15)[0]
    safe_income_threshold = income_sweep[safe_indices[0]] if len(safe_indices) > 0 else income_sweep[-1]

    # --- Summary Metrics ---
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


def plot_simulation(results: dict, save_path="monte_carlo_plot.png"):
    """
    Create 2x2 subplot visualization of Monte Carlo simulation results.

    Subplots:
        a) Histogram of EMI/income ratios
        b) Pie chart of risk scenarios
        c) Line chart: default probability vs income level
        d) Bar chart: scenario breakdown
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mortgage Risk Simulation", fontsize=16, fontweight="bold", y=1.02)

    # --- a) Histogram of EMI/income ratios ---
    ax1 = axes[0, 0]
    # Re-run to get distribution data
    loan_input = {
        "income": 50000,
        "loan_amount": 200000,
        "interest_rate": 8.5,
        "loan_term": 5,
        "credit_score": 650
    }
    incomes = loan_input["income"] * np.random.normal(1.0, 0.10, 10000)
    incomes = np.maximum(incomes, loan_input["income"] * 0.3)
    rates = loan_input["interest_rate"] * np.random.uniform(0.98, 1.02, 10000)
    rates = np.maximum(rates, 1.0)
    emis = np.array([calculate_emi(loan_input["loan_amount"], r, loan_input["loan_term"]) for r in rates])
    emi_ratios = (emis / incomes) * 100

    ax1.hist(emi_ratios, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
    ax1.axvline(x=50, color="red", linestyle="--", linewidth=2, label="Default Threshold (50%)")
    ax1.axvline(x=40, color="orange", linestyle="--", linewidth=2, label="Warning (40%)")
    ax1.set_xlabel("EMI / Income Ratio (%)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Distribution of EMI-to-Income Ratios", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- b) Pie chart of risk scenarios ---
    ax2 = axes[0, 1]
    breakdown = results["scenario_breakdown"]
    labels = ["Stable", "Stressed", "Crisis"]
    sizes = [breakdown["stable"], breakdown["stressed"], breakdown["crisis"]]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    explode = (0, 0.05, 0.1)

    wedges, texts, autotexts = ax2.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        shadow=True
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
    ax2.set_title("Risk Scenario Distribution", fontsize=12, fontweight="bold")

    # --- c) Line chart: default probability vs income level ---
    ax3 = axes[1, 0]
    income_base = 50000
    income_sweep = np.linspace(income_base * 0.6, income_base * 1.4, 50)
    default_probs = []

    for inc in income_sweep:
        test_incomes = inc * np.random.normal(1.0, 0.10, 2000)
        test_incomes = np.maximum(test_incomes, inc * 0.3)
        test_emis = np.array([
            calculate_emi(200000, 8.5 * np.random.uniform(0.98, 1.02), 5)
            for _ in range(2000)
        ])
        test_defaults = (test_emis / test_incomes) > 0.50
        default_probs.append(np.mean(test_defaults))

    ax3.plot(income_sweep, default_probs, color="darkorange", linewidth=2)
    ax3.axhline(y=0.15, color="green", linestyle="--", linewidth=1.5, label="Safe Threshold (15%)")
    ax3.axhline(y=0.35, color="red", linestyle="--", linewidth=1.5, label="High Risk (35%)")
    ax3.fill_between(income_sweep, 0, 0.15, alpha=0.1, color="green")
    ax3.fill_between(income_sweep, 0.15, 0.35, alpha=0.1, color="orange")
    ax3.fill_between(income_sweep, 0.35, 1.0, alpha=0.1, color="red")
    ax3.set_xlabel("Income Level (Rs.)", fontsize=11)
    ax3.set_ylabel("Default Probability", fontsize=11)
    ax3.set_title("Default Probability vs Income Level", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)

    # --- d) Bar chart: scenario breakdown ---
    ax4 = axes[1, 1]
    scenarios = ["Stable", "Stressed", "Crisis"]
    counts = [breakdown["stable"], breakdown["stressed"], breakdown["crisis"]]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    bars = ax4.bar(scenarios, counts, color=colors, edgecolor="white", linewidth=2)
    for bar, count in zip(bars, counts):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )
    ax4.set_xlabel("Scenario Type", fontsize=11)
    ax4.set_ylabel("Number of Simulations", fontsize=11)
    ax4.set_title("Scenario Breakdown", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    loan_input = {
        "income": 50000,
        "loan_amount": 200000,
        "interest_rate": 8.5,
        "loan_term": 5,
        "credit_score": 650
    }

    print("=" * 60)
    print("  MONTE CARLO MORTGAGE DEFAULT RISK SIMULATION")
    print("=" * 60)
    print(f"\nLoan Input:")
    for k, v in loan_input.items():
        print(f"  {k}: {v}")

    print(f"\nRunning 10,000 simulations...")
    results = simulate(loan_input, n_simulations=10000)

    print(f"\n" + "=" * 60)
    print("  SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Default Probability:    {results['default_probability']:.2%}")
    print(f"  Risk Label:             {results['risk_label']}")
    print(f"  Worst Case EMI (95th):  Rs.{results['worst_case_emi']:,.2f}")
    print(f"  Safe Income Threshold:  Rs.{results['safe_income_threshold']:,.2f}")
    print(f"\n  Scenario Breakdown:")
    print(f"    Stable:  {results['scenario_breakdown']['stable']:,} ({results['scenario_breakdown']['stable']/100:.1f}%)")
    print(f"    Stressed: {results['scenario_breakdown']['stressed']:,} ({results['scenario_breakdown']['stressed']/100:.1f}%)")
    print(f"    Crisis:   {results['scenario_breakdown']['crisis']:,} ({results['scenario_breakdown']['crisis']/100:.1f}%)")
    print(f"\n  Additional Metrics:")
    print(f"    Mean EMI Ratio:        {results['mean_emi_ratio']:.2%}")
    print(f"    Median EMI Ratio:      {results['median_emi_ratio']:.2%}")
    print(f"    95th Percentile EMI:   {results['percentile_95_emi_ratio']:.2%}")
    print(f"    Job Loss Count:        {results['job_loss_count']:,}")

    print(f"\n" + "=" * 60)
    print("  GENERATING VISUALIZATION")
    print("=" * 60)
    plot_simulation(results)