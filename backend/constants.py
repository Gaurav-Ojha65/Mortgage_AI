"""Application constants - centralized configuration."""

# Risk thresholds
RISK_HIGH_THRESHOLD = 0.35
RISK_LOW_THRESHOLD = 0.15

# Default triggers
DEFAULT_TRIGGER_RATIO = 0.50
BURDEN_WEIGHT = 0.15

# Monte Carlo parameters
JOB_LOSS_PROBABILITY = 0.08
INCOME_VARIATION_STD = 0.10
RATE_VARIATION_MIN = -0.02
RATE_VARIATION_MAX = 0.02
MAX_SIMULATIONS_DISPLAY = 1000

# EMI safety threshold
EMI_TO_INCOME_MAX = 40.0

# Credit score normalization
CREDIT_SCORE_MIN = 300
CREDIT_SCORE_MAX = 850

# Loan term limits
LOAN_TERM_YEARS_MIN = 1
LOAN_TERM_YEARS_MAX = 30
