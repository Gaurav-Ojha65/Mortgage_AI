# Mortgage Risk Analytics AI/ML — System Overview

## 1. System Overview

A mortgage loan risk analytics and decision-support platform featuring calibrated machine learning, fair-lending fairness auditing, game-theoretic explainability (SHAP), and operational economic cost modeling.

---

## 2. Model Version History

### v3.1 — Canonical Production Baseline (Promoted)
- **Model Description:** **v3.1 HPO-optimized LightGBM**
- **Pipeline:** Real training data → Fold-internal SMOTE → LightGBM (HPO-tuned: 651 trees, lr=0.022, depth=10, leaves=20) → Fresh 5-Fold OOF Isotonic Calibration (`oof-iso-v3.1`) → Validation-derived 3-tier policy (`v3.1-policy-v1`: Approve ≤ 0.045, Reject ≥ 0.335).
- **HPO Provenance:** 75-trial Optuna search (TPE sampler, MedianPruner), best trial #47.
- **Measured Test-set Benchmark** (N=21,398 untouched):
  - ROC-AUC: 0.8615 | PR-AUC: 0.3995 | Brier: 0.0492 | wECE: 0.0012 | Macro ECE: 0.0129
  - Expected Cost: $4,062,100 ($189.84/applicant) | Review Rate: 24.09%
- **Statistical Uncertainty (1,000 paired bootstrap resamples vs v3.0.0):**
  - Δ ROC-AUC: 95% CI [+0.0002, +0.0032] — excludes zero (statistically distinguishable).
  - Δ Brier: 95% CI [-0.0005, -0.0001] — excludes zero (statistically distinguishable).
  - Δ PR-AUC: 95% CI [-0.0011, +0.0104] — includes zero (not statistically distinguishable).
  - Δ Total Cost: 95% CI [-$153,881, +$109,024] — includes zero (observed cost improvement is not statistically distinguishable under bootstrap).
- **Status:** Promoted canonical baseline.

### v3.0.0 — Previous Canonical Baseline (Archived in `ml/models/archive/v3.0.0/`)
- **Pipeline:** Real training data → Fold-internal SMOTE → LightGBM (500 trees, lr=0.03) → 5-Fold OOF Isotonic Calibration → Validation-derived 3-tier policy (Approve ≤ 0.055, Reject ≥ 0.405).
- **Test-set Benchmark** (N=21,398 untouched):
  - ROC-AUC: 0.8599 | PR-AUC: 0.3947 | Brier: 0.0494 | wECE: 0.0018 | Macro ECE: 0.0353
  - Expected Cost: $4,082,900 ($190.81/applicant) | Review Rate: 24.61%
- **Status:** Archived and preserved for historical reproducibility.

---

## 3. Core System Components

### 1. Machine Learning & Calibration Engine (`ml/`)
- **Canonical Model:** LightGBM classifier with 5-Fold Out-of-Fold (OOF) Isotonic Regression calibration fitted on $N=99,856$ real training samples without synthetic distortion.
- **Feature Schema:** Exactly 15 standardized features matching `MODEL_FEATURES`, utilizing the delinquency composite `late_payment_severity_score`.

### 2. Economic Decision Policy Engine (`risk/decision_policy.py`)
- **Architecture:** 3-tier routing based on calibrated default probability ($p_{\text{cal}}$).
- **Cost Model (Demonstration / Illustrative):**
  - False Negative Cost ($C_{FN}$): $\$10,000$ (Loss Given Default)
  - False Positive Cost ($C_{FP}$): $\$1,000$ (Lost Net Interest Margin)
  - Manual Review Cost ($C_{\text{Review}}$): $\$150$ (Operational Triage)

### 3. Model Explainability & SHAP (`ml/training/shap_validation.py`, `backend/shap_explainer.py`)
- **TreeExplainer:** Exact game-theoretic Shapley feature attributions computed on raw tree log-odds scores.
- **Exact Additivity:** Verified to machine precision ($1.24 \times 10^{-14}$ max reconstruction error for v3.1 HPO candidate).
- **Architectural Separation:** Explicit boundary between Model Output Explanation (Tree log-odds) and Policy Routing Explanation (Economic loss trade-off).

### 4. Fair-Lending Fairness Auditing & Subgroup Calibration
- **Fairness Audits:** Disparity tracking across home ownership, career tenure (age proxy), income bands, and loan purpose.
- **Calibration Transfer:** Global OOF calibrator verified across all cohorts.
- **Governance Role:** Analytical decision-support metrics for model risk management (MRM) without legal compliance claims.

### 5. Distribution Drift & Performance Monitoring (`monitoring/`)
- **Data Drift:** PSI, Kolmogorov-Smirnov (KS) tests, and Wasserstein distance tracking for feature distributions.
- **Model Drift:** Brier score degradation and calibration drift alerting against frozen baselines.

### 6. Backend API & Privacy Controls (`backend/`)
- **FastAPI Framework:** Asynchronous endpoints for inference, 3-tier policy evaluation, and SHAP explainability.
- **Role-Based Access Control:** Configurable role tiers (admin, analyst, auditor).
- **Privacy-Oriented Controls:** Structured JSON audit logging and request tracing.
- **Measured Inference Latency:** 2.64 ms/applicant in the benchmark environment.

