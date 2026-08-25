# Model Card: Mortgage AI v3.1 HPO-Optimized LightGBM

## 1. Model Details & Provenance
- **Model Name:** Mortgage AI Credit Risk & Underwriting Decision Model
- **Model Version:** `v3.1` (Canonical Baseline)
- **Model Architecture:** Gradient Boosted Decision Trees via LightGBM (`LGBMClassifier`)
- **Optimization:** Hyperparameter Optimization via Optuna (5-Fold Stratified Cross-Validation on `real_train.csv`, Trial #47)
- **Hyperparameters:**
  - `n_estimators`: 651
  - `learning_rate`: 0.02197
  - `max_depth`: 10
  - `num_leaves`: 20
  - `min_child_samples`: 23
  - `subsample`: 0.6806
  - `colsample_bytree`: 0.5162
  - `reg_alpha`: 0.4564
  - `reg_lambda`: 1.493e-05
- **Calibration Engine:** 5-Fold Stratified Out-of-Fold (OOF) Isotonic Regression (`oof-iso-v3.1`)
- **Decision Policy:** 3-Tier Cost-Sensitive Policy (`v3.1-policy-v1`) optimized strictly on validation data:
  - **Auto-Approve:** $p_{\text{cal}} \le 0.045$
  - **Manual Review:** $0.045 < p_{\text{cal}} < 0.335$
  - **Auto-Reject:** $p_{\text{cal}} \ge 0.335$

---

## 2. Intended Use & Scope
- **Intended Use:** Automated credit risk scoring, calibrated probability of default estimation, and underwriting triage for residential mortgage applications.
- **Intended Users:** Mortgage underwriters, credit risk analysts, and loan officers.
- **Out-of-Scope Use:**
  - Fully automated adverse action (rejection) without human review for applicants falling in the manual review tier ($0.045 < p < 0.335$).
  - Commercial, agricultural, or non-residential lending applications.
  - Direct substitute for regulatory compliance audits or legal determinations.

---

## 3. Dataset Provenance & Training Methodology
- **Training Source:** `data/real_train.csv` ($N = 99,856$ historical residential mortgage applications).
- **Validation Source:** `ml/data/val.csv` ($N = 21,398$).
- **Test Source:** `ml/data/test.csv` ($N = 21,398$, untouched holdout used strictly for single qualification benchmark).
- **Class Balancing:** Fold-internal SMOTE applied exclusively within CV training folds; never applied to validation or test splits.
- **Input Features (15 canonical features):**
  1. `credit_score`: FICO/bureau score [300–850]
  2. `annual_income`: Gross annual income ($USD)
  3. `loan_amount`: Requested principal amount ($USD)
  4. `loan_term`: Loan term in months [12–360]
  5. `dti_ratio`: Debt-to-income ratio [0.0–1.0]
  6. `employment_years`: Years in current employment [0–40]
  7. `num_credit_lines`: Total open credit lines [0–30]
  8. `num_derogatory_marks`: Derogatory records/delinquencies [0–10]
  9. `credit_utilization`: Revolving credit utilization ratio [0.0–1.0]
  10. `late_payment_severity_score`: Severity-weighted payment history score [0.0–1.0]
  11. `home_ownership`: Categorical encoded (0=Rent, 1=Own, 2=Mortgage)
  12. `purpose_encoded`: Loan purpose category code [0–9]
  13. `num_late_payments`: Count of 30+ day past-due payments [0–20]
  14. `savings_balance`: Liquid reserve assets ($USD)
  15. `monthly_expenses`: Recurring monthly debt & living expenses ($USD)

---

## 4. Benchmark Evaluation (Untouched Test Holdout, $N = 21,398$)

| Metric | Measured Value | 95% Bootstrap Confidence Interval ($B=1,000$) |
|---|---|---|
| **ROC-AUC** | **0.8615** | $[0.8519, 0.8711]$ |
| **PR-AUC** | **0.3995** | $[0.3701, 0.4300]$ |
| **Brier Score** | **0.0492** | $[0.0470, 0.0515]$ |
| **Weighted ECE** | **0.0012** ($0.12\%$) | — |
| **Macro ECE** | **0.0129** ($1.29\%$) | — |
| **Expected Portfolio Cost** | **$4,062,100** ($189.84$/app) | $[-\$153,881, +\$109,024]$ vs v3.0.0 |
| **Auto-Approval Rate** | **71.05%** | Approved default rate: $1.82\%$ |
| **Manual Review Rate** | **24.09%** | Within operational ceiling $\le 25\%$ |
| **Auto-Rejection Rate** | **4.86%** | High-risk applicant cohort |
| **SHAP Additivity Error** | **$1.24 \times 10^{-14}$** | Machine precision exact |

---

## 5. Explainability & Fair Lending Governance
- **Local Explanations:** Exact Shapley values calculated per prediction using `shap.TreeExplainer` on the raw tree ensemble log-odds margin.
- **Top Risk Drivers:** `late_payment_severity_score`, `credit_utilization`, `dti_ratio`, `credit_score`.
- **Fairness Monitoring:** Demographic proxy tracking across age bands and regional tiers. Manual review triage directs borderline decisions to human underwriters.
- **Disclaimer:** This model card documents technical characteristics and statistical benchmarks. It does not constitute legal certification or regulatory compliance certification under ECOA/FCRA.

---

## 6. Known Limitations & Risks
- **Macroeconomic Shifts:** Model is calibrated under baseline economic conditions; severe interest rate shocks or macroeconomic stress may require threshold adjustments via validation split recalibration.
- **Sparse Feature Subspaces:** Extreme outliers in `savings_balance` or `annual_income` may have wider prediction uncertainty intervals.
- **Non-Linear Threshold Dynamics:** Policy thresholds ($0.045 / 0.335$) are optimized for the specified demonstration cost matrix ($C_{\text{FN}}=\$10,000, C_{\text{FP}}=\$1,000, C_{\text{REV}}=\$150$); real-world deployment requires enterprise cost calibration.

---

## 7. Reproducibility & Artifact Verification
- **Model Checksum (SHA-256):** Verified against `ml/models/training_metadata.json`
- **Calibrated Predictor:** `ml/models/lightgbm_calibrated_pipeline.joblib`
- **Frozen Policy Config:** `ml/models/frozen_policy_config.json`
