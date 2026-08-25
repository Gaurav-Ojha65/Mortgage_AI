# Explainability & SHAP Architecture Validation

## 1. Overview & Architectural Principles

In credit risk decision systems, model explainability must be mathematically exact, computationally consistent, and clearly delineated from downstream decision routing.

This document details the audit and verification of SHAP (SHapley Additive exPlanations) for the **LightGBM credit default model**.

### Crucial Architectural Separation
```
+-----------------------------------------------------------------------------------------+
|                                  INFERENCE PIPELINE                                     |
|                                                                                         |
|  [Applicant Data]                                                                       |
|         │                                                                               |
|         ▼                                                                               |
|  [Feature Preprocessing] ──────► Exactly 15 MODEL_FEATURES                              |
|         │                                                                               |
|         ▼                                                                               |
|  [LightGBM Model]        ──────► Raw margin f(x) = log(p/(1-p))                         |
|         │                        │                                                      |
|         │                        └──────────────────────► [SHAP TreeExplainer]          |
|         │                                                 - Mathematical log-odds sum   |
|         │                                                 - Local feature contributions |
|         │                                                 - Explains MODEL output       |
|         ▼                                                                               |
|  [OOF Isotonic Calibrator] ────► Calibrated Default Probability p_cal ∈ [0.0, 1.0]      |
|         │                                                                               |
|         ▼                                                                               |
|  [3-Tier Economic Policy] ─────► Decision: APPROVE / MANUAL_REVIEW / REJECT             |
|                                  - Threshold routing (p_cal ≤ 0.045 / ≥ 0.335)          |
|                                  - Explains POLICY decision (Economic loss tradeoff)    |
+-----------------------------------------------------------------------------------------+
```

1. **Model Explanation (SHAP TreeExplainer):**
   - Answers: *"Why did the LightGBM model assign this specific raw risk margin / default probability to this applicant?"*
   - Calculated via game-theoretic Shapley attributions on the tree ensemble's raw log-odds score.
2. **Probability Calibration (Isotonic Regression):**
   - Maps raw model scores to empirically observed default frequencies without modifying the rank-order feature contributions.
3. **Policy Explanation (Economic Decision Engine):**
   - Answers: *"Why was this applicant routed to Automatic Approval, Manual Underwriting Review, or Rejection?"*
   - Determined by the two-threshold economic cost policy balancing False Negative costs ($10,000), False Positive costs ($1,000), and Manual Review costs ($150).

> [!NOTE]
> **Governance Note on Explanations:**
> SHAP values explain the **raw predictive mechanics of the LightGBM classifier**. They do **not** directly explain the isotonic monotonic transfer function, nor do they constitute formal adverse action notices or legal justifications.

---

## 2. Feature Schema & Order Audit

The SHAP pipeline was audited to ensure strict alignment with model training and inference schema:

- **Total Active Features:** Exact $15$ features matching `MODEL_FEATURES`.
- **Feature Order Consistency:** Exact 1-to-1 match between training columns, inference preprocessing, and TreeExplainer masks.
- **Obsolete Feature Detection:** Confirmed that legacy feature `payment_history_score` is **completely eliminated** from all inference, explainer, and template files.
- **Active Delinquency Composite:** `late_payment_severity_score` is consistently utilized across inference preprocessing, explainer dictionaries, and customer-facing templates.

---

## 3. Mathematical Additivity & Precision Check

For TreeExplainer, exact additivity requires that the sum of all local SHAP values plus the base expectation equals the model's raw margin:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$

Where:
- $\phi_0 = \mathbb{E}[f(X)] = -0.52241$ (Model Base Value / Expected Log-Odds)
- $\phi_i(x)$ = SHAP attribution for feature $i$ on applicant $x$

### Empirical Audit Results ($N=100$ Test Samples):
- **Is Mathematically Additive:** **TRUE**
- **Maximum Absolute Reconstruction Error:** **$1.15 \times 10^{-14}$**
- **Mean Absolute Reconstruction Error:** **$3.75 \times 10^{-15}$**

This proves that feature attributions are exact to floating-point machine precision.

---

## 4. Global Feature Importance (Test Set)

Calculated on test applications ($N = 1,000$, sampled across the entire distribution):

| Rank | Feature Name | Label | Mean Absolute SHAP ($\overline{\|\phi\|}$) | Risk Interpretation |
|---|---|---|---|---|
| 1 | `late_payment_severity_score` | Late Payment Severity Composite | **$1.1049$** | Primary driver of default risk (severity-weighted delinquency) |
| 2 | `credit_utilization` | Credit Utilization Ratio | **$0.5796$** | Revolving debt burden relative to credit limits |
| 3 | `loan_term` | Loan Term (months) | **$0.5618$** | Duration of principal exposure |
| 4 | `purpose_encoded` | Loan Purpose | **$0.5117$** | Purpose category (e.g. debt consolidation vs purchase) |
| 5 | `num_derogatory_marks` | Severe Delinquency (90+ days) | **$0.3660$** | Severe credit events (credit bureau derogatories) |
| 6 | `num_late_payments` | Mild Late Payments (30–59 days) | **$0.2618$** | Short-term payment friction |
| 7 | `num_credit_lines` | Open Credit Lines | **$0.1695$** | Credit depth and borrowing activity |
| 8 | `home_ownership` | Home Ownership | **$0.1449$** | Housing tenure stability (Mortgage vs Rent) |
| 9 | `employment_years` | Employment History (years) | **$0.1148$** | Career and income stability |
| 10 | `annual_income` | Annual Income | **$0.0923$** | Borrower gross repayment capacity |

---

## 5. Example Local Applicant Explanation

For a representative applicant with moderate credit risk:
- **Model Base Value ($\phi_0$):** $-0.5224$
- **Applicant Raw Margin ($f(x)$):** $-0.7719$
- **Raw Probability ($p_{\text{raw}}$):** $0.3161$
- **Calibrated Default Probability ($p_{\text{cal}}$):** $0.0612$
- **Policy Decision:** **`MANUAL_REVIEW`** (routed to human underwriters because $0.045 < 0.0612 < 0.335$)

### Top Feature Attributions:
1. `late_payment_severity_score = 1.0` $\to \phi = -0.9234$ (Strongly reduces default risk)
2. `credit_utilization = 0.178` $\to \phi = -0.4512$ (Reduces default risk)
3. `loan_term = 360` $\to \phi = +0.6120$ (Increases default risk due to 30-year term)
4. `num_derogatory_marks = 0` $\to \phi = -0.3210$ (Reduces default risk)
5. `home_ownership = Mortgage` $\to \phi = -0.1540$ (Reduces default risk)

Reconstruction Check:
$$\phi_0 + \sum \phi_i = -0.52241 - 0.24949 = -0.77190 = f(x)$$
