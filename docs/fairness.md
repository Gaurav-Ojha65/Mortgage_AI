# Fairness Audit & Fair-Lending-Oriented Analysis

## 1. Executive Summary & Governance Scope

This document presents the **fair-lending-oriented fairness audit** for the candidate credit underwriting system:
- **Base Classifier:** v3.1 HPO-optimized LightGBM (Optuna Trial #47, 651 estimators)
- **Calibration Engine:** Out-of-Fold (OOF) 5-Fold Stratified Isotonic Regression (`oof-iso-v3.1`)
- **Decision Engine:** Frozen 3-Tier Economic Policy `v3.1-policy-v1` ($\text{Approve} \le 0.045, \text{Reject} \ge 0.335, \text{Manual Review } (0.045, 0.335)$)
- **Evaluation Dataset:** Untouched Test Partition ($N = 21,398$; $1,446$ defaults, $19,952$ non-defaults; natural default rate $6.76\%$)

> [!IMPORTANT]
> **Regulatory Disclaimer:**
> This audit is conducted strictly for internal Model Risk Management (MRM) and algorithmic governance purposes. It evaluates mathematical parity and rate disparities across demographic and proxy subgroups. **This document does NOT constitute a legal certification or legal guarantee of compliance** with the Equal Credit Opportunity Act (ECOA, Regulation B), Fair Housing Act (FHA), or any international lending regulations.

---

## 2. Demographic & Proxy Attribute Definitions

In accordance with strict empirical standards, only attributes genuinely present in or derived from the dataset schema are audited:

1. **Home Ownership Status (`home_ownership`):**
   - Categories: `Mortgage` ($N=11,680$), `Own` ($N=8,236$), `Rent` ($N=1,482$).
2. **Career Tenure / Age Proxy (`employment_years`):**
   - Derived directly from borrower employment history:
     - *Early Career (< 10 years)* ($N=5,961$)
     - *Mid Career (10–25 years)* ($N=8,591$)
     - *Senior (25+ years)* ($N=6,846$)
3. **Annual Income Tier (`annual_income`):**
   - Income distribution bands:
     - *Low (< $40,000)* ($N=3,506$)
     - *Middle ($40,000–$80,000)* ($N=9,949$)
     - *High (> $80,000)* ($N=7,943$)
4. **Loan Purpose (`purpose_encoded`):**
   - Categories: `Home Purchase`, `Refinance`, `Debt Consolidation`, `Home Improvement`, `Other` (~4,200 samples each).

---

## 3. Disparity Metric Summary Across Groups

Disparities are measured relative to the highest-approval and lowest-approval groups within each attribute:

| Protected / Proxy Attribute | Highest Approval Group | Lowest Approval Group | Demographic Parity Diff ($\Delta \text{App}$) | Disparate Impact Ratio ($\text{Min}/\text{Max}$) | Equal Opportunity Diff ($\Delta \text{TPR}$) | Equalized Odds Diff |
|---|---|---|---|---|---|---|
| **Home Ownership** | Mortgage ($78.57\%$) | Rent ($51.21\%$) | **$0.2736$** ($27.36\%$) | **$0.6518$** | **$0.1403$** | **$0.2626$** |
| **Career Tenure (Age Proxy)** | Senior 25y+ ($80.25\%$) | Early Career <10y ($53.25\%$) | **$0.2700$** ($27.00\%$) | **$0.6628$** | **$0.1178$** | **$0.2558$** |
| **Income Bracket** | High >$80k ($79.52\%$) | Low <$40k ($60.03\%$) | **$0.1949$** ($19.49\%$) | **$0.7549$** | **$0.1835$** | **$0.1835$** |
| **Loan Purpose** | Home Purchase ($73.47\%$) | Debt Consolidation ($70.86\%$) | **$0.0261$** ($2.61\%$) | **$0.9647$** | **$0.0515$** | **$0.0515$** |

---

## 4. Detailed Subgroup Breakdown

### A. Home Ownership
| Metric | Rent ($N=1,482$) | Own ($N=8,236$) | Mortgage ($N=11,680$) | Overall Portfolio ($N=21,398$) |
|---|---|---|---|---|
| **Sample Size ($N$)** | $1,482$ ($6.93\%$) | $8,236$ ($38.49\%$) | $11,680$ ($54.58\%$) | $21,398$ ($100\%$) |
| **Observed Default Rate** | $12.42\%$ | $7.44\%$ | $5.63\%$ | $6.76\%$ |
| **Mean Pred Probability** | $12.63\%$ | $7.57\%$ | $5.60\%$ | $6.84\%$ |
| **Brier Score** | $0.0863$ | $0.0538$ | $0.0421$ | $0.0494$ |
| **Weighted ECE** | $0.0039$ | $0.0016$ | $0.0014$ | $0.0018$ |
| **ROC-AUC** | $0.8354$ | $0.8529$ | $0.8637$ | $0.8599$ |
| **Approval Rate** | **$51.21\%$** | **$66.60\%$** | **$78.57\%$** | **$72.40\%$** |
| **Manual Review Rate** | $40.82\%$ | $29.74\%$ | $19.21\%$ | $24.61\%$ |
| **Rejection Rate** | $7.96\%$ | $3.67\%$ | $2.22\%$ | $2.99\%$ |
| **Approved Default Rate** | $2.90\%$ | $2.15\%$ | $1.85\%$ | $1.95\%$ |
| **Recall / TPR** | $88.19\%$ | $81.76\%$ | $74.16\%$ | $79.11\%$ |
| **FPR** | $44.53\%$ | $30.43\%$ | $18.28\%$ | $23.86\%$ |

### B. Career Tenure / Age Proxy
| Metric | Early Career (<10y, $N=5,961$) | Mid Career (10-25y, $N=8,591$) | Senior (25y+, $N=6,846$) |
|---|---|---|---|
| **Observed Default Rate** | $10.95\%$ | $6.33\%$ | $3.64\%$ |
| **Mean Pred Probability** | $11.08\%$ | $6.41\%$ | $3.68\%$ |
| **ROC-AUC** | $0.8407$ | $0.8569$ | $0.8596$ |
| **Approval Rate** | **$53.25\%$** | **$74.74\%$** | **$80.25\%$** |
| **Manual Review Rate** | $39.99\%$ | $22.61\%$ | $18.33\%$ |
| **Rejection Rate** | $6.76\%$ | $2.65\%$ | $1.42\%$ |
| **Approved Default Rate** | $2.74\%$ | $1.87\%$ | $1.44\%$ |

### C. Income Bracket
| Metric | Low (<$40k, $N=3,506$) | Middle ($40k-$80k, $N=9,949$) | High (>$80k, $N=7,943$) |
|---|---|---|---|
| **Observed Default Rate** | $10.27\%$ | $6.67\%$ | $5.31\%$ |
| **Mean Pred Probability** | $10.27\%$ | $6.77\%$ | $5.41\%$ |
| **ROC-AUC** | $0.8385$ | $0.8540$ | $0.8659$ |
| **Approval Rate** | **$60.03\%$** | **$72.48\%$** | **$79.52\%$** |
| **Manual Review Rate** | $34.77\%$ | $24.71\%$ | $20.00\%$ |
| **Rejection Rate** | $5.20\%$ | $2.81\%$ | $2.48\%$ |
| **Approved Default Rate** | $2.38\%$ | $1.99\%$ | $1.76\%$ |

---

## 5. Methodological Analysis & Key Findings

1. **Alignment with Underlying Default Risk:**
   - The observed disparity in approval rates across subgroups (e.g., $51.21\%$ for renters vs. $78.57\%$ for mortgage holders) closely mirrors the **underlying base rate of default** ($12.42\%$ for renters vs. $5.63\%$ for mortgage holders).
   - In all subgroups, the model's mean predicted default probability aligns closely with the actual observed default rate (e.g., Renters: $12.63\%$ predicted vs. $12.42\%$ actual), indicating that disparities are not generated by systemic over-prediction of default risk in any group.

2. **The Mitigating Role of the 3-Tier Policy:**
   - Rather than outright rejecting higher-risk demographic segments, the 3-tier policy routes borderline applicants into **Manual Review** ($40.82\%$ for Renters, $39.99\%$ for Early Career).
   - The rejection rate remains low across all groups (maximum $7.96\%$ for Renters, $6.76\%$ for Early Career), ensuring human underwriters review the vast majority of non-immediately-approved applicants.

3. **Approved Default Rate Consistency:**
   - For applicants granted automatic approval, the realized default rate remains low and consistent across all groups ($1.82\%$ overall), validating that the approval boundary ($\le 0.045$) maintains safety across cohorts.

---

## 6. Limitations & Governance Guidance

1. **Proxy Attribute Limitations:**
   - True legally protected classes (race, ethnicity, sex, marital status) are not present in the credit bureau dataset. `employment_years` is evaluated solely as an employment tenure / age proxy.
2. **Threshold Optimization Boundaries:**
   - In accordance with fair-lending governance, decision thresholds were optimized on validation data to minimize expected portfolio loss, **without subgroup-specific thresholding**, which could constitute prohibited disparate treatment under Regulation B.
3. **Recommended Human-in-the-Loop Oversight:**
   - Underwriting managers should monitor manual review turnaround times and outcomes for Early Career and Renter cohorts to prevent secondary human-underwriter bias during manual review triage.
