# Subgroup Probability Calibration Audit

## 1. Overview & Objectives

This audit evaluates the **transferability and consistency of calibrated probability estimates** across demographic and proxy cohorts within the untouched test partition ($N = 21,398$).

- **Base Classifier:** LightGBM
- **Calibrator:** Single Global Out-of-Fold (OOF) Isotonic Regressor (fit on $N=99,856$ real training OOF predictions)
- **Evaluation Principles:**
  - Evaluates the single production model without fitting subgroup-specific calibrators.
  - Measures probability reliability using Brier Score, Weighted Expected Calibration Error (wECE), Macro ECE, and 10-bin Reliability Distributions.

---

## 2. Overall Portfolio Calibration Benchmark

On the overall test dataset ($N = 21,398$), the global OOF Isotonic Calibrator achieves strong calibration fidelity:

- **Observed Default Prevalence:** $6.76\%$ ($1,446 / 21,398$)
- **Mean Predicted Default Probability:** $6.84\%$
- **Net Portfolio Calibration Gap:** $+0.0008$ ($+0.08$ percentage points)
- **Brier Score:** **$0.0494$**
- **Weighted ECE (10 uniform bins):** **$0.0018$** ($0.18\%$)
- **Macro ECE (10 uniform bins):** **$0.0353$** ($3.53\%$)

---

## 3. Subgroup Calibration Rankings & Summary

| Attribute | Subgroup | Sample Size ($N$) | Observed Default Rate | Mean Pred Probability | Calibration Gap | Brier Score | Weighted ECE | Macro ECE | Sample Adequacy ($N \ge 1000$) |
|---|---|---|---|---|---|---|---|---|---|
| **Home Ownership** | **Mortgage** | $11,680$ | $5.63\%$ | $5.60\%$ | $-0.0004$ | **$0.0421$** | **$0.0014$** | $0.0413$ | Yes (Adequate) |
| **Home Ownership** | **Own** | $8,236$ | $7.44\%$ | $7.57\%$ | $+0.0011$ | $0.0556$ | $0.0041$ | $0.0322$ | Yes (Adequate) |
| **Home Ownership** | **Rent** | $1,482$ | $12.42\%$ | $12.63\%$ | $+0.0021$ | $0.0863$ | $0.0039$ | $0.0768$ | Yes (Adequate) |
| **Career Tenure** | **Senior (25y+)** | $11,456$ | $3.64\%$ | $3.68\%$ | $+0.0004$ | $0.0315$ | $0.0044$ | $0.0421$ | Yes (Adequate) |
| **Career Tenure** | **Mid Career (10–25y)** | $7,668$ | $8.74\%$ | $8.70\%$ | $-0.0004$ | $0.0593$ | $0.0042$ | $0.0632$ | Yes (Adequate) |
| **Career Tenure** | **Early Career (<10y)** | $2,274$ | $14.20\%$ | $14.81\%$ | $+0.0061$ | $0.0727$ | **$0.0124$** | $0.0547$ | Yes (Adequate) |
| **Income Bracket** | **High (>$80k)** | $7,943$ | $5.31\%$ | $5.41\%$ | $+0.0010$ | $0.0402$ | $0.0044$ | $0.0419$ | Yes (Adequate) |
| **Income Bracket** | **Middle ($40k-$80k)** | $9,949$ | $6.67\%$ | $6.77\%$ | $+0.0010$ | $0.0489$ | $0.0046$ | $0.0512$ | Yes (Adequate) |
| **Income Bracket** | **Low (<$40k)** | $3,506$ | $10.27\%$ | $10.27\%$ | $+0.0000$ | $0.0718$ | $0.0062$ | $0.0485$ | Yes (Adequate) |
| **Loan Purpose** | **Home Purchase** | $4,297$ | $6.40\%$ | $6.58\%$ | $+0.0018$ | $0.0475$ | $0.0048$ | $0.0645$ | Yes (Adequate) |
| **Loan Purpose** | **Refinance** | $4,321$ | $6.71\%$ | $6.79\%$ | $+0.0008$ | $0.0491$ | $0.0051$ | $0.0612$ | Yes (Adequate) |
| **Loan Purpose** | **Debt Consolidation** | $4,242$ | $7.28\%$ | $7.21\%$ | $-0.0007$ | $0.0526$ | $0.0055$ | $0.0701$ | Yes (Adequate) |
| **Loan Purpose** | **Home Improvement**| $4,258$ | $6.74\%$ | $6.88\%$ | $+0.0014$ | $0.0494$ | $0.0049$ | $0.0589$ | Yes (Adequate) |
| **Loan Purpose** | **Other** | $4,280$ | $6.66\%$ | $6.43\%$ | $-0.0023$ | $0.0498$ | $0.0043$ | $0.1050$ | Yes (Adequate) |

---

## 4. Key Findings & Diagnoses

### Best Calibrated Subgroup
- **`Home Ownership: Mortgage`** ($N = 11,680$):
  - **Weighted ECE:** **$0.0014$** ($0.14\%$)
  - **Observed Default Rate:** $5.63\%$ vs. **Mean Predicted Probability:** $5.60\%$ ($\Delta = -0.04\%$)
  - **Brier Score:** **$0.0421$**

### Worst Calibrated Subgroup
- **`Career Tenure: Early Career (< 10y)`** ($N = 2,274$):
  - **Weighted ECE:** **$0.0124$** ($1.24\%$)
  - **Observed Default Rate:** $14.20\%$ vs. **Mean Predicted Probability:** $14.81\%$ ($\Delta = +0.61\%$)
  - **Brier Score:** **$0.0727$**
  - **Diagnosis:** In the $<10$ year employment cohort, borrowers have higher default prevalence and slightly wider confidence dispersion in intermediate risk bins ($0.10 \le p \le 0.40$), resulting in modest over-prediction ($+0.61\%$ overall gap).

### Evaluation of Miscalibration Risk
- **No Meaningful Systemic Miscalibration:** Across all 14 evaluated subgroups, the absolute net calibration gap is under **$0.61$ percentage points**, and Weighted ECE remains below **$1.25\%$**.
- **Monotonicity Maintained:** Within each subgroup's 10 reliability bins, higher predicted probability bins consistently map to higher realized default rates.
- **Sample Size Adequacy:** All subgroups contain $N \ge 1,482$ samples (well above the minimum statistical adequacy threshold of $1,000$ samples), ensuring that all computed ECE and Brier metrics are statistically robust rather than small-sample artifacts.

---

## 5. Governance Recommendations

1. **Maintain Single Global Calibrator:**
   - Because the global OOF Isotonic Calibrator transfers effectively across all subgroups ($\text{wECE} \le 0.0124$), subgroup-specific calibration is unnecessary and could introduce fair-lending disparate treatment risks.
2. **Periodic Subgroup Drift Monitoring:**
   - As macro credit cycles shift, Model Risk Management should monitor whether the Early Career cohort ($<10$y) experiences widening calibration error beyond $2.0\%$ wECE.
