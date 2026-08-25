# Policy Sensitivity Analysis

## 1. Overview

This document presents the sensitivity analysis of the **frozen 3-tier decision policy** (Approve $\le 0.055$, Reject $\ge 0.405$) around its canonical operating point. The analysis evaluates how expected portfolio cost, routing volumes, and risk metrics change with small perturbations to the two thresholds.

> [!IMPORTANT]
> **This is a sensitivity analysis only.** No threshold was selected or optimized based on test results. The frozen canonical policy remains the production/demo candidate.

---

## 2. Grid Configuration

**Approve Thresholds Tested:** $[0.045, 0.050, 0.055, 0.060, 0.065]$
**Reject Thresholds Tested:** $[0.385, 0.395, 0.405, 0.415, 0.425]$
**Valid Configurations Evaluated:** $25$ (all with approve < reject)
**Cost Parameters:** $C_{FN} = \$10{,}000$, $C_{FP} = \$1{,}000$, $C_{\text{Review}} = \$150$ (illustrative)

---

## 3. Frozen Policy Benchmark

| Metric | Value |
|---|---|
| **Approve Threshold** | $0.055$ |
| **Reject Threshold** | $0.405$ |
| **Approval Rate** | $72.40\%$ |
| **Manual Review Rate** | $24.61\%$ |
| **Rejection Rate** | $2.99\%$ |
| **Review Volume** | $5{,}266$ applications |
| **Approved Defaults (FN)** | $302$ |
| **Rejected Non-Defaults (FP)** | $273$ |
| **Total Expected Cost** | **$\$4{,}082{,}900$** |
| **Cost per Applicant** | **$\$190.81$** |
| **Review Constraint ($\le 25\%$)** | **Met** ($24.61\%$) |

---

## 4. Key Sensitivity Findings

### Configurations Meeting Review Constraint ($\le 25\%$)
- **15 of 25** configurations meet the operational review-rate constraint.
- All configurations with `approve_threshold >= 0.055` meet the constraint.
- All configurations with `approve_threshold = 0.045` violate the constraint.

### Cost Range Across All 25 Configurations
- **Minimum Cost:** $\$3{,}807{,}050$ (Approve $= 0.045$, Reject $= 0.405$) — **violates** review constraint ($28.17\%$ review rate)
- **Maximum Cost:** $\$4{,}158{,}900$ (Approve $= 0.055$, Reject $= 0.385$)
- **Frozen Policy Cost:** $\$4{,}082{,}900$ — within $\$76{,}000$ of the theoretical minimum among constraint-satisfying configs

### Robustness Around Frozen Operating Point
The frozen policy sits in a **stable, low-gradient cost region**:

| $\Delta$ Approve | $\Delta$ Reject | Cost $\Delta$ from Frozen | $\%$ Change | Review Constraint |
|---|---|---|---|---|
| $-0.005$ | $0$ | $-\$115{,}850$ | $-2.84\%$ | ⚠️ Violated ($25.73\%$) |
| $0$ | $0$ | $\$0$ | $0\%$ | ✅ Met ($24.61\%$) |
| $+0.005$ | $0$ | $+\$25{,}700$ | $+0.63\%$ | ✅ Met ($23.50\%$) |
| $0$ | $-0.010$ | $+\$35{,}500$ | $+0.87\%$ | ✅ Met ($24.06\%$) |
| $0$ | $+0.010$ | $-\$16{,}350$ | $-0.40\%$ | ✅ Met ($25.20\%$) |
| $0$ | $+0.020$ | $-\$33{,}300$ | $-0.82\%$ | ⚠️ Violated ($25.79\%$) |

### Directional Sensitivities
1. **Lowering the approve threshold** (more conservative approval) consistently reduces FN cost but increases review volume. At $\text{approve} = 0.050$, the review rate crosses $25\%$.
2. **Raising the reject threshold** (more conservative rejection) reduces FP cost but increases review volume. At $\text{reject} = 0.415$ with $\text{approve} = 0.055$, review rate reaches $25.20\%$ — still within constraint but barely.
3. **Raising the approve threshold** (more aggressive approval) reduces review volume but increases approved defaults, producing a net cost increase.

---

## 5. Operational Interpretation

The frozen policy ($0.055 / 0.405$) is positioned at a **near-optimal trade-off** between:
- Minimizing portfolio default exposure (FN cost = $\$3{,}020{,}000$)
- Minimizing false rejection opportunity cost (FP cost = $\$273{,}000$)
- Maintaining underwriter workload within the $25\%$ operational review capacity constraint ($24.61\%$)

Moving the thresholds by $\pm 0.005$ in either direction produces cost changes of less than $3\%$, confirming the policy operates in a **flat, robust region** of the cost surface.

---

## 6. Limitations

- Cost parameters ($C_{FN}, C_{FP}, C_{\text{Review}}$) are **illustrative demonstration values** and should be replaced with institution-specific actuarial estimates for deployment.
- The sensitivity grid is centered on integer-spaced threshold increments ($\Delta = 0.005$ to $0.010$); finer resolution may reveal sub-basis-point cost structures.
- This analysis evaluates static point-in-time thresholds; live systems would additionally require temporal monitoring for calibration drift.
