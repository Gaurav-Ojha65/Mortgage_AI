# Mortgage Risk Analytics AI/ML — System Architecture

## 1. System Overview

Mortgage Risk Analytics is a production-oriented machine learning decision-support platform designed for mortgage credit risk underwriting. The system combines tree-based probability estimation, out-of-fold calibration, game-theoretic explainability (SHAP), fair-lending fairness auditing, and an illustrative economic cost-sensitive decision engine.

---

## 2. High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Web Browser │  │ Analyst UI  │  │  API Client │  │ Auditor UI  │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
└─────────┼────────────────┼────────────────┼────────────────┼───────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        API & INFERENCE GATEWAY                             │
│                              (FastAPI)                                     │
│  - Request Validation & Schema Alignment (15 MODEL_FEATURES)               │
│  - Privacy-Oriented Controls & Structured Audit Logging                    │
│  - Role-Based Access Tiers (Admin / Analyst / Auditor)                     │
└──────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         MACHINE LEARNING PIPELINE                          │
│                                                                            │
│  ┌─────────────────────────┐     ┌──────────────────────────────────────┐  │
│  │ 1. LightGBM Base Model  │ ──► │ Raw Margin f(x) = log(p/(1-p))       │  │
│  │    (500 trees, lr=0.03) │     │ (Explains Model Score via SHAP)      │  │
│  └───────────┬─────────────┘     └──────────────────┬───────────────────┘  │
│              │                                      │                      │
│              ▼                                      ▼                      │
│  ┌─────────────────────────┐     ┌──────────────────────────────────────┐  │
│  │ 2. OOF Isotonic         │     │ SHAP TreeExplainer                   │  │
│  │    Calibrator           │     │ - Game-Theoretic Attributions        │  │
│  │    (5-Fold CV fit)      │     │ - Exact Additivity to Machine Prec   │  │
│  └───────────┬─────────────┘     └──────────────────────────────────────┘  │
│              │                                                             │
│              ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 3. Calibrated Default Probability p_cal ∈ [0.0, 1.0]                 │  │
│  │    (Brier = 0.0494, Weighted ECE = 0.0018 on untouched test data)    │  │
│  └───────────────────────────────────┬──────────────────────────────────┘  │
└──────────────────────────────────────┼─────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                  3-TIER ECONOMIC DECISION POLICY ENGINE                    │
│                                                                            │
│  - Automatic Approval:    p_cal <= 0.045  (Low Risk, FN Cost = $10,000)   │
│  - Manual Underwriting:   0.045 < p < 0.335 (Triage Cost = $150)           │
│  - Automatic Rejection:   p_cal >= 0.335  (High Risk, FP Cost = $1,000)   │
└──────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                   GOVERNANCE, AUDITING & MONITORING                        │
│                                                                            │
│  ┌─────────────────────────┐     ┌──────────────────────────────────────┐  │
│  │ Fairness Auditing       │     │ Drift & Calibration Monitoring       │  │
│  │ - Disparity Metrics     │     │ - PSI & Kolmogorov-Smirnov Tests     │  │
│  │ - Subgroup Calibrations │     │ - ECE & Brier Performance Tracking   │  │
│  └─────────────────────────┘     └──────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Machine Learning Subsystem

### 1. Feature Engineering & Schema Integrity
- Features: Exact 15 variables matching `MODEL_FEATURES`.
- Delinquency Representation: `late_payment_severity_score` (severity-weighted composite of 30, 60, and 90+ day delinquencies), `num_derogatory_marks` (formal 90+ day marks), and `num_late_payments` (30–59 day count).
- Data Segregation: Real training records ($N=99,856$) isolated in `data/real_train.csv` without synthetic distortion during out-of-fold calibration fitting.

### 2. Probability Calibration
- Out-of-Fold 5-Fold Cross-Calibration ensures zero data leakage between training, calibration, and policy validation.
- Isotonic regression transforms uncalibrated margins into empirical probabilities with a test-set Weighted ECE of $0.0018$.

### 3. Explainability Architecture
- TreeExplainer calculates feature-level contributions directly on raw margins.
- Clean separation between algorithmic predictive factors (SHAP) and operational underwriting thresholds (Decision Policy).

---

## 4. Operational Governance & Disclaimers

1. **Decision Support:** The platform functions as an automated decision-support tool. Applications with intermediate risk are systematically routed to human underwriters.
2. **Economic Cost Model:** Cost parameters ($C_{FN}=\$10,000$, $C_{FP}=\$1,000$, $C_{\text{Review}}=\$150$) are configurable demonstration values intended for simulation and trade-off analysis.
3. **Fair-Lending Fairness Audits:** Disparity metrics (demographic parity, equal opportunity, equalized odds) are computed to support internal risk management and do not constitute formal statutory certifications.
