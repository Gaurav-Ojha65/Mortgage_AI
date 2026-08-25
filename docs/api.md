# Mortgage Risk Analytics AI/ML — API & Integration Guide

## 1. Overview

This document provides the API specifications for the production-oriented Mortgage Risk Analytics backend service. The API delivers calibrated credit risk scoring, 3-tier economic decision routing, SHAP explainability, OCR document processing, fairness audits, and Prometheus monitoring.

**Active Baseline:** `v3.1 HPO-optimized LightGBM`
**Calibrator:** `oof-iso-v3.1` (5-Fold Stratified OOF Isotonic Regression)
**Underwriting Policy:** `v3.1-policy-v1` ($\text{Approve} \le 0.045, \text{Reject} \ge 0.335$)

---

## 2. Core API Endpoints

### 1. Calibrated Decision Analysis: `POST /analyze`
Comprehensive scoring combining calibrated LightGBM default prediction, 3-tier economic policy routing, and SHAP explainability.

**Request Payload:**
```json
{
  "income": 85000,
  "loan_amount": 250000,
  "credit_score": 720,
  "interest_rate": 6.25,
  "loan_term": 30,
  "property_value": 350000,
  "existing_loans": 0
}
```

**Response Payload (`200 OK`):**
```json
{
  "success": true,
  "data": {
    "decision": "APPROVE",
    "risk_level": "LOW",
    "risk_tier": "LOW",
    "raw_default_probability": 0.0312,
    "calibrated_default_probability": 0.0285,
    "approval_probability": 0.9715,
    "expected_economic_cost": 285.00,
    "model_name": "lightgbm",
    "model_version": "v3.1",
    "calibration_method": "isotonic",
    "calibration_version": "oof-iso-v3.1",
    "policy_version": "v3.1-policy-v1",
    "top_factors": [
      "Late Payment Severity Score",
      "Credit Utilization",
      "Loan Term",
      "Open Credit Lines",
      "Home Ownership"
    ],
    "plain_english": [
      "Your payment severity score reflects low delinquency history.",
      "Credit utilization is well-managed."
    ]
  }
}
```

---

### 2. Decision Policy Configuration: `GET /policy/config`
Retrieves the currently active frozen 3-tier decision policy and cost model.

**Response Payload (`200 OK`):**
```json
{
  "success": true,
  "data": {
    "active_policy": {
      "policy_name": "optimized_3tier_economic_policy",
      "policy_version": "v3.1-policy-v1",
      "approve_threshold": 0.045,
      "reject_threshold": 0.335,
      "cost_model": {
        "cost_fn": 10000.0,
        "cost_fp": 1000.0,
        "cost_manual_review": 150.0,
        "is_demonstration": true
      }
    }
  }
}
```

---

### 3. Custom Policy Simulation: `POST /policy/evaluate`
Simulates custom underwriting thresholds against an applicant without altering the canonical system policy.

**Request Payload:**
```json
{
  "application": {
    "income": 95000,
    "loan_amount": 200000,
    "credit_score": 740,
    "interest_rate": 6.5,
    "loan_term": 30
  },
  "approve_threshold": 0.045,
  "reject_threshold": 0.335
}
```

---

### 4. Service Health & Provenance: `GET /health`
Returns system health, database connectivity, and active model/policy versions.

**Response Payload (`200 OK`):**
```json
{
  "success": true,
  "data": {
    "status": "ok",
    "version": "2.0.0",
    "model_version": "v3.1",
    "calibration_version": "oof-iso-v3.1",
    "policy_version": "v3.1-policy-v1",
    "policy_thresholds": {
      "approve_threshold": 0.045,
      "reject_threshold": 0.335
    },
    "uptime_seconds": 124,
    "models_loaded": true,
    "active_model": "lightgbm",
    "db_connected": true,
    "predictions_served": 42
  }
}
```

---

### 5. Prometheus Metrics: `GET /metrics`
Exports Prometheus-formatted metric gauges and counters for operational monitoring.

---

### 6. Financial Document OCR: `POST /api/documents/upload`
Extracts financial fields from pay stubs, bank statements, and tax forms (PDF, PNG, JPG).

---

### 7. Fairness Auditing: `GET /api/fairness/report`
Retrieves demographic parity, disparate impact ratios, and subgroup calibration metrics across protected cohorts.
