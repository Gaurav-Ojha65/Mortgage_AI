# Mortgage AI v3.1 — 2-Minute End-to-End Demo Script

This guide outlines the standard **2-minute live walkthrough** of the Mortgage AI v3.1 platform for technical evaluations, portfolio reviews, and architecture demonstrations.

---

## Demo Overview & Timing Breakdown

```
0:00 ───────────── 0:20 ───────────── 0:45 ───────────── 1:10 ───────────── 1:35 ───────────── 2:00
 │                  │                  │                  │                  │                  │
 └─ Core Value Prop ┴─ System Health  ┴─ Decision Flow   ┴─ SHAP Waterfall  ┴─ What-If & Audit ─┘
```

| Phase | Duration | Focus Area | Key UI Component / Route |
| :--- | :--- | :--- | :--- |
| **1. Value Prop & Arch** | 20 sec | Problem, OOF Calibration, 3-Tier Policy | `/dashboard` Overview |
| **2. System Provenance** | 25 sec | Frozen ML core, live health provenance | Dashboard Provenance Card |
| **3. Live Decision Flow** | 25 sec | Applicant input → Calibrated probability | `/predict` or `/analyze` |
| **4. Explainability** | 25 sec | Exact TreeSHAP waterfall & risk drivers | Decision Explainability Tab |
| **5. What-If & History** | 25 sec | Real-time simulation & audit trail | Simulator & `/history` Table |

---

## Step-by-Step Demo Sequence

### Step 1: Core Problem & System Provenance (0:00 – 0:25)
* **Goal**: Establish technical credibility and present the frozen v3.1 architecture.
* **Navigation**: Navigate to `http://localhost:5173/` (`Dashboard`).
* **Talking Points**:
  > *"Traditional mortgage risk models suffer from two critical flaws: uncalibrated probability outputs and arbitrary 0.50 cutoffs that ignore asymmetric default loss. Mortgage AI v3.1 solves this with an out-of-fold probability-calibrated LightGBM model combined with a cost-sensitive 3-tier economic decision policy."*
* **Visual Focus**:
  - Point to the **System Provenance** card in the Dashboard sidebar.
  - Highlight the active metadata:
    - **Model**: `LightGBM v3.1 (Canonical Champion)`
    - **Calibration**: `oof-iso-v3.1` (Weighted ECE: `0.0012`, Brier: `0.0492`)
    - **Policy**: `v3.1-policy-v1` ($p \le 0.045$ Approve, $p \ge 0.335$ Reject)
    - **Database**: SQLite Connected (Immutable audit trail)

---

### Step 2: Live Applicant Inference & 3-Tier Decision (0:25 – 0:50)
* **Goal**: Show calibrated probability generation and automatic policy routing.
* **Navigation**: Click **Predict Risk** on the sidebar (`/predict`).
* **Applicant Data to Input**:
  - **Credit Score**: `670`
  - **Monthly Income**: `$6,250`
  - **Loan Amount**: `$180,000`
  - **Loan Term**: `30 years`
  - **Interest Rate**: `6.8%`
* **Action**: Click **"Run Analysis"**.
* **Talking Points**:
  > *"The input flows through the 15-feature transformation pipeline. Raw tree margin outputs are mapped via our monotonic Isotonic Calibrator to yield a true empirical default probability of 24.9%. Because this falls squarely in our 4.5% to 33.5% policy review band, the system automatically routes this application to Human Underwriter Triage rather than an erroneous auto-rejection."*
* **Visual Focus**:
  - Amber **MANUAL REVIEW** decision badge.
  - Gauge chart displaying **24.9% Calibrated Default Risk**.
  - Manual Review triage notification banner.

---

### Step 3: TreeSHAP Waterfall & Plain-English Explainability (0:50 – 1:15)
* **Goal**: Demonstrate mathematical transparency and regulatory-compliant feature attribution.
* **Navigation**: Switch to the **Waterfall Chart** and **Plain-English Reasons** tabs on the Decision Explainer.
* **Talking Points**:
  > *"Every decision is backed by exact TreeSHAP attribution with zero heuristic approximations. Here, the waterfall chart starts at the global base default rate of 14.9%. We can see precisely how positive factors like solid income stability pull the risk down, while credit utilization and debt-to-income push the risk up. Mathematical additivity is verified at machine precision ($1.24 \times 10^{-14}$ error)."*
* **Visual Focus**:
  - Green/Red directional bars indicating feature contributions.
  - Plain-English breakdown explaining the primary risk drivers.

---

### Step 4: Interactive What-If Scenario Simulator (1:15 – 1:40)
* **Goal**: Show borrower actionable recourse and interactive counterfactual simulation.
* **Navigation**: Click the **What-If Simulator** tab.
* **Action**:
  - Drag **Credit Score** slider from `670` $\to$ `730` (+60 pts).
  - Drag **Credit Utilization** slider from `35%` $\to$ `15%`.
  - Click **"Simulate"**.
* **Talking Points**:
  > *"Lenders and applicants need actionable recourse. In our What-If simulator, when the applicant improves their credit score to 730 and lowers utilization, the calibrated default risk drops from 24.9% down to 1.6%. The policy decision instantly transitions to AUTO-APPROVE."*
* **Visual Focus**:
  - Green delta badge: *"Default Risk improved from 24.9% to 1.6%"*.
  - Updated decision badge: **APPROVE**.

---

### Step 5: Audit Trail, Fairness & Observability (1:40 – 2:00)
* **Goal**: Demonstrate enterprise auditability and fair lending compliance.
* **Navigation**: Click **History** on the sidebar (`/history`).
* **Talking Points**:
  > *"Every inference request, probability score, SHAP vector, and economic decision is recorded in our append-only SQLite database with full timestamps and demographic metadata. The audit trail allows 3-tier filtering, batch trend monitoring, and one-click CSV export. System metrics are exposed live via Prometheus at `/metrics`."*
* **Visual Focus**:
  - History table with `✓ Approve`, `~ Manual Review`, and `✗ Reject` rows.
  - Click **Export CSV** to demonstrate audit export.

---

## 3 Representative Applicant Profiles for Quick Testing

| Profile Name | Inputs | Expected Output | Policy State |
| :--- | :--- | :--- | :--- |
| **1. Prime Borrower** | CS: `810`, Income: `$10k/mo`, Loan: `$100k`, Term: `30y`, Rate: `5.5%` | $p_{\text{cal}} = 0.70\%$ | **APPROVE** ($p \le 0.045$) |
| **2. Triage / Borderline** | CS: `670`, Income: `$6.25k/mo`, Loan: `$180k`, Term: `30y`, Rate: `6.8%` | $p_{\text{cal}} = 24.87\%$ | **MANUAL REVIEW** ($0.045 < p < 0.335$) |
| **3. Subprime High-Risk** | CS: `500`, Income: `$2k/mo`, Loan: `$250k`, Term: `30y`, Rate: `9.5%` | $p_{\text{cal}} = 66.67\%$ | **REJECT** ($p \ge 0.335$) |

---

## Verification Commands for Technical Evaluators

```bash
# 1. Run complete test suite (89 unit/integration tests)
pytest -v

# 2. Run inference smoke test
python -m ml.inference.smoke_test

# 3. Build frontend bundle
cd frontend && npm run build
```
