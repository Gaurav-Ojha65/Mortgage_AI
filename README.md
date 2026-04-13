# Mortgage AI Decision System

AI-powered mortgage loan approval system with FastAPI backend, Dash dashboard, and XGBoost ML model. Uses Monte Carlo simulation for risk assessment and provides real-time loan decision analysis.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Backend API** | FastAPI + Uvicorn |
| **Frontend Dashboard** | Dash + Plotly |
| **ML Model** | XGBoost |
| **Database** | SQLite |
| **Validation** | Pydantic |
| **Visualizations** | Matplotlib, Seaborn, Plotly |
| **Language** | Python 3.10+ |

---

## Installation

```bash
# Clone or navigate to project directory
cd Mortgage_AI

# Install backend dependencies
pip install fastapi uvicorn sqlalchemy pydantic

# Install ML dependencies
pip install scikit-learn xgboost joblib

# Install dashboard dependencies
pip install dash plotly dash-bootstrap-components requests pandas

# Install evaluation dependencies
pip install matplotlib seaborn numpy
```

---

## How to Run

### 1. Start the API server

```bash
python api.py
```

API runs on **http://localhost:8001**

Interactive docs: http://localhost:8001/docs

### 2. Start the Dashboard (in a new terminal)

```bash
python dashboard.py
```

Dashboard runs on **http://localhost:8050**

---

## API Endpoints

| Method | Endpoint | Description | Request/Params |
|---|---|---|---|
| `POST` | `/analyze` | Analyze loan application | JSON body with income, loan_amount, interest_rate, loan_term, credit_score, existing_loans |
| `GET` | `/history` | Get last 20 decisions | Query: `?limit=N` (max 100) |
| `GET` | `/compare` | Compare loan scenarios | Query: `?income=X&loan_amount=Y&credit_score=Z` |
| `GET` | `/health` | Health check | None |

### Example Request

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "income": 50000,
    "loan_amount": 200000,
    "interest_rate": 8.5,
    "loan_term": 5,
    "credit_score": 650,
    "existing_loans": 1
  }'
```

### Example Response

```json
{
  "decision": "APPROVE",
  "emi": 4103.31,
  "risk_level": "LOW",
  "default_probability": 0.08,
  "approval_probability": 0.82,
  "advice": "Application meets criteria - proceed with application",
  "feature_values": {
    "debt_to_income_ratio": 0.1333,
    "emi_to_income_ratio": 8.21,
    "credit_utilization_score": 0.6364,
    "loan_burden_index": 0.2833,
    "affordability_score": 0.9179
  },
  "monte_carlo": {
    "worst_case_emi": 4320.15,
    "safe_income_threshold": 42500.0,
    "scenario_breakdown": {"stable": 7400, "stressed": 1800, "crisis": 800}
  }
}
```

---

## Project Structure

```
Mortgage_AI/
├── api.py              # FastAPI REST API (port 8001)
├── dashboard.py         # Dash dashboard UI (port 8050)
├── emi.py              # EMI calculator
├── features.py         # Feature engineering for ML
├── model.py            # XGBoost model training & prediction
├── risk.py             # Risk level assessment
├── advisor.py          # AI mortgage advisor (Ollama)
├── monte_carlo.py      # Monte Carlo risk simulation
├── evaluate.py         # Model evaluation & visualizations
├── best_model.pkl      # Trained XGBoost model
├── mortgage.db         # SQLite database
├── CLAUDE.md           # Project documentation
│
├── design-system/      # Frontend design assets
└── mortgage-frontend/  # React frontend (separate)
```

---

## Model Evaluation

```bash
python evaluate.py
```

Generates:
- `confusion_matrix.png` — Prediction accuracy heatmap
- `roc_curve.png` — ROC curve with AUC score
- `precision_recall_curve.png` — Precision-Recall trade-off
- `feature_importance.png` — Top feature contributions
- `model_metrics.json` — Full metrics JSON

---

## Screenshots

> Add screenshots of the dashboard here:
> - Dashboard input panel with sliders
> - Analysis results with decision badge
> - Monte Carlo 3D risk visualization
> - Sensitivity analysis chart
> - History table of past decisions

---

## Database Schema

```sql
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    income REAL NOT NULL,
    loan_amount REAL NOT NULL,
    credit_score INTEGER NOT NULL,
    decision TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    default_probability REAL,
    emi REAL NOT NULL,
    advice TEXT
);
```