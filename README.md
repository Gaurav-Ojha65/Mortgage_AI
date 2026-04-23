# Mortgage AI Decision System

Production-grade mortgage loan approval system with ML ensemble, OCR document extraction, fairness auditing, and regulatory compliance. Built with FastAPI, React, XGBoost/LightGBM, and PostgreSQL.

---

## Features

### Core Capabilities
- **ML Ensemble**: XGBoost + LightGBM with SHAP explainability
- **Document OCR**: Extract income/employment data from pay stubs, bank statements, tax returns
- **Fairness Audit**: Bias detection across age, income, home ownership (ECOA/Fair Housing compliance)
- **Real-time Decisions**: WebSocket live feed, Redis caching, rate limiting
- **Production Ready**: Docker Compose, Prometheus metrics, Grafana dashboards

---

## Tech Stack

| Component | Technology |
|---|---|
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | React + IBM Plex Mono |
| **ML Models** | XGBoost, LightGBM, SHAP |
| **Database** | PostgreSQL (SQLite fallback) |
| **Cache** | Redis |
| **OCR** | pdfplumber, pytesseract, Tesseract |
| **Fairness** | fairlearn |
| **Monitoring** | Prometheus + Grafana |
| **Container** | Docker Compose |

---

## Installation

### Quick Start (Docker)
```bash
docker-compose up -d
```

### Manual Installation
```bash
cd Mortgage_AI

# Install all dependencies
pip install -r requirements.txt

# For OCR functionality (Linux/Mac)
sudo apt-get install tesseract-ocr poppler-utils  # Linux
brew install tesseract poppler                    # Mac

# For Kaggle dataset download
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## How to Run

### 1. Start the API server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API runs on **http://localhost:8000**  
Interactive docs: http://localhost:8000/docs

### 2. Start the Frontend
```bash
cd mortgage-frontend
npm install
npm start
```
Frontend runs on **http://localhost:3000**

### 3. Run Data Pipeline (first time setup)
```bash
python data_pipeline.py
```

### 4. Run Fairness Audit
```bash
python fairness_audit.py --model models/xgboost_model.joblib --data data/test.csv
```

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
``` -->


# 🚀 Mortgage AI Decision System

AI-powered mortgage loan approval system using **Machine Learning + Monte Carlo Simulation** for intelligent risk assessment and decision-making.

---

## 🔥 Features

* 🤖 ML-based loan approval (XGBoost)
* 📊 Monte Carlo simulation for risk analysis
* ⚡ FastAPI backend (high-performance APIs)
* 🌐 React frontend dashboard (modern UI)
* 🧠 AI advisor integration (Ollama)
* 📈 Real-time analytics & visualization
* 🗄️ SQLite database for decision tracking

---

## 🧱 Tech Stack

| Component         | Technology                  |
| ----------------- | --------------------------- |
| **Frontend**      | React + Tailwind + Plotly   |
| **Backend API**   | FastAPI + Uvicorn           |
| **ML Model**      | XGBoost                     |
| **Database**      | SQLite                      |
| **Validation**    | Pydantic                    |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Language**      | Python 3.10+                |

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/Mortgage_AI.git
cd Mortgage_AI
```

### Backend setup

```bash
pip install fastapi uvicorn sqlalchemy pydantic
pip install scikit-learn xgboost joblib
pip install matplotlib seaborn numpy
```

### Frontend setup

```bash
cd mortgage-frontend
npm install
```

---

## ▶️ How to Run

### 1️⃣ Start Backend API

```bash
python api.py
```

📍 Runs on:
http://localhost:8001

📘 API Docs:
http://localhost:8001/docs

---

### 2️⃣ Start Frontend (React)

```bash
cd mortgage-frontend
npm start
```

🌐 Runs on:
http://localhost:3000

---

## 🔌 API Endpoints

| Method | Endpoint   | Description              |
| ------ | ---------- | ------------------------ |
| `POST` | `/analyze` | Analyze loan application |
| `GET`  | `/history` | Fetch recent decisions   |
| `GET`  | `/compare` | Compare loan scenarios   |
| `GET`  | `/health`  | API health check         |

---

## 📥 Example Request

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

---

## 📤 Example Response

```json
{
  "decision": "APPROVE",
  "emi": 4103.31,
  "risk_level": "LOW",
  "default_probability": 0.08,
  "approval_probability": 0.82,
  "advice": "Application meets criteria",
  "monte_carlo": {
    "worst_case_emi": 4320.15,
    "safe_income_threshold": 42500.0
  }
}
```

---

## 📁 Project Structure

```
Mortgage_AI/
│
├── api.py                  # FastAPI backend (port 8001)
├── emi.py
├── features.py
├── model.py
├── risk.py
├── advisor.py
├── monte_carlo.py
├── evaluate.py
├── best_model.pkl
├── mortgage.db
│
├── mortgage-frontend/      # React frontend (port 3000)
│
└── design-system/
```

---

## 📊 Model Evaluation

```bash
python evaluate.py
```

Generates:

* confusion matrix
* ROC curve
* precision-recall curve
* feature importance

---

## 🗄️ Database Schema

```sql
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    income REAL,
    loan_amount REAL,
    credit_score INTEGER,
    decision TEXT,
    risk_level TEXT,
    default_probability REAL,
    emi REAL,
    advice TEXT
);
```

---

## ⚠️ Notes

* Old Dash dashboard (port 8050) is deprecated
* Use React frontend (port 3000)
* Ensure backend is running before frontend

---

## 💡 Future Improvements

* 🔐 Authentication system
* ☁️ Cloud deployment (AWS/Vercel)
* 📊 Advanced analytics dashboard
* 🤖 Better AI advisor integration

---

## 👨‍💻 Author

Gaurav Ojha
Engineering Student | AI/ML Developer

---

## ⭐ If you like this project, give it a star!
