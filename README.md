# Mortgage AI Decision System

A full-stack mortgage risk assessment project combining machine learning, explainability, fairness auditing, document processing, a FastAPI backend, React UI, containerization, testing, and observability.

> **Portfolio note:** This repository contains multiple modeling paths. The simpler `model.py` baseline uses Logistic Regression, Random Forest, and XGBoost on a synthetic dataset, while `ml/ensemble.py` contains the XGBoost + LightGBM + neural-network stacking implementation used for the advanced ensemble workflow. Performance claims should be taken from reproducible evaluation artifacts rather than README marketing numbers.

## Why this project?

Mortgage decisions combine predictive modeling with explainability, fairness, and operational constraints. The project explores how a model can be exposed through an API and dashboard while keeping evaluation, testing, monitoring, and deployment concerns visible in the repository.

## Architecture

```text
                     ┌──────────────────────┐
                     │      React UI        │
                     │ Loan / Risk Dashboard│
                     └──────────┬───────────┘
                                │ HTTP / WS
                                ▼
                     ┌──────────────────────┐
                     │     FastAPI API      │
                     │ Auth · Validation    │
                     │ Risk · History       │
                     └──────┬───────┬───────┘
                            │       │
                 ┌──────────┘       └─────────────┐
                 ▼                                ▼
        ┌────────────────┐                ┌────────────────┐
        │ PostgreSQL     │                │ Redis          │
        │ decisions/data │                │ cache/limits   │
        └────────────────┘                └────────────────┘

        ┌──────────────────────────────────────────────┐
        │ ML pipeline                                  │
        │ XGBoost · LightGBM · NN · SHAP · Fairlearn  │
        └──────────────────────────────────────────────┘
                 │
        ┌────────┴─────────┐
        ▼                  ▼
   Docker Compose     Prometheus / Grafana

Document path:
PDF / image → OCR → extracted application fields → risk workflow
```

## Engineering highlights

- **ML:** XGBoost, LightGBM, neural-network ensemble, SHAP explainability, SMOTE and model evaluation utilities.
- **Fairness:** Fairlearn-based auditing and explicit removal of `CODE_GENDER` from the fairness-aware credit-risk implementation, with proxy-bias checks in the code.
- **Backend:** FastAPI, validation, authentication, WebSockets, Redis caching/rate limiting, PostgreSQL/SQLAlchemy.
- **Document processing:** PDF/image extraction with `pdfplumber`, `pytesseract`, `pdf2image`, and Pillow.
- **Observability:** Prometheus metrics, Grafana dashboards, and drift-detection tests.
- **Delivery:** Docker Compose, separate backend/frontend images, GitHub Actions CI, linting, type checking, tests, coverage, load testing, and vulnerability scanning.

## Tech stack

| Layer | Technologies |
|---|---|
| ML | XGBoost, LightGBM, TensorFlow, scikit-learn, SHAP, Fairlearn, imbalanced-learn |
| Backend | Python, FastAPI, Uvicorn, Pydantic, SQLAlchemy |
| Frontend | React |
| Data | PostgreSQL, SQLite, Redis |
| Document processing | pdfplumber, pytesseract, pdf2image, Pillow |
| MLOps / observability | MLflow, Prometheus, Grafana, Evidently |
| Infrastructure | Docker, Docker Compose, GitHub Actions |
| Quality | pytest, pytest-asyncio, pytest-cov, flake8, Black, mypy, Locust |

## Repository structure

```text
Mortgage_AI/
├── api/                         # FastAPI application modules
├── ml/                          # Ensemble / model implementations
├── tests/                       # Unit, drift and load tests
├── mortgage-frontend/           # React dashboard
├── data/                        # Data/preprocessing artifacts
├── docker/                      # Backend/frontend container definitions
├── monitoring/                  # Prometheus/Grafana configuration
├── .github/workflows/           # CI/CD workflow
├── model.py                     # Baseline model comparison workflow
├── credit_risk_fair.py          # Fairness-aware credit-risk workflow
├── data_pipeline.py             # Data preparation pipeline
├── docker-compose.yml           # Local multi-service stack
├── evaluate.py                  # Evaluation and model-card generation
└── README.md
```

## Run locally

### Docker

```bash
docker compose up --build
```

The compose stack defines backend, frontend, PostgreSQL, Redis, MLflow, Prometheus, and Grafana services.

### Backend

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

FastAPI documentation is available at `/docs` while the API is running.

### Frontend

```bash
cd mortgage-frontend
npm install
npm start
```

### Tests

```bash
pytest tests/ -v
```

The repository includes ensemble tests covering initialization, fitting, prediction, probabilities, evaluation, SHAP explanations, persistence, individual model predictions, and edge cases.

## CI/CD

The GitHub Actions workflow runs a Python matrix, dependency installation, lint checks, Black formatting checks, mypy, pytest with coverage, Locust load tests, and a Trivy filesystem security scan. It also contains Docker image build/push and deployment stages.

## Model evaluation

`evaluate.py` generates confusion-matrix, ROC, precision-recall, and feature-importance visualizations and performs 5-fold cross-validation.

The baseline data generator intentionally uses **synthetic loan data**, so its results should not be interpreted as real-world underwriting performance. The evaluation code documents limitations including synthetic data, missing macroeconomic factors, limited application features, and static approval thresholds.

## Important engineering caveats

- This is a portfolio/learning system, **not a production underwriting authority**.
- The repository contains historical and advanced implementations; the current code should be treated as the source of truth when describing the project.
- Do not commit real credentials, API keys, or production borrower data.
- Real lending decisions require appropriate validation, governance, legal/compliance review, and human oversight.

## Demo

Live application: https://mortgage-ai-dyb5.vercel.app

## Author

**Gaurav Ojha**  
Computer Science student focused on software engineering, backend systems, cloud infrastructure, and practical AI/ML.

[GitHub](https://github.com/Gaurav-Ojha65) · [Portfolio](https://gaurav-ojha-portfolio.netlify.app) · [LinkedIn](https://www.linkedin.com/in/gaurav-ojha18/)
