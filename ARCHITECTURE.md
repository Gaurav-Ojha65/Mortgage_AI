# Mortgage AI - Production Architecture

## System Overview

Mortgage AI is a production-grade machine learning system for automated mortgage loan approval, featuring explainable AI, drift detection, and enterprise-grade observability.

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Web Browser │  │ Mobile App  │  │  API Client │  │   Admin UI  │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
└─────────┼────────────────┼────────────────┼────────────────┼───────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           LOAD BALANCER                                     │
│                           (Nginx / ALB)                                     │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ REACT FRONTEND  │    │  FASTAPI         │    │  MLFLOW         │
│                 │    │  BACKEND         │    │  TRACKING       │
│ - Loan Form     │    │                  │    │                 │
│ - Simulator     │    │ - Auth (JWT)     │    │ - Experiments   │
│ - SHAP Charts   │    │ - Prediction     │    │ - Model Registry│
│ - Live Feed     │    │ - Batch API      │    │ - Artifacts     │
│ - Analytics     │    │ - WebSocket      │    │                 │
└─────────────────┘    └─────────┬────────┘    └─────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌──────────────┐      ┌──────────────────┐     ┌──────────────────┐
│    REDIS     │      │   POSTGRESQL     │     │   PROMETHEUS     │
│              │      │                  │     │                  │
│ - Caching    │      │ - Decisions      │     │ - Metrics        │
│ - Rate Limit │      │ - Audit Logs     │     │ - Alerts         │
│ - Pub/Sub    │      │ - Model Versions │     │ - Scraping       │
└──────────────┘      └──────────────────┘     └─────────┬────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │     GRAFANA    │
                                              │                │
                                              │ - Dashboards   │
                                              │ - Alerting     │
                                              └──────────────────┘
```

## Component Details

### 1. ML Ensemble Model (`ml/ensemble.py`)

**Architecture:**
- Base Models: XGBoost, LightGBM, Neural Network
- Meta-Learner: Logistic Regression
- Technique: Stacking with cross-validation

**Key Features:**
- SMOTE for class imbalance (default ratio: 0.5)
- SHAP explainability via TreeExplainer
- Individual model predictions for debugging
- Model persistence with joblib

**Training Pipeline:**
1. Feature scaling (RobustScaler)
2. SMOTE oversampling
3. Cross-validated meta-feature generation
4. Meta-learner training
5. SHAP explainer initialization

### 2. Drift Detection (`ml/drift_detector.py`)

**Two-Layer Detection:**

1. **Data Drift:**
   - PSI (Population Stability Index)
   - Kolmogorov-Smirnov test
   - Wasserstein distance
   - Mean difference (sigma)

2. **Model Drift:**
   - Performance degradation tracking
   - Prediction distribution drift (JS divergence)
   - AUC monitoring

**Alerting:**
- Configurable threshold (default: 3 consecutive drifts)
- Automatic recommendations
- Historical drift tracking

### 3. FastAPI Backend (`api/main.py`)

**Features:**
- JWT authentication with role-based access
- Redis caching (TTL: 5 minutes)
- Rate limiting (100 req/min per user)
- Structured JSON logging
- Prometheus metrics
- WebSocket live feed
- Batch prediction endpoint

**Security:**
- OAuth2 with password flow
- Role enforcement (admin/analyst/auditor)
- Audit logging for all predictions
- IP tracking
- Request ID correlation

**Endpoints:**
```
POST   /auth/login           # Authentication
POST   /predict              # Single prediction (analyst+)
POST   /predict/batch        # Batch prediction (analyst+)
POST   /explain              # SHAP explanation
GET    /audit/logs           # Audit trail (auditor only)
GET    /health               # Health check
GET    /metrics              # Prometheus metrics
WS     /ws/live              # Real-time decision feed
```

### 4. React Frontend

**Components:**

1. **DecisionExplainer** - SHAP waterfall chart with:
   - Interactive feature breakdown
   - What-if simulator
   - Plain English explanations

2. **LoanSimulator** - Real-time probability:
   - Sliding controls for all features
   - Live probability updates
   - Visual gauge

3. **LiveDecisionFeed** - WebSocket stream:
   - Real-time decision updates
   - Statistics dashboard
   - Connection status indicator

### 5. Database Schema

**Tables:**

1. **decisions** - Store all predictions
2. **audit_logs** - Immutable audit trail
3. **model_versions** - Model registry
4. **drift_logs** - Drift detection history
5. **rate_limit_logs** - Rate limiting tracking

**Indexes:**
- Timestamp indexes for time-series queries
- User ID indexes for user analytics
- Action indexes for audit filtering

### 6. Monitoring Stack

**Prometheus Metrics:**
- `http_requests_total` - Request counting
- `http_request_duration_seconds` - Latency histogram
- `model_predictions_total` - Prediction tracking
- `websocket_active_connections` - Connection gauge
- `cache_hits_total` / `cache_misses_total` - Cache performance

**Grafana Dashboards:**
- API Performance (RPS, latency, errors)
- Model Metrics (predictions, drift)
- Infrastructure (CPU, memory, disk)
- Business (approval rates, decision trends)

## Data Flow

### Single Prediction Flow:

```
1. Client → POST /predict
2. Auth middleware validates JWT
3. Rate limiter checks Redis
4. Cache lookup (skip if miss)
5. Feature engineering
6. Ensemble model prediction
7. SHAP explanation generation
8. Database persistence
9. WebSocket broadcast
10. Audit logging
11. Response to client
```

### Batch Prediction Flow:

```
1. Client → POST /predict/batch
2. Input validation
3. Parallel prediction processing
4. Results aggregation
5. Database persistence
6. Response with results array
```

## Deployment

### Docker Services:

| Service | Image | Memory | CPU |
|---------|-------|--------|-----|
| Backend | Custom | 2GB | 1.0 |
| Frontend | Nginx | 256MB | 0.25 |
| Redis | Redis:7 | 512MB | 0.25 |
| PostgreSQL | Postgres:15 | 1GB | 0.5 |
| MLflow | Python:3.11 | 512MB | 0.5 |
| Prometheus | Prometheus | 512MB | 0.25 |
| Grafana | Grafana | 256MB | 0.25 |

### Resource Requirements:
- Minimum: 8GB RAM, 4 vCPU, 50GB disk
- Recommended: 16GB RAM, 8 vCPU, 100GB SSD

## Scaling Strategy

### Horizontal Scaling:
1. Backend: Stateless, scales behind load balancer
2. Redis: Use Redis Cluster for session/cache
3. PostgreSQL: Read replicas for queries

### Vertical Scaling:
1. ML model: GPU for inference (optional)
2. Database: Increase connection pool
3. Caching: Increase Redis memory

## Security Considerations

1. **Authentication:**
   - JWT with short expiry (30 min)
   - Refresh tokens (7 days)
   - Password hashing with bcrypt

2. **Authorization:**
   - Role-based access control
   - Endpoint-level permissions
   - API key option for service accounts

3. **Data Protection:**
   - Encryption at rest (PostgreSQL)
   - TLS in transit
   - PII handling compliance

4. **Monitoring:**
   - Audit logs for all access
   - Failed login tracking
   - Unusual activity alerts

## Performance Targets

| Metric | Target |
|--------|--------|
| Single Prediction | < 100ms |
| Batch (100) | < 2s |
| SHAP Explanation | < 500ms |
| API Availability | 99.9% |
| Concurrent Users | 1000+ |

## Disaster Recovery

1. **Database:**
   - Daily backups to S3
   - Point-in-time recovery
   - Cross-region replication

2. **Models:**
   - Versioned in MLflow
   - Multi-region artifact storage
   - Rollback capability

3. **Application:**
   - Blue-green deployment
   - Health checks with auto-failover
   - Circuit breakers

## Development Workflow

1. **Local:** `docker-compose up`
2. **Testing:** `pytest` + `locust`
3. **CI/CD:** GitHub Actions
4. **Staging:** Auto-deploy from develop
5. **Production:** Manual approval required

## Future Enhancements

1. **ML:**
   - Online learning
   - A/B testing framework
   - Feature store

2. **Infrastructure:**
   - Kubernetes deployment
   - Multi-region support
   - Serverless inference

3. **Features:**
   - Document upload (OCR)
   - Alternative data sources
   - Mobile app
