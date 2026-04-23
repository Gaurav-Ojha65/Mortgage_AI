# Mortgage AI - Production Deployment Guide

## Overview

Production-grade mortgage loan approval system with ML ensemble (XGBoost + LightGBM + Neural Network), SHAP explainability, drift detection, and full observability.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENTS                               │
│  (Web Browser / Mobile / API Consumers)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    NGINX (Port 80)                           │
│              Static Files / Reverse Proxy                  │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴──────────┐
         ▼                      ▼
┌─────────────────┐  ┌─────────────────────────┐
│  REACT FRONTEND │  │   FASTAPI BACKEND       │
│   (Port 3000)   │  │     (Port 8000)         │
└─────────────────┘  └───────────┬─────────────┘
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       ▼                         ▼                         ▼
┌──────────────┐      ┌──────────────────┐      ┌────────────────┐
│    REDIS     │      │  POSTGRESQL      │      │    MLFLOW      │
│  (Caching)   │      │   (Database)     │      │ (Model Registry)│
│   Port 6379  │      │   Port 5432      │      │   Port 5000    │
└──────────────┘      └──────────────────┘      └────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              PROMETHEUS + GRAFANA                            │
│              (Monitoring & Observability)                   │
│              Ports: 9090, 3000                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Git
- 8GB+ RAM
- 50GB+ disk space

### Production Deployment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Mortgage_AI.git
cd Mortgage_AI

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

### Services Overview

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 80 | React app served by Nginx |
| Backend API | 8000 | FastAPI with ML endpoints |
| Redis | 6379 | Caching & rate limiting |
| PostgreSQL | 5432 | Primary database |
| MLflow | 5000 | Model tracking & registry |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards & alerting |

## API Endpoints

### Authentication
```bash
# Login
POST /auth/login
{
  "username": "admin",
  "password": "admin123"
}
```

### Prediction
```bash
# Single prediction (requires auth)
POST /predict
Authorization: Bearer <token>
{
  "income": 60000,
  "loan_amount": 25000,
  "credit_score": 650,
  "interest_rate": 8.5,
  "loan_term": 5,
  "existing_loans": 0
}

# Batch prediction
POST /predict/batch
Authorization: Bearer <token>
{
  "applications": [...]
}
```

### SHAP Explainability
```bash
# Get SHAP explanation
POST /explain
Authorization: Bearer <token>
{
  "income": 60000,
  "loan_amount": 25000,
  ...
}
```

### Health & Metrics
```bash
# Health check
GET /health

# Prometheus metrics
GET /metrics

# Audit logs (auditor only)
GET /audit/logs
```

## Testing

### Unit Tests
```bash
pytest tests/ -v --cov=ml --cov=api
```

### Load Testing
```bash
# Start Locust
locust -f tests/load_test.py --host http://localhost:8000

# Or headless mode
locust -f tests/load_test.py --headless -u 100 -r 10 --run-time 5m
```

## Monitoring

### Prometheus Metrics
- `http_requests_total` - Request count
- `http_request_duration_seconds` - Response time
- `model_predictions_total` - Prediction count
- `websocket_active_connections` - Active WebSocket connections

### Grafana Dashboards
- Application Performance
- ML Model Metrics
- API Usage
- Database Performance

Access: http://localhost:3000 (admin/admin)

## MLflow Model Registry

Access: http://localhost:5000

Track experiments, register models, manage versions.

## Security

### Authentication
- JWT tokens with refresh
- Role-based access (admin/analyst/auditor)

### Rate Limiting
- 100 requests/minute per user
- Configured via Redis

### Audit Logging
- All predictions logged
- Decision explanations stored
- Immutable audit trail

## Drift Detection

Automatic monitoring for:
- Data drift (PSI, KS test)
- Model drift (performance degradation)
- Alert threshold: 3 consecutive drift detections

## Scaling

### Horizontal Scaling
```bash
# Scale backend instances
docker-compose up -d --scale backend=3

# Add load balancer in front
```

### Database Scaling
- Read replicas for analytics
- Connection pooling (PgBouncer)

## Troubleshooting

### Common Issues

**Backend won't start**
```bash
docker-compose logs backend
# Check: Redis and PostgreSQL are running
docker-compose up -d redis db
```

**Out of memory**
```bash
# Increase Docker memory limit
# Or reduce batch sizes
```

**Database connection errors**
```bash
# Wait for database to be ready
docker-compose exec db pg_isready -U postgres
```

### Health Check
```bash
# Check all services
docker-compose ps

# API health
curl http://localhost:8000/health

# Database health
docker-compose exec db pg_isready -U postgres
```

## CI/CD

GitHub Actions workflow:
- Automated testing on PR
- Security scanning (Trivy)
- Docker image build
- Deploy to staging/production

See `.github/workflows/ci-cd.yml`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_HOST` | Redis server | redis |
| `DATABASE_URL` | PostgreSQL connection | postgresql://... |
| `MLFLOW_TRACKING_URI` | MLflow server | http://mlflow:5000 |
| `SECRET_KEY` | JWT secret | (required) |
| `ENVIRONMENT` | dev/staging/production | production |

### Production Configuration

```bash
# Create .env file
cat > .env << EOF
SECRET_KEY=your-super-secret-key-change-this
ENVIRONMENT=production
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://yourdomain.com
EOF

# Deploy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Backup & Recovery

### Database Backup
```bash
# Automated daily backups
docker-compose exec db pg_dump -U postgres mortgage > backup.sql
```

### Model Backup
```bash
# MLflow artifacts
# Models stored in /mlflow volume
```

## Support

- Issues: https://github.com/YOUR_USERNAME/Mortgage_AI/issues
- Documentation: See `docs/` folder

## License

MIT License - See LICENSE file
