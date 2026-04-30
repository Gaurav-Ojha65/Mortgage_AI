"""
Production FastAPI Application for Mortgage AI
Features: Redis caching, rate limiting, auth, WebSocket, Prometheus metrics
"""

import os
import json
import time
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from api.auth import (
    Token, authenticate_user, create_access_token, create_refresh_token,
    get_current_user, get_current_active_user, require_admin, require_analyst,
    require_auditor, audit_logger, User
)

# Import routers
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_router import router as model_router
from document_router import router as document_router
from fairness_router import router as fairness_router

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
ACTIVE_CONNECTIONS = Gauge('websocket_active_connections', 'Number of active WebSocket connections')
PREDICTION_COUNT = Counter('model_predictions_total', 'Total predictions', ['result'])
CACHE_HIT = Counter('cache_hits_total', 'Cache hits')
CACHE_MISS = Counter('cache_misses_total', 'Cache misses')

# Redis connection
redis_client: Optional[redis.Redis] = None

# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        ACTIVE_CONNECTIONS.set(len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        ACTIVE_CONNECTIONS.set(len(self.active_connections))

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


# Pydantic models
class LoanApplication(BaseModel):
    income: float = Field(..., gt=0, description="Annual income")
    loan_amount: float = Field(..., gt=0, description="Loan amount requested")
    interest_rate: float = Field(..., ge=0, le=100, description="Interest rate %")
    loan_term: int = Field(..., ge=1, le=30, description="Loan term in years")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    existing_loans: int = Field(default=0, ge=0, description="Number of existing loans")
    employment_type: str = Field(default="salaried", description="Employment type")
    dti_ratio: Optional[float] = Field(default=None, ge=0, le=100, description="Debt-to-income ratio")


class PredictionResponse(BaseModel):
    decision: str
    approval_probability: float
    default_probability: float
    risk_level: str
    emi: float
    confidence_score: float
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    applications: List[LoanApplication]


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    processed_count: int
    failed_count: int


class AuditLogQuery(BaseModel):
    user: Optional[str] = None
    action: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = 100


# Cache utilities
async def get_cached_prediction(key: str) -> Optional[dict]:
    """Get cached prediction from Redis."""
    if redis_client is None:
        return None
    try:
        data = await redis_client.get(f"prediction:{key}")
        if data:
            CACHE_HIT.inc()
            return json.loads(data)
        CACHE_MISS.inc()
    except Exception as e:
        print(f"Redis error: {e}")
    return None


async def cache_prediction(key: str, result: dict, ttl: int = 300):
    """Cache prediction result."""
    if redis_client is None:
        return
    try:
        await redis_client.setex(
            f"prediction:{key}",
            ttl,
            json.dumps(result, default=str)
        )
    except Exception as e:
        print(f"Redis error: {e}")


async def rate_limit_check(key: str, limit: int, window: int) -> bool:
    """Check rate limit."""
    if redis_client is None:
        return True
    try:
        current = await redis_client.get(f"rate_limit:{key}")
        if current is None:
            await redis_client.setex(f"rate_limit:{key}", window, 1)
            return True
        if int(current) >= limit:
            return False
        await redis_client.incr(f"rate_limit:{key}")
        return True
    except Exception as e:
        print(f"Redis error: {e}")
        return True


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global redis_client

    # Startup
    try:
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0,
            decode_responses=True
        )
        await redis_client.ping()
        print("Connected to Redis")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        redis_client = None

    yield

    # Shutdown
    if redis_client:
        await redis_client.close()
        print("Redis connection closed")


# Create app
app = FastAPI(
    title="Mortgage AI API",
    description="Production API for mortgage loan approval with ML + explainability",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(model_router, prefix="/api")
app.include_router(document_router, prefix="/api")
app.include_router(fairness_router, prefix="/api")

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Collect request metrics."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_LATENCY.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    return response


# Authentication endpoints
@app.post("/auth/login", response_model=Token)
async def login(credentials: dict):
    """Login and get access token."""
    user = authenticate_user(
        {"admin": {"username": "admin", "hashed_password": "$2b$12$...", "role": "admin"}},
        credentials.get("username"),
        credentials.get("password")
    )
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}
    )

    audit_logger.log("login", user.username, {"ip": "unknown"})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 1800
    }


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    application: LoanApplication,
    current_user: User = Depends(get_current_active_user)
):
    """Make single loan prediction."""

    # Rate limiting
    if not await rate_limit_check(current_user.username, 100, 60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Check cache
    cache_key = f"{application.credit_score}_{application.income}_{application.loan_amount}"
    cached = await get_cached_prediction(cache_key)
    if cached:
        return PredictionResponse(**cached)

    # Mock prediction (replace with actual model)
    approval_prob = min(0.99, max(0.01, (application.credit_score - 300) / 550))
    decision = "APPROVED" if approval_prob > 0.5 else "REJECTED"
    risk_level = "LOW" if approval_prob > 0.8 else "MEDIUM" if approval_prob > 0.5 else "HIGH"

    result = {
        "decision": decision,
        "approval_probability": approval_prob,
        "default_probability": 1 - approval_prob,
        "risk_level": risk_level,
        "emi": round((application.loan_amount * 0.05) / 12, 2),
        "confidence_score": 0.95,
        "model_version": "ensemble_v2.0",
        "timestamp": datetime.utcnow().isoformat()
    }

    # Cache result
    await cache_prediction(cache_key, result)

    # Log prediction
    PREDICTION_COUNT.labels(result=decision).inc()
    audit_logger.log(
        "prediction",
        current_user.username,
        {"input": application.dict(), "output": result}
    )

    # Broadcast to WebSocket clients
    await manager.broadcast({
        "type": "prediction",
        "user": current_user.username,
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    })

    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    current_user: User = Depends(require_analyst)
):
    """Batch prediction endpoint."""

    if len(request.applications) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 applications per batch")

    results = []
    failed = 0

    for app in request.applications:
        try:
            # Mock prediction
            approval_prob = min(0.99, max(0.01, (app.credit_score - 300) / 550))
            decision = "APPROVED" if approval_prob > 0.5 else "REJECTED"

            results.append(PredictionResponse(
                decision=decision,
                approval_probability=approval_prob,
                default_probability=1 - approval_prob,
                risk_level="LOW" if approval_prob > 0.8 else "MEDIUM",
                emi=round((app.loan_amount * 0.05) / 12, 2),
                confidence_score=0.95,
                model_version="ensemble_v2.0",
                timestamp=datetime.utcnow().isoformat()
            ))
        except Exception as e:
            failed += 1
            print(f"Batch prediction failed: {e}")

    audit_logger.log(
        "batch_prediction",
        current_user.username,
        {"count": len(request.applications), "failed": failed}
    )

    return BatchPredictionResponse(
        results=results,
        processed_count=len(results),
        failed_count=failed
    )


# SHAP explainability endpoint
@app.post("/explain")
async def explain_prediction(
    application: LoanApplication,
    current_user: User = Depends(get_current_active_user)
):
    """Get SHAP explanation for prediction."""

    # Mock SHAP values (replace with actual SHAP explainer)
    shap_values = {
        "base_value": 0.5,
        "features": {
            "credit_score": 0.35,
            "income": 0.25,
            "dti_ratio": -0.20,
            "loan_amount": -0.15,
            "existing_loans": -0.05
        },
        "waterfall_data": [
            {"feature": "Base", "value": 0.5, "cumulative": 0.5},
            {"feature": "Credit Score", "value": 0.35, "cumulative": 0.85},
            {"feature": "Income", "value": 0.25, "cumulative": 1.10},
            {"feature": "DTI Ratio", "value": -0.20, "cumulative": 0.90},
            {"feature": "Loan Amount", "value": -0.15, "cumulative": 0.75},
        ],
        "force_plot_html": None  # Would contain actual SHAP force plot
    }

    audit_logger.log("explain", current_user.username, {"input": application.dict()})

    return {
        "shap_values": shap_values,
        "top_features": [
            {"name": "credit_score", "impact": 0.35, "direction": "positive"},
            {"name": "income", "impact": 0.25, "direction": "positive"},
            {"name": "dti_ratio", "impact": -0.20, "direction": "negative"},
        ],
        "summary": "Credit score and income are the strongest positive factors. High DTI and loan amount reduce approval probability."
    }


# WebSocket endpoint for live updates
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time decision feed."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_text(f"pong: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Audit log endpoints
@app.get("/audit/logs")
async def get_audit_logs(
    user: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = Query(100, le=1000),
    current_user: User = Depends(require_auditor)
):
    """Get audit logs (auditor only)."""
    logs = audit_logger.get_logs(user=user, action=action, limit=limit)
    return {"logs": logs, "total": len(logs)}


# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return StreamingResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "services": {
            "api": "up",
            "redis": "up" if redis_client else "down"
        }
    }
    return health


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
