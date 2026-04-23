# pip install fastapi uvicorn sqlalchemy pydantic
"""
Mortgage AI Decision System - FastAPI REST API
Exposes the mortgage advisor system as a REST API with database persistence.
"""

from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
import sqlite3
import logging
import json
import sys
from logging.handlers import RotatingFileHandler

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import uuid
import signal
import asyncio

from features import engineer_features
from emi import calculate_emi
from risk import calculate_risk
from monte_carlo import simulate as mc_simulate
from model_router import router as model_router
from shap_router import shap_router


# =============================================================================
# Structured Logging Setup
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging():
    """Configure structured JSON logging."""
    logger = logging.getLogger("mortgage_api")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        "mortgage_api.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


# =============================================================================
# Response Standardization
# =============================================================================

def create_response(data=None, success=True, error=None, request_id=None):
    """Create standardized API response envelope."""
    response = {
        "success": success,
        "request_id": request_id or str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat()
    }
    if data is not None:
        response["data"] = data
    if error is not None:
        response["error"] = error
    return response


# =============================================================================
# Input Sanitization Utilities
# =============================================================================

def sanitize_string(value: str) -> str:
    """Sanitize string input - strip whitespace and basic injection protection."""
    if not isinstance(value, str):
        return value
    # Strip whitespace
    value = value.strip()
    # Remove null bytes
    value = value.replace('\x00', '')
    # Basic SQL injection pattern detection (for logging, not blocking)
    suspicious = ['--', '/*', '*/', ';', 'DROP', 'SELECT', 'INSERT', 'DELETE', 'UPDATE']
    for pattern in suspicious:
        if pattern.upper() in value.upper():
            logger.warning(f"Suspicious pattern detected in input: {pattern}")
    return value


def validate_numeric_range(value, min_val, max_val, name):
    """Validate numeric value is within safe range."""
    if value is None:
        return None
    try:
        num = float(value)
        if num < min_val or num > max_val:
            raise HTTPException(
                status_code=422,
                detail=f"{name} must be between {min_val} and {max_val}"
            )
        return num
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"{name} must be a valid number"
        )


# =============================================================================
# Pydantic Models - Input Validation
# =============================================================================

class LoanApplication(BaseModel):
    """Validated loan application input."""
    income: float = Field(..., gt=0, description="Monthly income (must be positive)")
    loan_amount: float = Field(..., gt=0, description="Loan amount (must be positive)")
    interest_rate: float = Field(..., gt=0, description="Annual interest rate %")
    loan_term: int = Field(..., gt=0, description="Loan term in years")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score 300-850")
    existing_loans: int = Field(default=0, ge=0, description="Number of existing loans")

    @field_validator("income", "loan_amount", "interest_rate", "loan_term")
    @classmethod
    def reject_negative(cls, v):
        if v < 0:
            raise ValueError("Value must not be negative")
        return v

    @field_validator("credit_score")
    @classmethod
    def validate_credit_score(cls, v):
        if v < 300 or v > 850:
            raise ValueError("Credit score must be between 300 and 850")
        return v


# =============================================================================
# Database Setup
# =============================================================================

DATABASE_PATH = "mortgage.db"


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
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
        )
    """)
    conn.commit()
    conn.close()


def save_decision(data: dict):
    """Save a decision record to the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO decisions
        (timestamp, income, loan_amount, credit_score, decision, risk_level, default_probability, emi, advice)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["timestamp"],
        data["income"],
        data["loan_amount"],
        data["credit_score"],
        data["decision"],
        data["risk_level"],
        data.get("default_probability"),
        data["emi"],
        data.get("advice")
    ))
    conn.commit()
    conn.close()


def get_history(limit: int = 20) -> List[dict]:
    """Retrieve last N decisions from database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM decisions ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# =============================================================================
# Global State
# =============================================================================

global_model = None
startup_time = datetime.now()
prediction_count = 0

APP_VERSION = "1.0.0"

# Error tracking (last 100 errors)
error_log = []
MAX_ERROR_LOG = 100


def log_error(error_data: dict):
    """Store error in memory log (last 100)."""
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "error": error_data
    }
    error_log.insert(0, error_entry)
    if len(error_log) > MAX_ERROR_LOG:
        error_log.pop()


def get_uptime_seconds():
    """Get application uptime in seconds."""
    return int((datetime.now() - startup_time).total_seconds())


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return None


# =============================================================================
# Lifespan - Startup/Shutdown
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model into memory once at startup."""
    global global_model
    try:
        global_model = joblib.load("best_model.pkl")
        init_db()
        print("[Startup] Model loaded, database initialized")
    except Exception as e:
        print(f"[Startup] Warning: Could not load model - {e}")
        global_model = None
    yield
    # Cleanup on shutdown if needed


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Mortgage AI Decision API",
    description="AI-powered mortgage loan approval system with Monte Carlo risk simulation",
    version="1.0.0",
    lifespan=lifespan
)

# Request size limit middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request body size to 1MB."""
    max_size = 1 * 1024 * 1024  # 1MB
    body = await request.body()
    if len(body) > max_size:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large", "type": "payload_too_large"}
        )
    # Recreate request with body for downstream
    async def receive():
        return {"type": "http.request", "body": body}
    request._receive = receive
    return await call_next(request)


# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include model comparison router
app.include_router(model_router, prefix="/api")
app.include_router(shap_router, prefix="/api")


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint with system metrics."""
    db_ok = False
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("SELECT 1").fetchone()
        conn.close()
        db_ok = True
    except Exception:
        pass

    response = {
        "status": "ok",
        "version": APP_VERSION,
        "uptime_seconds": get_uptime_seconds(),
        "models_loaded": global_model is not None,
        "db_connected": db_ok,
        "predictions_served": prediction_count
    }

    memory_mb = get_memory_usage()
    if memory_mb:
        response["memory_usage_mb"] = round(memory_mb, 2)

    return create_response(data=response)


# =============================================================================
# Analyze Endpoint
# =============================================================================

@app.post("/analyze")
def analyze_application(application: LoanApplication):
    global prediction_count
    """
    Analyze a loan application and return decision with full metrics.

    Pipeline:
    1. engineer_features() - create derived features
    2. predict() - ML model approval probability
    3. calculate_emi() - monthly payment
    4. calculate_risk() - risk level assessment
    5. simulate() - Monte Carlo default probability
    6. Return comprehensive decision
    """
    logger.info(f"Analyzing application: income={application.income}, loan={application.loan_amount}")

    try:
        # Build input dict
        loan_data = application.model_dump()

        # Calculate EMI
        emi = calculate_emi(
            loan_data["loan_amount"],
            loan_data["interest_rate"],
            loan_data["loan_term"]
        )

        # Engineer features
        enriched = engineer_features(loan_data)
        feature_cols = [
            "income", "loan_amount", "credit_score", "existing_loans",
            "loan_term", "debt_to_income_ratio", "emi_to_income_ratio",
            "credit_utilization_score"
        ]

        # ML prediction
        if global_model is not None:
            X = pd.DataFrame([enriched])[feature_cols]
            approved = bool(global_model.predict(X)[0])
            approval_prob = float(global_model.predict_proba(X)[0][1])
        else:
            approved = None
            approval_prob = None

        # Risk assessment
        risk_level = calculate_risk(
            loan_data["income"],
            loan_data["loan_amount"],
            loan_data["credit_score"],
            loan_data["existing_loans"]
        )

        # Monte Carlo simulation
        mc_results = mc_simulate({
            "income": loan_data["income"],
            "loan_amount": loan_data["loan_amount"],
            "interest_rate": loan_data["interest_rate"],
            "loan_term": loan_data["loan_term"],
            "credit_score": loan_data["credit_score"]
        }, n_simulations=5000)

        # Build decision
        if risk_level == "HIGH" or mc_results["default_probability"] > 0.35:
            decision = "REJECT"
        elif risk_level == "LOW" and mc_results["default_probability"] < 0.15:
            decision = "APPROVE"
        else:
            decision = "CONDITIONAL"

        # Build advice
        advice = []
        if mc_results["default_probability"] > 0.15:
            advice.append(f"Default risk too high ({mc_results['default_probability']:.1%})")
        if risk_level == "HIGH":
            advice.append("Risk assessment is HIGH")
        if loan_data["credit_score"] < 650:
            advice.append("Credit score below 650 - improve before applying")
        if mc_results["mean_emi_ratio"] > 0.35:
            advice.append(f"EMI burden high ({mc_results['mean_emi_ratio']:.1%} of income)")
        if not advice:
            advice.append("Application meets criteria - proceed with application")

        # Compile response
        response = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "emi": emi,
            "risk_level": risk_level,
            "default_probability": mc_results["default_probability"],
            "approval_probability": approval_prob,
            "advice": "; ".join(advice),
            "ai_advice": False,
            "feature_values": {
                "debt_to_income_ratio": enriched["debt_to_income_ratio"],
                "emi_to_income_ratio": enriched["emi_to_income_ratio"],
                "credit_utilization_score": enriched["credit_utilization_score"],
                "loan_burden_index": enriched["loan_burden_index"],
                "affordability_score": enriched["affordability_score"]
            },
            "monte_carlo": {
                "worst_case_emi": mc_results["worst_case_emi"],
                "safe_income_threshold": mc_results["safe_income_threshold"],
                "scenario_breakdown": mc_results["scenario_breakdown"]
            }
        }

        # Save to database
        save_decision({
            "timestamp": response["timestamp"],
            "income": loan_data["income"],
            "loan_amount": loan_data["loan_amount"],
            "credit_score": loan_data["credit_score"],
            "decision": decision,
            "risk_level": risk_level,
            "default_probability": mc_results["default_probability"],
            "emi": emi,
            "advice": response["advice"]
        })

        logger.info(f"Analysis complete: decision={decision}, risk={risk_level}")
        prediction_count += 1
        return create_response(data=response)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# =============================================================================
# History Endpoint
# =============================================================================

@app.get("/history")
def get_decisions_history(limit: int = Query(default=20, le=100, ge=1)):
    """Get last N decisions from database."""
    try:
        history = get_history(limit)
        return create_response(data=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


# =============================================================================
# Error Monitoring Endpoint
# =============================================================================

class ErrorReport(BaseModel):
    """Client-side error report."""
    message: str = Field(..., max_length=1000)
    stack: Optional[str] = Field(None, max_length=5000)
    url: Optional[str] = Field(None, max_length=500)
    user_agent: Optional[str] = Field(None, max_length=500)


@app.post("/errors")
def report_error(report: ErrorReport):
    """Receive client-side error reports."""
    log_error(report.model_dump(exclude_none=True))
    logger.warning(f"Client error reported: {report.message[:200]}")
    return create_response(data={"status": "logged"})


@app.get("/errors")
def get_errors(limit: int = Query(default=20, le=100, ge=1)):
    """Get recent errors (for admin/monitoring)."""
    return create_response(data=error_log[:limit])


# =============================================================================
# Compare Endpoint
# =============================================================================

@app.get("/compare")
def compare_loan_amounts(
    income: float = Query(..., gt=0, description="Monthly income"),
    loan_amount: float = Query(..., gt=0, description="Base loan amount to compare around"),
    credit_score: int = Query(..., ge=300, le=850, description="Credit score")
):
    """
    Compare loan outcomes for LOW, MEDIUM, HIGH loan amounts.

    Compares loan_amount * 0.5, loan_amount, loan_amount * 1.5 for same person.
    Returns decision and metrics for each scenario.
    """
    try:
        multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5}
        results = {}

        for label, mult in multipliers.items():
            amt = round(loan_amount * mult, 2)
            loan_app = LoanApplication(
                income=income,
                loan_amount=amt,
                interest_rate=8.5,
                loan_term=5,
                credit_score=credit_score,
                existing_loans=0
            )

            # Calculate EMI
            emi = calculate_emi(amt, 8.5, 5)

            # Risk
            risk = calculate_risk(income, amt, credit_score, 0)

            # MC simulation
            mc = mc_simulate({
                "income": income,
                "loan_amount": amt,
                "interest_rate": 8.5,
                "loan_term": 5,
                "credit_score": credit_score
            }, n_simulations=3000)

            if risk == "HIGH" or mc["default_probability"] > 0.35:
                decision = "REJECT"
            elif risk == "LOW" and mc["default_probability"] < 0.15:
                decision = "APPROVE"
            else:
                decision = "CONDITIONAL"

            results[label] = {
                "loan_amount": amt,
                "decision": decision,
                "emi": emi,
                "risk_level": risk,
                "default_probability": mc["default_probability"],
                "worst_case_emi": mc["worst_case_emi"]
            }

        return create_response(data={
            "income": income,
            "credit_score": credit_score,
            "comparison": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_error"}
    )


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation failed",
            "type": "validation_error",
            "errors": errors
        }
    )


@app.exception_handler(sqlite3.Error)
def sqlite_exception_handler(request: Request, exc: sqlite3.Error):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Database error occurred",
            "type": "database_error"
        }
    )


@app.exception_handler(Exception)
def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"}
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)