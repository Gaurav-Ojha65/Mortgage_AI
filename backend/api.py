"""
Mortgage AI Decision System - FastAPI REST API
Exposes the mortgage advisor system as a REST API with database persistence.
Uses ml.predict for all model inference — the single source of truth.
"""

from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
import sqlite3
import logging
import json
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uuid

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address)

# ─── Path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from ml.inference.predict import predict_single, predict_all_models, calculate_emi, MODEL_FEATURES
from ml.utils.features import engineer_features, applicant_to_15_features
from ml.inference.ensemble import MortgageEnsembleModel
from risk_calc import calculate_risk
from risk.decision_policy import get_active_policy, DecisionPolicy, CostModel, get_policy_audit_log
from monte_carlo import simulate as mc_simulate
from model_router import router as model_router
from shap_router import shap_router
from document_router import router as document_router
from fairness_router import router as fairness_router
from routers.data_router import router as data_router
from routers.analytics_router import router as analytics_router
from auth import (
    init_users_table, authenticate_user, create_token, revoke_token,
    get_current_user, get_optional_user, require_role, require_min_role,
    LoginRequest, UserCreate, get_all_users, create_user_db, update_last_login,
)
from audit_log import (
    init_audit_table, log_action, log_from_request,
    get_audit_logs, get_audit_stats,
)


# =============================================================================
# Structured Logging Setup
# =============================================================================

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.now().isoformat(),
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
    logger = logging.getLogger("mortgage_api")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    file_handler = RotatingFileHandler(
        "mortgage_api.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# =============================================================================
# Response Standardization
# =============================================================================

def create_response(data=None, success=True, error=None, request_id=None):
    response = {
        "success": success,
        "request_id": request_id or str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
    }
    if data is not None:
        response["data"] = data
    if error is not None:
        response["error"] = error
    return response


# =============================================================================
# Input Sanitization
# =============================================================================

def sanitize_string(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = value.strip().replace("\x00", "")
    suspicious = ["--", "/*", "*/", ";", "DROP", "SELECT", "INSERT", "DELETE", "UPDATE"]
    for pattern in suspicious:
        if pattern.upper() in value.upper():
            logger.warning(f"Suspicious pattern detected in input: {pattern}")
    return value


# =============================================================================
# Pydantic Models
# =============================================================================

class LoanApplication(BaseModel):
    """Validated loan application input — maps to simple /analyze endpoint."""
    income: float = Field(..., gt=0, le=5000000, description="Monthly income (max $5M)")
    loan_amount: float = Field(..., gt=0, le=25000000, description="Loan amount (max $25M)")
    interest_rate: float = Field(..., gt=0, le=50, description="Annual interest rate %")
    loan_term: int = Field(..., gt=0, le=50, description="Loan term in years")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score 300-850")
    existing_loans: int = Field(default=0, ge=0, le=50, description="Number of existing loans")
    # Optional extended fields for 15-feature model
    employment_years: Optional[float] = Field(default=None, ge=0, le=70)
    num_credit_lines: Optional[int] = Field(default=None, ge=0)
    num_derogatory_marks: Optional[int] = Field(default=None, ge=0)
    credit_utilization: Optional[float] = Field(default=None, ge=0, le=1.0)
    late_payment_severity_score: Optional[float] = Field(default=None, ge=0, le=1.0)
    dti_ratio: Optional[float] = Field(default=None, ge=0, le=1.0)
    home_ownership: Optional[int] = Field(default=1, ge=0, le=2)
    purpose_encoded: Optional[int] = Field(default=0, ge=0, le=9)
    num_late_payments: Optional[int] = Field(default=0, ge=0)
    savings_balance: Optional[float] = Field(default=None, ge=0)
    monthly_expenses: Optional[float] = Field(default=None, ge=0)
    # Demographic proxies for bias monitoring
    age_band: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default=None)


    @field_validator("income", "loan_amount", "interest_rate")
    @classmethod
    def reject_nonpositive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
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

from database import DATABASE_PATH, get_connection


def init_db():
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
            approval_probability REAL,
            emi REAL NOT NULL,
            model_used TEXT,
            advice TEXT,
            age_band TEXT,
            region TEXT
        )
    """)
    
    # Migration: Check for missing columns if table already existed
    cursor.execute("PRAGMA table_info(decisions)")
    columns = [row[1] for row in cursor.fetchall()]
    
    required_columns = [
        ("model_used", "TEXT"),
        ("approval_probability", "REAL"),
        ("age_band", "TEXT"),
        ("region", "TEXT"),
        ("user_id", "INTEGER"),
        ("model_version", "TEXT"),
        ("calibration_version", "TEXT"),
        ("policy_version", "TEXT")
    ]
    
    for col_name, col_type in required_columns:
        if col_name not in columns:
            logger.info(f"Migrating database: Adding missing column {col_name} to decisions table")
            try:
                cursor.execute(f"ALTER TABLE decisions ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                logger.error(f"Failed to add column {col_name}: {e}")

    conn.commit()
    conn.close()


def save_decision(data: dict):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO decisions
        (timestamp, income, loan_amount, credit_score, decision, risk_level,
         default_probability, approval_probability, emi, model_used, advice, age_band, region, user_id,
         model_version, calibration_version, policy_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["timestamp"], data["income"], data["loan_amount"],
        data["credit_score"], data["decision"], data["risk_level"],
        data.get("default_probability"), data.get("approval_probability"),
        data["emi"], data.get("model_used"), data.get("advice"),
        data.get("age_band"), data.get("region"), data.get("user_id"),
        data.get("model_version"), data.get("calibration_version"), data.get("policy_version")
    ))

    conn.commit()
    conn.close()


def get_history(limit: int = 20, user_id: Optional[int] = None) -> List[dict]:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if user_id is not None:
        cursor.execute("SELECT * FROM decisions WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit))
    else:
        cursor.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# =============================================================================
# Global State
# =============================================================================

startup_time = datetime.now()
prediction_count = 0
_models_loaded = False
error_log = []
MAX_ERROR_LOG = 100
APP_VERSION = "2.0.0"


def log_error(error_data: dict):
    error_log.insert(0, {"timestamp": datetime.now().isoformat(), "error": error_data})
    if len(error_log) > MAX_ERROR_LOG:
        error_log.pop()


def get_uptime_seconds():
    return int((datetime.now() - startup_time).total_seconds())


def get_memory_usage():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return None


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_loaded
    logger.info(f"==================================================")
    logger.info(f"   Mortgage AI Decision System v{APP_VERSION}")
    logger.info(f"==================================================")
    
    try:
        # Initialize resources
        init_db()
        init_users_table()
        init_audit_table()
        
        # Log model status
        from ml.inference.predict import get_model, get_active_model_name, AVAILABLE_MODELS, MODELS_DIR, MODEL_FEATURES
        active = get_active_model_name()
        logger.info(f"[Startup] Active Model Target : {active}")
        logger.info(f"[Startup] Expected Features   : {len(MODEL_FEATURES)}")
        
        for name in AVAILABLE_MODELS:
            path = MODELS_DIR / f"{name}.joblib"
            if path.exists():
                logger.info(f"[Startup] Model Status - {name:15}: READY ({(path.stat().st_size/1024):.1f} KB)")
            else:
                logger.warning(f"[Startup] Model Status - {name:15}: MISSING")
        
        # Warm up model cache
        try:
            get_model(active)
            _models_loaded = True
            logger.info(f"[Startup] Cache Status         : WARMED")
        except Exception as me:
            logger.error(f"[Startup] Cache Status         : FAILED ({str(me)})")
            _models_loaded = False
            
    except Exception as e:
        logger.error(f"[Startup] CRITICAL INITIALIZATION ERROR: {str(e)}")
        log_error({"type": "startup_failure", "msg": str(e)})
    
    logger.info(f"==================================================")
    log_action(action="SYSTEM_START", metadata={"version": APP_VERSION})
    yield
    logger.info("=== Mortgage AI Shutdown ===")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Mortgage AI Decision API",
    description="AI-powered mortgage loan approval with real ML models, Monte Carlo simulation, and SHAP explainability",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Request size limit
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    max_size = 1 * 1024 * 1024
    body = await request.body()
    if len(body) > max_size:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large", "type": "payload_too_large"},
        )
    async def receive():
        return {"type": "http.request", "body": body}
    request._receive = receive
    return await call_next(request)


# CORS
cors_origins_str = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(model_router, prefix="/api")
app.include_router(shap_router, prefix="/api")
app.include_router(document_router, prefix="/api")
app.include_router(fairness_router, prefix="/api")
app.include_router(data_router)
app.include_router(analytics_router)


@app.on_event("startup")
async def load_ensemble_startup():
    try:
        from ml.inference.predict import MODELS_DIR
        ensemble_path = MODELS_DIR / "ensemble.joblib"
        if ensemble_path.exists():
            model = MortgageEnsembleModel()
            model.load(str(ensemble_path))
            version = getattr(model, "version", "unknown")
            feature_names = getattr(model, "feature_names", [])
            feature_count = len(feature_names) if feature_names else 0
            logger.info(f"MortgageEnsembleModel loaded successfully. Version: {version}, Features: {feature_count}")
    except Exception as e:
        logger.warning(f"Optional MortgageEnsembleModel startup check: {e}")


# =============================================================================
# Health & Metrics Endpoints
# =============================================================================

@app.get("/health")
def health_check():
    db_ok = False
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("SELECT 1").fetchone()
        conn.close()
        db_ok = True
    except Exception:
        pass

    from ml.inference.predict import get_active_model_name, MODELS_DIR
    active_model = get_active_model_name()
    model_files = list(MODELS_DIR.glob("*.joblib")) if MODELS_DIR.exists() else []
    active_policy = get_active_policy()

    response = {
        "status": "ok",
        "version": APP_VERSION,
        "model_version": "v3.1",
        "calibration_version": "oof-iso-v3.1",
        "policy_version": getattr(active_policy, "policy_version", "v3.1-policy-v1"),
        "policy_thresholds": {
            "approve_threshold": getattr(active_policy, "approve_threshold", 0.045),
            "reject_threshold": getattr(active_policy, "reject_threshold", 0.335),
        },
        "uptime_seconds": get_uptime_seconds(),
        "models_loaded": _models_loaded,
        "active_model": active_model,
        "model_count": len(model_files),
        "db_connected": db_ok,
        "predictions_served": prediction_count,
    }

    mem = get_memory_usage()
    if mem:
        response["memory_usage_mb"] = round(mem, 2)

    return create_response(data=response)


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    from fastapi.responses import PlainTextResponse
    uptime = get_uptime_seconds()
    mem = get_memory_usage() or 0.0
    active_policy = get_active_policy()
    p_ver = getattr(active_policy, "policy_version", "v3.1-policy-v1")
    
    lines = [
        "# HELP mortgage_uptime_seconds Total uptime of the Mortgage AI service in seconds",
        "# TYPE mortgage_uptime_seconds gauge",
        f"mortgage_uptime_seconds {uptime}",
        "# HELP mortgage_predictions_total Total predictions served since startup",
        "# TYPE mortgage_predictions_total counter",
        f"mortgage_predictions_total {prediction_count}",
        "# HELP mortgage_memory_usage_mb Process resident memory usage in MB",
        "# TYPE mortgage_memory_usage_mb gauge",
        f"mortgage_memory_usage_mb {mem:.2f}",
        "# HELP mortgage_model_info Active model and policy version information",
        "# TYPE mortgage_model_info gauge",
        f'mortgage_model_info{{model_version="v3.1",calibration_version="oof-iso-v3.1",policy_version="{p_ver}"}} 1',
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


# =============================================================================
# Analyze Endpoint (Primary Prediction)
# =============================================================================

@app.post("/analyze")
async def analyze_application(application: LoanApplication, request: Request, user: dict = Depends(get_optional_user)):
    """
    Analyze a loan application with real ML + Monte Carlo simulation.

    Pipeline:
    1. engineer_features() → derived ratio features
    2. applicant_to_15_features() → map to full 15-feature schema
    3. predict_single() → real ML model probability
    4. calculate_risk() → rule-based risk level
    5. mc_simulate() → Monte Carlo default probability
    6. Combine into final decision
    """
    global prediction_count
    logger.info(
        f"Analyzing: income={application.income}, "
        f"loan={application.loan_amount}, credit={application.credit_score}"
    )

    try:
        loan_data = application.model_dump()

        # Step 1: Engineer basic features + compute EMI
        emi = calculate_emi(
            loan_data["loan_amount"],
            loan_data["interest_rate"],
            loan_data["loan_term"] * 12,  # years → months
        )
        loan_data["emi"] = emi
        enriched = engineer_features(loan_data)

        # Step 2: Map to 15-feature schema for ML model
        features_15 = applicant_to_15_features(loan_data)

        # Step 3: ML prediction (canonical source of truth)
        try:
            ml_result = predict_single(features_15)
            model_used = ml_result.get("model_used", "unknown")
            decision = ml_result["decision"]
            raw_default_probability = ml_result["raw_default_probability"]
            calibrated_default_probability = ml_result["calibrated_default_probability"]
            risk_tier = ml_result["risk_tier"]
            approval_prob = 1.0 - calibrated_default_probability
        except Exception as e:
            logger.error(f"Canonical ML model failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={"code": "MODEL_UNAVAILABLE", "message": "Canonical risk model unavailable"}
            )

        # Step 4: Rule-based risk (for heuristic/comparison only, strictly non-canonical)
        heuristic_risk_level = calculate_risk(
            loan_data["income"],
            loan_data["loan_amount"],
            loan_data["credit_score"],
            loan_data["existing_loans"],
        )

        # Step 5: Monte Carlo simulation (for scenario/sensitivity only)
        mc_results = mc_simulate(
            {
                "income": loan_data["income"],
                "loan_amount": loan_data["loan_amount"],
                "interest_rate": loan_data["interest_rate"],
                "loan_term": loan_data["loan_term"],
                "credit_score": loan_data["credit_score"],
            },
            n_simulations=5000,
        )

        # Step 6: SHAP Explanation
        try:
            from shap_explainer import explain_decision
            from ml.inference.predict import get_model
            
            clf = get_model(model_used)
            explanation = explain_decision(features_15, clf, model_used)
            
            explanation_status = explanation.get("explanation_status", "available")
            if explanation_status == "available":
                top_factors = [f["label"] for f in explanation["top_factors"][:5]]
                plain_english = explanation["plain_english"]
            else:
                top_factors = []
                plain_english = []
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            explanation_status = "unavailable"
            top_factors = []
            plain_english = []
            explanation_error = str(e)

        # Step 7: Advice
        advice = []
        if mc_results["default_probability"] > 0.15:
            advice.append(f"Monte Carlo scenario risk is {mc_results['default_probability']:.1%} — consider reducing loan amount")
        if heuristic_risk_level == "HIGH":
            advice.append("Heuristic risk profile is HIGH based on income-to-loan ratio")
        if loan_data["credit_score"] < 650:
            advice.append("Credit score below 650 — improving it will significantly help")
        if enriched["emi_to_income_ratio"] > 35:
            advice.append(f"EMI burden is {enriched['emi_to_income_ratio']:.1f}% of income — aim for under 35%")
        if not advice:
            advice.append("Application meets criteria — proceed with submission")

        # Compile response with calibrated model and policy metadata
        response_data = {
            "timestamp": datetime.now().isoformat(),
            
            # Canonical Fields
            "decision": decision,
            "raw_default_probability": raw_default_probability,
            "calibrated_default_probability": calibrated_default_probability,
            "risk_tier": risk_tier,
            
            # Non-Canonical / Heuristic Fields
            "heuristic_risk_level": heuristic_risk_level,
            "emi": emi,
            
            # Additional ML info
            "model_used": model_used,
            "model_name": ml_result.get("model_name", model_used),
            "model_version": ml_result.get("model_version", "v3.1"),
            "calibration_method": ml_result.get("calibration_method", "isotonic"),
            "calibration_version": ml_result.get("calibration_version", "oof-iso-v3.1"),
            "policy_version": ml_result.get("policy_version", "v3.1-policy-v1"),
            "decision_policy": ml_result.get("policy_metadata"),
            "expected_economic_cost": ml_result.get("expected_economic_cost"),

            # SHAP
            "explanation_status": explanation_status,
            "top_factors": top_factors,
            "plain_english": plain_english,
            "ai_advice": False,
            "advice": "; ".join(advice),

            "feature_values": {
                "debt_to_income_ratio": enriched["debt_to_income_ratio"],
                "emi_to_income_ratio": enriched["emi_to_income_ratio"],
                "credit_utilization_score": enriched["credit_utilization_score"],
                "loan_burden_index": enriched["loan_burden_index"],
                "affordability_score": enriched["affordability_score"],
            },
            
            # Monte Carlo Scenario Block
            "monte_carlo": {
                "scenario_default_probability": mc_results["default_probability"],
                "worst_case_emi": mc_results["worst_case_emi"],
                "safe_income_threshold": mc_results["safe_income_threshold"],
                "scenario_breakdown": mc_results["scenario_breakdown"],
                "mean_emi_ratio": mc_results["mean_emi_ratio"],
            },
        }

        # Persist to DB using canonical values where possible
        save_decision({
            "timestamp": response_data["timestamp"],
            "income": loan_data["income"],
            "loan_amount": loan_data["loan_amount"],
            "credit_score": loan_data["credit_score"],
            "decision": decision,
            "risk_level": risk_tier, # Storing canonical tier in DB
            "default_probability": calibrated_default_probability,
            "approval_probability": approval_prob,
            "emi": emi,
            "model_used": model_used,
            "advice": response_data["advice"],
            "age_band": loan_data.get("age_band"),
            "region": loan_data.get("region"),
            "user_id": user.get("user_id") if user else None,
            "model_version": response_data["model_version"],
            "calibration_version": response_data["calibration_version"],
            "policy_version": response_data["policy_version"],
        })

        prediction_count += 1
        logger.info(f"Analysis done: decision={decision}, risk_tier={risk_tier}, "
                    f"calibrated_default_prob={calibrated_default_probability}, model={model_used}")

        # Audit log
        log_from_request(
            request, action="PREDICT", user=user,
            target_type="application", target_id=str(prediction_count),
            after_value={
                "decision": decision,
                "risk_tier": risk_tier,
                "raw_default_probability": raw_default_probability,
                "calibrated_default_probability": calibrated_default_probability,
                "model_version": ml_result.get("model_version", "v3.1"),
                "calibration_version": ml_result.get("calibration_version", "oof-iso-v3.1"),
                "policy_version": ml_result.get("policy_version", "v3.1-policy-v1"),
                "expected_economic_cost": ml_result.get("expected_economic_cost"),
                "heuristic_risk_level": heuristic_risk_level
            },
            metadata={"income": loan_data["income"], "loan_amount": loan_data["loan_amount"],
                      "credit_score": loan_data["credit_score"]},
        )

        return create_response(data=response_data)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# =============================================================================
# Decision Policy Endpoints (Inspection & Simulation)
# =============================================================================

class PolicyEvaluateRequest(BaseModel):
    application: LoanApplication
    approve_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    reject_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    cost_fn: Optional[float] = Field(None, ge=0.0)
    cost_fp: Optional[float] = Field(None, ge=0.0)
    cost_manual_review: Optional[float] = Field(None, ge=0.0)


@app.get("/policy/config")
def get_policy_configuration():
    """Inspect the current active frozen decision policy and audit history."""
    active = get_active_policy()
    audit_trail = get_policy_audit_log()
    return create_response(data={
        "active_policy": active.to_dict(),
        "audit_trail_count": len(audit_trail),
        "audit_trail": audit_trail[-10:],  # last 10 audit records
    })


@app.post("/policy/evaluate")
def evaluate_policy_simulation(payload: PolicyEvaluateRequest, user: dict = Depends(get_optional_user)):
    """
    Analyst/admin-only simulation endpoint to evaluate an application under candidate policy parameters.
    Does NOT mutate global active policy.
    """
    active = get_active_policy()
    
    # Construct candidate custom policy for simulation
    cost_m = CostModel(
        cost_fn=payload.cost_fn if payload.cost_fn is not None else active.cost_model.cost_fn,
        cost_fp=payload.cost_fp if payload.cost_fp is not None else active.cost_model.cost_fp,
        cost_manual_review=payload.cost_manual_review if payload.cost_manual_review is not None else active.cost_model.cost_manual_review,
    )
    
    app_t = payload.approve_threshold if payload.approve_threshold is not None else active.approve_threshold
    rej_t = payload.reject_threshold if payload.reject_threshold is not None else active.reject_threshold
    
    if app_t > rej_t:
        raise HTTPException(
            status_code=400,
            detail=f"approve_threshold ({app_t}) cannot exceed reject_threshold ({rej_t})"
        )
    
    candidate_policy = DecisionPolicy(
        policy_name="simulation_candidate",
        policy_version="sim-v1.0",
        approve_threshold=app_t,
        reject_threshold=rej_t,
        cost_model=cost_m,
        description="Ad-hoc simulation run"
    )
    
    loan_data = payload.application.model_dump()
    features_15 = applicant_to_15_features(loan_data)
    
    result = predict_single(
        features_15,
        model_name="lightgbm",
        calibration_method="isotonic",
        policy=candidate_policy
    )
    
    return create_response(data=result)


# =============================================================================
# History Endpoint
# =============================================================================

@app.get("/history")
def get_decisions_history(limit: int = Query(default=20, le=100, ge=1), user: dict = Depends(get_current_user)):
    try:
        if user["role"] in ["admin", "underwriter"]:
            history = get_history(limit)
        else:
            history = get_history(limit, user_id=user["user_id"])
        return create_response(data=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@app.post("/api/data/delete/{decision_id}")
def delete_decision(decision_id: int, admin: dict = Depends(require_role("admin"))):
    """Admin-only endpoint for GDPR/DPDP data deletion compliance."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM decisions WHERE id = ?", (decision_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Decision record not found")
    
    log_action(
        action="DATA_DELETION", 
        user_id=admin["user_id"], 
        metadata={"decision_id": decision_id, "reason": "GDPR/DPDP compliance"}
    )
    
    return create_response(data={"message": f"Record {decision_id} deleted successfully"})


@app.get("/api/analytics/fairness")
def get_fairness_metrics(admin: dict = Depends(require_role("admin"))):
    """Monitor approval rates by demographic proxies for bias detection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Approval rate by Age Band
    cursor.execute("""
        SELECT age_band, 
               COUNT(*) as total, 
               SUM(CASE WHEN decision = 'APPROVED' THEN 1 ELSE 0 END) as approved
        FROM decisions 
        WHERE age_band IS NOT NULL
        GROUP BY age_band
    """)
    age_stats = [dict(r) for r in cursor.fetchall()]
    for s in age_stats:
        s["approval_rate"] = s["approved"] / s["total"] if s["total"] > 0 else 0
        
    # Approval rate by Region
    cursor.execute("""
        SELECT region, 
               COUNT(*) as total, 
               SUM(CASE WHEN decision = 'APPROVED' THEN 1 ELSE 0 END) as approved
        FROM decisions 
        WHERE region IS NOT NULL
        GROUP BY region
    """)
    region_stats = [dict(r) for r in cursor.fetchall()]
    for s in region_stats:
        s["approval_rate"] = s["approved"] / s["total"] if s["total"] > 0 else 0
        
    conn.close()
    
    return create_response(data={
        "by_age": age_stats,
        "by_region": region_stats,
        "timestamp": datetime.now().isoformat()
    })



# =============================================================================
# Auth Endpoints
# =============================================================================

@app.post("/auth/login")
@limiter.limit("5/minute")
async def login(body: LoginRequest, request: Request):
    """Secure login with rate-limiting and audit logging."""
    try:
        user = authenticate_user(body.username, body.password)
    except HTTPException as e:
        # Log failed attempt if it was a 401 or 429
        log_action(action="LOGIN_FAILED", metadata={"username": body.username, "reason": str(e.detail)},
                   ip_address=request.client.host if request.client else "unknown")
        raise e

    token = create_token(user["id"], user["username"], user["role"])
    update_last_login(user["id"])
    log_action(action="LOGIN", user_id=user["id"], username=user["username"],
               user_role=user["role"],
               ip_address=request.client.host if request.client else "unknown")

    return create_response(data={
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "full_name": user["full_name"],
        }
    })



@app.post("/auth/logout")
async def logout(request: Request, user: dict = Depends(get_current_user)):
    # Revoke token from Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        revoke_token(auth_header[7:])
    log_from_request(request, action="LOGOUT", user=user)
    return create_response(data={"status": "logged_out"})


@app.get("/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    return create_response(data={
        "user_id": user["user_id"],
        "username": user["username"],
        "role": user["role"],
    })


@app.get("/auth/users")
async def list_users(user: dict = Depends(require_role("admin"))):
    users = get_all_users()
    return create_response(data=users)


@app.post("/auth/users")
async def create_user(body: UserCreate, request: Request, user: dict = Depends(require_role("admin"))):
    new_user = create_user_db(body)
    log_from_request(request, action="USER_CREATE", user=user,
                     target_type="user", target_id=str(new_user["id"]),
                     after_value={"username": new_user["username"], "role": new_user["role"]})
    return create_response(data=new_user)


# =============================================================================
# Audit Log Endpoints (Admin Only)
# =============================================================================

@app.get("/audit")
async def get_audit(
    limit: int = Query(default=50, le=200, ge=1),
    offset: int = Query(default=0, ge=0),
    user_id: Optional[int] = Query(default=None),
    action: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    user: dict = Depends(require_role("admin")),
):
    logs = get_audit_logs(limit=limit, offset=offset, user_id=user_id,
                          action=action, date_from=date_from, date_to=date_to)
    return create_response(data=logs)


@app.get("/audit/stats")
async def audit_stats(user: dict = Depends(require_role("admin"))):
    stats = get_audit_stats()
    return create_response(data=stats)


# =============================================================================
# What-If Risk Simulator
# =============================================================================

class WhatIfRequest(BaseModel):
    """What-if simulation: original inputs + one or more modified fields."""
    # Original application
    income: float = Field(..., gt=0, le=5000000)
    loan_amount: float = Field(..., gt=0, le=25000000)
    interest_rate: float = Field(..., gt=0, le=50)
    loan_term: int = Field(..., gt=0, le=50)
    credit_score: int = Field(..., ge=300, le=850)
    existing_loans: int = Field(default=0, ge=0, le=50)
    # Modified scenario
    new_income: Optional[float] = Field(default=None, gt=0, le=5000000)
    new_loan_amount: Optional[float] = Field(default=None, gt=0, le=25000000)
    new_interest_rate: Optional[float] = Field(default=None, gt=0, le=50)
    new_credit_score: Optional[int] = Field(default=None, ge=300, le=850)
    new_existing_loans: Optional[int] = Field(default=None, ge=0, le=50)
    new_loan_term: Optional[int] = Field(default=None, gt=0, le=50)


def _run_risk_pipeline(params: dict) -> dict:
    """Run the full risk pipeline on a set of parameters and return results."""
    emi = calculate_emi(params["loan_amount"], params["interest_rate"], params["loan_term"] * 12)
    params["emi"] = emi
    enriched = engineer_features(params)

    features_15 = applicant_to_15_features(params)
    try:
        ml_result = predict_single(features_15)
        model_used = ml_result.get("model_used", "unknown")
        decision = ml_result["decision"]
        calibrated_default_probability = ml_result["calibrated_default_probability"]
        approval_prob = 1.0 - calibrated_default_probability
        risk_tier = ml_result["risk_tier"]
    except Exception as e:
        logger.error(f"WhatIf ML model failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_UNAVAILABLE", "message": "Canonical risk model unavailable"}
        )

    heuristic_risk_level = calculate_risk(params["income"], params["loan_amount"],
                                          params["credit_score"], params["existing_loans"])

    mc_results = mc_simulate({
        "income": params["income"], "loan_amount": params["loan_amount"],
        "interest_rate": params["interest_rate"], "loan_term": params["loan_term"],
        "credit_score": params["credit_score"],
    }, n_simulations=3000)

    mc_default = mc_results["default_probability"]

    return {
        "decision": decision,
        "risk_level": risk_tier,
        "heuristic_risk_level": heuristic_risk_level,
        "default_probability": calibrated_default_probability,
        "approval_probability": approval_prob,
        "scenario_default_probability": mc_default,
        "emi": emi,
        "model_used": model_used,
        "feature_values": {
            "debt_to_income_ratio": enriched["debt_to_income_ratio"],
            "emi_to_income_ratio": enriched["emi_to_income_ratio"],
            "credit_utilization_score": enriched.get("credit_utilization_score", 0),
            "loan_burden_index": enriched.get("loan_burden_index", 0),
            "affordability_score": enriched.get("affordability_score", 0),
        },
    }


@app.post("/whatif")
async def whatif_simulator(body: WhatIfRequest, request: Request, user: dict = Depends(get_optional_user)):
    """
    Run the original scenario and a modified scenario side-by-side.
    Returns both results + deltas for every metric.
    """
    try:
        original_params = {
            "income": body.income, "loan_amount": body.loan_amount,
            "interest_rate": body.interest_rate, "loan_term": body.loan_term,
            "credit_score": body.credit_score, "existing_loans": body.existing_loans,
        }

        modified_params = {
            "income": body.new_income or body.income,
            "loan_amount": body.new_loan_amount or body.loan_amount,
            "interest_rate": body.new_interest_rate or body.interest_rate,
            "loan_term": body.new_loan_term or body.loan_term,
            "credit_score": body.new_credit_score or body.credit_score,
            "existing_loans": body.new_existing_loans if body.new_existing_loans is not None else body.existing_loans,
        }

        original_result = _run_risk_pipeline(original_params)
        modified_result = _run_risk_pipeline(modified_params)

        # Compute deltas
        deltas = {
            "default_probability": (modified_result["default_probability"] or 0) - (original_result["default_probability"] or 0),
            "approval_probability": (modified_result["approval_probability"] or 0) - (original_result["approval_probability"] or 0),
            "emi": modified_result["emi"] - original_result["emi"],
            "decision_changed": modified_result["decision"] != original_result["decision"],
            "risk_level_changed": modified_result["risk_level"] != original_result["risk_level"],
        }

        # Which factors improved / worsened
        factor_changes = []
        for key in modified_result["feature_values"]:
            orig_val = original_result["feature_values"].get(key, 0) or 0
            mod_val = modified_result["feature_values"].get(key, 0) or 0
            if abs(mod_val - orig_val) > 0.5:
                factor_changes.append({
                    "factor": key,
                    "original": round(orig_val, 2),
                    "modified": round(mod_val, 2),
                    "delta": round(mod_val - orig_val, 2),
                    "direction": "improved" if mod_val < orig_val else "worsened",
                })

        log_from_request(request, action="WHATIF_SIMULATE", user=user,
                         metadata={"original_params": original_params, "modified_params": modified_params})

        return create_response(data={
            "original": original_result,
            "modified": modified_result,
            "deltas": deltas,
            "factor_changes": factor_changes,
            "changes_applied": {k: v for k, v in {
                "income": body.new_income, "loan_amount": body.new_loan_amount,
                "interest_rate": body.new_interest_rate, "credit_score": body.new_credit_score,
                "existing_loans": body.new_existing_loans, "loan_term": body.new_loan_term,
            }.items() if v is not None},
        })

    except Exception as e:
        logger.error(f"What-if simulation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


# =============================================================================
# Compare Endpoint
# =============================================================================

@app.get("/compare")
def compare_loan_amounts(
    income: float = Query(..., gt=0),
    loan_amount: float = Query(..., gt=0),
    credit_score: int = Query(..., ge=300, le=850),
):
    """Compare LOW / MEDIUM / HIGH loan amounts for the same applicant."""
    try:
        multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5}
        results = {}

        for label, mult in multipliers.items():
            amt = round(loan_amount * mult, 2)
            emi = calculate_emi(amt, 8.5, 60)  # 8.5% / 5yr default
            risk = calculate_risk(income, amt, credit_score, 0)
            mc = mc_simulate(
                {"income": income, "loan_amount": amt,
                 "interest_rate": 8.5, "loan_term": 5, "credit_score": credit_score},
                n_simulations=3000,
            )

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
                "worst_case_emi": mc["worst_case_emi"],
            }

        return create_response(data={
            "income": income,
            "credit_score": credit_score,
            "comparison": results,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


# =============================================================================
# Error Reporting
# =============================================================================

class ErrorReport(BaseModel):
    message: str = Field(..., max_length=1000)
    stack: Optional[str] = Field(None, max_length=5000)
    url: Optional[str] = Field(None, max_length=500)
    user_agent: Optional[str] = Field(None, max_length=500)


@app.post("/errors")
def report_error(report: ErrorReport):
    log_error(report.model_dump(exclude_none=True))
    logger.warning(f"Client error: {report.message[:200]}")
    return create_response(data={"status": "logged"})


@app.get("/errors")
def get_errors(limit: int = Query(default=20, le=100, ge=1)):
    return create_response(data=error_log[:limit])


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_error"},
    )


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        errors.append({"field": field, "message": error["msg"], "type": error["type"]})
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation failed", "type": "validation_error", "errors": errors},
    )


@app.exception_handler(sqlite3.Error)
def sqlite_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Database error", "type": "database_error"},
    )


@app.exception_handler(Exception)
def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)