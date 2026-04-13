# pip install fastapi uvicorn sqlalchemy pydantic
"""
Mortgage AI Decision System - FastAPI REST API
Exposes the mortgage advisor system as a REST API with database persistence.
"""

from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
import sqlite3

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib

from features import engineer_features
from emi import calculate_emi
from risk import calculate_risk
from monte_carlo import simulate as mc_simulate


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

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    db_ok = False
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("SELECT 1").fetchone()
        conn.close()
        db_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "models_loaded": global_model is not None,
        "db_connected": db_ok
    }


# =============================================================================
# Analyze Endpoint
# =============================================================================

@app.post("/analyze")
def analyze_application(application: LoanApplication):
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

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# =============================================================================
# History Endpoint
# =============================================================================

@app.get("/history")
def get_decisions_history(limit: int = Query(default=20, le=100, ge=1)):
    """Get last N decisions from database."""
    try:
        history = get_history(limit)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


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

        return {
            "income": income,
            "credit_score": credit_score,
            "comparison": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    return {"detail": exc.detail}


@app.exception_handler(Exception)
def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)