"""
Fairness Router for Mortgage AI

FastAPI router for fairness audit endpoints.
Provides endpoints for running audits, retrieving reports, and group metrics.

Usage:
    from fairness_router import router
    app.include_router(router, prefix="/api/fairness")
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/fairness", tags=["synthetic fairness stress test"])

# =============================================================================
# Configuration
# =============================================================================

REPORTS_DIR = os.getenv("FAIRNESS_REPORTS_DIR", "reports/metrics")
_CANDIDATE_PATHS = [
    Path(REPORTS_DIR) / "fairness_report.json",
    Path(__file__).resolve().parent.parent / "reports" / "metrics" / "fairness_report.json",
    Path(__file__).resolve().parent.parent / "reports" / "metrics" / "hpo_fairness_report.json",
    Path("models/fairness_report.json"),
]
LATEST_REPORT_PATH = next((p for p in _CANDIDATE_PATHS if p.exists()), _CANDIDATE_PATHS[0])

# =============================================================================
# Pydantic Models
# =============================================================================


class GroupMetricsResponse(BaseModel):
    """Metrics for a single demographic group."""
    group_name: str
    sample_size: int
    approval_rate: float
    average_credit_score: float
    false_positive_rate: float
    false_negative_rate: float


class FairnessReportResponse(BaseModel):
    """Response from fairness report endpoint."""
    audit_timestamp: str
    model_name: str
    overall_fairness_score: float
    total_samples: int
    group_metrics: List[Dict[str, Any]]
    violations: List[Dict[str, Any]]
    mitigations: List[Dict[str, Any]]
    summary: str
    recommendations: List[str]


class FairnessRunRequest(BaseModel):
    """Request to trigger fairness audit."""
    model_name: Optional[str] = "mortgage_model"
    sample_size: Optional[int] = 10000


class FairnessGroupResponse(BaseModel):
    """Response from groups endpoint."""
    groups: List[Dict[str, Any]]
    reference_group: str
    disparity_summary: Dict[str, float]


# =============================================================================
# In-memory cache for latest report
# =============================================================================

_latest_report: Optional[Dict] = None
_last_audit_time: Optional[datetime] = None


def get_cached_report() -> Optional[Dict]:
    """Get cached fairness report."""
    global _latest_report

    if _latest_report is None:
        # Try to load from file
        if LATEST_REPORT_PATH.exists():
            try:
                with open(LATEST_REPORT_PATH, 'r') as f:
                    _latest_report = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cached report: {e}")

    return _latest_report


def set_cached_report(report: Dict):
    """Cache fairness report."""
    global _latest_report, _last_audit_time
    _latest_report = report
    _last_audit_time = datetime.now()


def _normalize_disk_report(report: Dict) -> Dict:
    """
    Transform the on-disk fairness report format (produced by
    ml/training/fairness_audit.py) into the FairnessReportResponse schema.

    On-disk keys: metadata, overall_portfolio, disparity_summary, subgroups
    Response keys: audit_timestamp, model_name, overall_fairness_score,
                   total_samples, group_metrics, violations, mitigations,
                   summary, recommendations
    """
    metadata = report.get("metadata", {})
    portfolio = report.get("overall_portfolio", {})
    disparities = report.get("disparity_summary", {})
    subgroups = report.get("subgroups", {})

    # --- model_name & audit_timestamp ---
    model_name = metadata.get("model", "unknown")
    audit_timestamp = metadata.get("audit_timestamp", datetime.now().isoformat())

    # --- total_samples ---
    total_samples = portfolio.get("sample_size", 0)

    # --- overall_fairness_score (0-100) ---
    # Average of all disparate_impact_ratios × 100; higher = fairer.
    di_ratios = [
        feat.get("disparate_impact_ratio", 1.0)
        for feat in disparities.values()
    ]
    overall_fairness_score = (
        round((sum(di_ratios) / len(di_ratios)) * 100, 1)
        if di_ratios else 0.0
    )

    # --- group_metrics: flatten subgroups ---
    group_metrics: List[Dict[str, Any]] = []
    for feature_name, groups in subgroups.items():
        for group_name, stats in groups.items():
            group_metrics.append({
                "group_name": f"{feature_name}:{group_name}",
                "sample_size": stats.get("sample_size", 0),
                "approval_rate": stats.get("approval_rate", 0.0),
                "average_credit_score": 0.0,  # not tracked in on-disk format
                "false_positive_rate": stats.get("fpr", 0.0),
                "false_negative_rate": stats.get("fnr", 0.0),
            })

    # --- violations: flag features failing the 4/5ths rule ---
    violations: List[Dict[str, Any]] = []
    FOUR_FIFTHS = 0.80
    for feature_name, disp in disparities.items():
        di = disp.get("disparate_impact_ratio", 1.0)
        if di < FOUR_FIFTHS:
            violations.append({
                "feature": feature_name,
                "type": "disparate_impact",
                "severity": "high" if di < 0.70 else "medium",
                "disparate_impact_ratio": di,
                "highest_group": disp.get("highest_approval_group", ""),
                "lowest_group": disp.get("lowest_approval_group", ""),
                "message": (
                    f"{feature_name}: disparate impact ratio {di:.4f} "
                    f"is below the 4/5ths threshold ({FOUR_FIFTHS})"
                ),
            })

    # --- mitigations ---
    mitigations: List[Dict[str, Any]] = []
    if violations:
        mitigations.append({
            "type": "recommendation",
            "message": "Review features contributing to disparate impact and consider re-calibration or threshold adjustments per subgroup.",
        })
    else:
        mitigations.append({
            "type": "info",
            "message": "All features pass the 4/5ths disparate impact threshold.",
        })

    # --- summary ---
    n_violations = len(violations)
    summary = (
        f"Fairness audit on {total_samples:,} samples using {model_name}. "
        f"Overall fairness score: {overall_fairness_score}/100. "
        f"{n_violations} violation(s) detected across {len(disparities)} protected features."
    )

    # --- recommendations ---
    recommendations: List[str] = []
    if n_violations > 0:
        recommendations.append("Investigate features with disparate impact ratio below 0.80.")
        recommendations.append("Consider post-hoc threshold tuning per subgroup to mitigate disparities.")
    recommendations.append("Re-run fairness audit after any model or policy change.")

    return {
        "audit_timestamp": audit_timestamp,
        "model_name": model_name,
        "overall_fairness_score": overall_fairness_score,
        "total_samples": total_samples,
        "group_metrics": group_metrics,
        "violations": violations,
        "mitigations": mitigations,
        "summary": summary,
        "recommendations": recommendations,
    }


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/report", response_model=FairnessReportResponse)
async def get_fairness_report():
    """
    Get the latest fairness audit report.

    Returns comprehensive fairness metrics including:
    - Overall fairness score (0-100)
    - Per-group metrics (approval rates, FPR, FNR)
    - Detected violations (ECOA, Fair Housing)
    - Recommended mitigations
    """
    report = get_cached_report()

    if report is None:
        # Return empty report with instructions
        return FairnessReportResponse(
            audit_timestamp=datetime.now().isoformat(),
            model_name="unknown",
            overall_fairness_score=0.0,
            total_samples=0,
            group_metrics=[],
            violations=[],
            mitigations=[{
                "type": "info",
                "message": "No audit available. Run POST /api/fairness/run to generate."
            }],
            summary="No synthetic fairness stress test has been run yet.",
            recommendations=["Run a synthetic fairness stress test to evaluate model behavior"],
        )

    # If the report is in the on-disk format (has 'metadata'/'subgroups' keys
    # instead of 'audit_timestamp'/'model_name'), normalize it first.
    if "metadata" in report and "audit_timestamp" not in report:
        report = _normalize_disk_report(report)

    return FairnessReportResponse(**report)


@router.post("/run")
async def run_fairness_audit(
    request: FairnessRunRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger a new synthetic fairness stress test (runs in background).

    The audit evaluates model predictions across:
    - Age bands (<30, 30-45, 45-60, 60+)
    - Income quartiles (Q1-Q4)
    - Home ownership (rent, own, mortgage)

    Results will be available at GET /api/fairness/report once complete.
    """
    logger.info(f"Starting fairness audit for {request.model_name}...")

    async def run_audit():

        try:
            # Import here to avoid blocking imports
            import numpy as np
            import joblib
            from fairness_audit import run_fairness_audit, save_report

            # Load model
            model_paths = [
                f"{request.model_name}.joblib",
                "best_model.pkl",
                "xgboost_model.joblib",
                "lightgbm_model.joblib",
            ]

            model = None
            for path in model_paths:
                if Path(path).exists():
                    model = joblib.load(path)
                    logger.info(f"Loaded model from {path}")
                    break

            if model is None:
                logger.error("No model found for fairness audit")
                return

            # Generate synthetic test data for audit
            # In production, this would use real historical decisions
            np.random.seed(42)
            n_samples = request.sample_size

            # Create realistic synthetic data
            X = np.column_stack([
                np.random.randint(300, 850, n_samples),  # credit_score
                np.random.lognormal(10.5, 0.5, n_samples),  # annual_income
                np.random.uniform(10000, 500000, n_samples),  # loan_amount
                np.random.choice([12, 24, 36, 48, 60, 120, 180, 360], n_samples),  # loan_term
                np.random.uniform(0, 1, n_samples),  # dti_ratio
                np.random.uniform(0, 30, n_samples),  # employment_years
                np.random.randint(1, 15, n_samples),  # num_credit_lines
                np.random.randint(0, 5, n_samples),  # num_derogatory_marks
                np.random.uniform(0, 1, n_samples),  # credit_utilization
                np.random.uniform(0, 1, n_samples),  # late_payment_severity_score
                np.random.choice([0, 1, 2], n_samples),  # home_ownership
                np.random.choice([0, 1, 2, 3, 4], n_samples),  # purpose_encoded
                np.random.randint(0, 10, n_samples),  # num_late_payments
                np.random.lognormal(8, 1, n_samples),  # savings_balance
                np.random.uniform(1000, 5000, n_samples),  # monthly_expenses
            ])

            # Get predictions
            y_pred = model.predict(X)
            y_true = np.random.binomial(1, 0.3, n_samples)  # Synthetic true labels

            # Feature names
            feature_names = [
                "credit_score", "annual_income", "loan_amount", "loan_term",
                "dti_ratio", "employment_years", "num_credit_lines",
                "num_derogatory_marks", "credit_utilization",
                "late_payment_severity_score", "home_ownership", "purpose_encoded",
                "num_late_payments", "savings_balance", "monthly_expenses"
            ]

            # Run audit
            report = run_fairness_audit(
                model=model,
                X=X,
                y_true=y_true,
                feature_names=feature_names,
                model_name=request.model_name
            )

            # Save report
            report_path = save_report(report)
            logger.info(f"Fairness report saved to {report_path}")

            # Cache report
            set_cached_report({
                "audit_timestamp": report.audit_timestamp,
                "model_name": report.model_name,
                "overall_fairness_score": report.overall_fairness_score,
                "total_samples": report.total_samples,
                "group_metrics": report.group_metrics,
                "violations": report.violations,
                "mitigations": report.mitigations,
                "summary": report.summary,
                "recommendations": report.recommendations,
            })

            logger.info(f"Fairness audit complete. Score: {report.overall_fairness_score:.0f}/100")

        except Exception as e:
            logger.error(f"Fairness audit failed: {e}", exc_info=True)

    # Run in background
    background_tasks.add_task(run_audit)

    return {
        "status": "started",
        "message": f"Fairness audit initiated for {request.model_name}",
        "estimated_completion": "30-60 seconds",
    }


@router.get("/groups", response_model=FairnessGroupResponse)
async def get_group_metrics(
    feature: Optional[str] = Query(None, description="Filter by sensitive feature")
):
    """
    Get approval rates and metrics by demographic group.

    Optional query param 'feature' filters to specific feature:
    - age_bands
    - income_quartiles
    - home_ownership
    """
    report = get_cached_report()

    if report is None:
        return FairnessGroupResponse(
            groups=[],
            reference_group="unknown",
            disparity_summary={}
        )

    groups = report.get("group_metrics", [])

    # Filter by feature if specified
    if feature:
        groups = [g for g in groups if g["group_name"].startswith(feature)]

    # Calculate disparity summary
    approval_rates = [g["approval_rate"] for g in groups]
    disparity_summary = {
        "max_approval_rate": max(approval_rates) if approval_rates else 0,
        "min_approval_rate": min(approval_rates) if approval_rates else 0,
        "disparity_range": max(approval_rates) - min(approval_rates) if approval_rates else 0,
    }

    # Find reference group (highest approval rate)
    reference_group = ""
    if groups:
        reference_group = max(groups, key=lambda g: g["approval_rate"])["group_name"]

    return FairnessGroupResponse(
        groups=groups,
        reference_group=reference_group,
        disparity_summary=disparity_summary,
    )


@router.get("/violations")
async def get_violations():
    """Get detected fairness violations."""
    report = get_cached_report()

    if report is None:
        return {"violations": [], "count": 0}

    return {
        "violations": report.get("violations", []),
        "count": len(report.get("violations", [])),
        "has_critical": any(
            v.get("severity") == "high"
            for v in report.get("violations", [])
        ),
    }


@router.post("/export")
async def export_fairness_report():
    """
    Export fairness report as downloadable JSON.
    """
    report = get_cached_report()

    if report is None:
        raise HTTPException(status_code=404, detail="No report available")

    return JSONResponse(
        content=report,
        headers={
            "Content-Disposition": "attachment; filename=fairness_report.json"
        }
    )


@router.get("/status")
async def get_audit_status():
    """Get status of fairness auditing."""
    report = get_cached_report()

    return {
        "has_report": report is not None,
        "last_audit": _last_audit_time.isoformat() if _last_audit_time else None,
        "fairness_score": report.get("overall_fairness_score") if report else None,
        "violations_count": len(report.get("violations", [])) if report else 0,
    }
