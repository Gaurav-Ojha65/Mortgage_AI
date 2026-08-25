"""
Analytics Router — Fairness metrics endpoint.
Returns approval rates grouped by age band and region for bias monitoring.
"""
from datetime import datetime
import sqlite3
from fastapi import APIRouter, Depends, HTTPException

from auth import require_role
from database import DATABASE_PATH

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Canonical age-band buckets
AGE_BANDS = ["18-30", "31-45", "46-60", "60+"]


@router.get("/fairness")
def get_fairness_report(admin: dict = Depends(require_role("admin"))):
    """
    Approval rates grouped by age band and region.
    Returns structured data ready for Chart.js bar charts.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ── By Age Band ───────────────────────────────────────────────────────
    cursor.execute("""
        SELECT age_band,
               COUNT(*) AS total,
               SUM(CASE WHEN decision IN ('APPROVE', 'APPROVED') THEN 1 ELSE 0 END) AS approved
        FROM decisions
        WHERE age_band IS NOT NULL
        GROUP BY age_band
    """)
    raw_age = {row["age_band"]: dict(row) for row in cursor.fetchall()}

    # Ensure every canonical bucket exists (even if count is 0)
    by_age = []
    for band in AGE_BANDS:
        entry = raw_age.get(band, {"age_band": band, "total": 0, "approved": 0})
        entry["approval_rate"] = round(
            entry["approved"] / entry["total"], 4
        ) if entry["total"] > 0 else 0
        by_age.append(entry)

    # ── By Region ─────────────────────────────────────────────────────────
    cursor.execute("""
        SELECT region,
               COUNT(*) AS total,
               SUM(CASE WHEN decision IN ('APPROVE', 'APPROVED') THEN 1 ELSE 0 END) AS approved
        FROM decisions
        WHERE region IS NOT NULL
        GROUP BY region
    """)
    by_region = []
    for row in cursor.fetchall():
        entry = dict(row)
        entry["approval_rate"] = round(
            entry["approved"] / entry["total"], 4
        ) if entry["total"] > 0 else 0
        by_region.append(entry)

    conn.close()

    return {
        "success": True,
        "data": {
            "by_age": by_age,
            "by_region": by_region,
            "timestamp": datetime.now().isoformat(),
        },
    }
