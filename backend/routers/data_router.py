from datetime import datetime, timedelta
import sqlite3
from fastapi import APIRouter, Depends, HTTPException, Query

from auth import require_role
from audit_log import log_action
from database import DATABASE_PATH
router = APIRouter()

@router.delete("/api/data/applicant/{applicant_id}")
def delete_applicant(applicant_id: int, admin: dict = Depends(require_role("admin"))):
    """Admin-only endpoint to delete an applicant record."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM decisions WHERE id = ?", (applicant_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Applicant record not found")
        
    timestamp = datetime.now().isoformat()
    log_action(
        action="DATA_DELETION", 
        user_id=admin["user_id"], 
        metadata={"applicant_id": applicant_id, "reason": "Admin deleted applicant record"}
    )
    
    return {
        "deleted_id": applicant_id,
        "timestamp": timestamp,
        "message": "Applicant record deleted successfully"
    }


@router.get("/api/data/old-count")
def count_old_records(
    older_than_days: int = Query(..., ge=1, le=365),
    admin: dict = Depends(require_role("admin")),
):
    """Return the count of decision records older than the given threshold."""
    cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM decisions WHERE timestamp < ?", (cutoff,))
    count = cursor.fetchone()[0]
    conn.close()
    return {
        "older_than_days": older_than_days,
        "cutoff_date": cutoff,
        "record_count": count,
    }


@router.delete("/api/data/purge")
def purge_old_records(
    older_than_days: int = Query(..., ge=1, le=365),
    admin: dict = Depends(require_role("admin")),
):
    """Permanently delete decision records older than the given threshold."""
    cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM decisions WHERE timestamp < ?", (cutoff,))
    purged = cursor.rowcount
    conn.commit()
    conn.close()

    log_action(
        action="DATA_PURGE",
        user_id=admin["user_id"],
        metadata={
            "older_than_days": older_than_days,
            "cutoff_date": cutoff,
            "records_purged": purged,
        },
    )

    return {
        "purged_count": purged,
        "older_than_days": older_than_days,
        "cutoff_date": cutoff,
        "timestamp": datetime.now().isoformat(),
        "message": f"Purged {purged} record(s) older than {older_than_days} days",
    }
