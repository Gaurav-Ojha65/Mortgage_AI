"""
Mortgage AI — Audit Log System

Append-only, tamper-resistant audit trail for every significant action.
Stored in a separate table from application data.

Every entry records:
  - user_id, username, role
  - action (e.g. 'PREDICT', 'LOGIN', 'STATUS_CHANGE', 'EXPORT', 'USER_CREATE')
  - target_id (application ID, user ID, etc.)
  - timestamp
  - ip_address
  - before_value / after_value (for state changes)
  - metadata (JSON blob for extra context)

This table is APPEND-ONLY — no UPDATE or DELETE operations are exposed.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import Request

from database import DATABASE_PATH


# ─── Schema ───────────────────────────────────────────────────────────────────
def init_audit_table():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id INTEGER,
            username TEXT,
            user_role TEXT,
            action TEXT NOT NULL,
            target_type TEXT,
            target_id TEXT,
            ip_address TEXT,
            before_value TEXT,
            after_value TEXT,
            metadata TEXT,
            session_token_hash TEXT
        )
    """)
    # Index for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")
    conn.commit()
    conn.close()
    print("[Audit] Audit log table initialized")


# ─── Write (Append-Only) ─────────────────────────────────────────────────────
def log_action(
    action: str,
    user_id: Optional[int] = None,
    username: Optional[str] = None,
    user_role: Optional[str] = None,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    before_value: Optional[dict] = None,
    after_value: Optional[dict] = None,
    metadata: Optional[dict] = None,
):
    """
    Append a single audit entry. This is the ONLY write function —
    there is no update or delete.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_log
        (timestamp, user_id, username, user_role, action, target_type, target_id,
         ip_address, before_value, after_value, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        user_id,
        username,
        user_role,
        action,
        target_type,
        str(target_id) if target_id is not None else None,
        ip_address,
        json.dumps(before_value) if before_value else None,
        json.dumps(after_value) if after_value else None,
        json.dumps(metadata) if metadata else None,
    ))
    conn.commit()
    conn.close()


def log_from_request(
    request: Request,
    action: str,
    user: Optional[dict] = None,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    before_value: Optional[dict] = None,
    after_value: Optional[dict] = None,
    metadata: Optional[dict] = None,
):
    """Convenience wrapper that extracts IP from the FastAPI request."""
    ip = request.client.host if request.client else "unknown"
    log_action(
        action=action,
        user_id=user.get("user_id") if user else None,
        username=user.get("username") if user else None,
        user_role=user.get("role") if user else None,
        target_type=target_type,
        target_id=target_id,
        ip_address=ip,
        before_value=before_value,
        after_value=after_value,
        metadata=metadata,
    )


# ─── Read (Query Only) ───────────────────────────────────────────────────────
def get_audit_logs(
    limit: int = 50,
    offset: int = 0,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    target_type: Optional[str] = None,
) -> dict:
    """
    Query audit logs with filters. Returns paginated results.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    where_clauses = []
    params = []

    if user_id is not None:
        where_clauses.append("user_id = ?")
        params.append(user_id)
    if action:
        where_clauses.append("action = ?")
        params.append(action)
    if date_from:
        where_clauses.append("timestamp >= ?")
        params.append(date_from)
    if date_to:
        where_clauses.append("timestamp <= ?")
        params.append(date_to)
    if target_type:
        where_clauses.append("target_type = ?")
        params.append(target_type)

    where_sql = " AND ".join(where_clauses)
    if where_sql:
        where_sql = "WHERE " + where_sql

    # Count total
    cursor.execute(f"SELECT COUNT(*) FROM audit_log {where_sql}", params)
    total = cursor.fetchone()[0]

    # Fetch page
    cursor.execute(
        f"SELECT * FROM audit_log {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    )
    rows = cursor.fetchall()
    conn.close()

    entries = []
    for row in rows:
        entry = dict(row)
        # Parse JSON fields
        for field in ("before_value", "after_value", "metadata"):
            if entry.get(field):
                try:
                    entry[field] = json.loads(entry[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        entries.append(entry)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "entries": entries,
    }


def get_audit_stats() -> dict:
    """Summary statistics for the audit dashboard."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM audit_log")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT action, COUNT(*) as cnt FROM audit_log GROUP BY action ORDER BY cnt DESC")
    by_action = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM audit_log WHERE user_id IS NOT NULL")
    unique_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM audit_log WHERE timestamp >= date('now', '-1 day')")
    last_24h = cursor.fetchone()[0]

    conn.close()

    return {
        "total_entries": total,
        "by_action": by_action,
        "unique_users": unique_users,
        "last_24h": last_24h,
    }
