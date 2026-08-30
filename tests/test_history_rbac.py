"""
Regression Tests — History RBAC & User-ID Consistency
=====================================================
Covers the P1 KeyError: 'id' fix and user-scoped history.

Tests:
1.  Loan Officer GET /history -> 200
2.  Prediction created by Officer A appears in Officer A history
3.  Officer B cannot see Officer A's prediction
4.  Officer A cannot see Officer B's prediction
5.  Underwriter sees organization-wide history
6.  Admin sees organization-wide history
7.  Unauthenticated GET /history -> 401
8.  Invalid token -> 401
9.  Prediction save stores authenticated canonical user_id
10. No newly-created prediction has NULL user_id
11. Admin delete audit action does not raise KeyError
12. Regression: old id/user_id mismatch would KeyError
"""

import pytest
import os
import sys
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient

# Path setup
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))
sys.path.insert(0, str(_ROOT))

from backend.api import app, init_db, save_decision
from backend.auth import init_users_table, create_token, hash_password, _token_store
from backend.audit_log import init_audit_table
from backend.database import DATABASE_PATH


# ─── Helpers ─────────────────────────────────────────────────────────────────

LOAN_PAYLOAD = {
    "income": 70000,
    "loan_amount": 250000,
    "interest_rate": 5.5,
    "loan_term": 30,
    "credit_score": 700,
    "employment_years": 5,
    "existing_loans": 1,
    "home_ownership": 1,
    "purpose_encoded": 0,
}


def _create_test_user(conn, username, password, role, full_name=""):
    """Insert a user directly into the DB for testing."""
    pw_hash, salt = hash_password(password)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO users (username, password_hash, password_salt, role, full_name, created_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'))",
        (username, pw_hash, salt, role, full_name),
    )
    conn.commit()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    return cursor.fetchone()[0]


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    # The rest of the setup uses the default DATABASE_PATH
    conn = sqlite3.connect(DATABASE_PATH)
    # Create two distinct loan officers, one underwriter, one admin
    officer_a_id = _create_test_user(conn, "officer_a", "testpass1", "loan_officer", "Officer A")
    officer_b_id = _create_test_user(conn, "officer_b", "testpass2", "loan_officer", "Officer B")
    uw_id = _create_test_user(conn, "test_uw", "uwpass", "underwriter", "Test Underwriter")
    admin_id = _create_test_user(conn, "test_admin", "adminpass", "admin", "Test Admin")
    conn.close()

    with TestClient(app) as c:
        # Perform logins once to avoid rate limits
        token_a = c.post("/auth/login", json={"username": "officer_a", "password": "testpass1"}).json()["data"]["token"]
        token_b = c.post("/auth/login", json={"username": "officer_b", "password": "testpass2"}).json()["data"]["token"]
        token_uw = c.post("/auth/login", json={"username": "test_uw", "password": "uwpass"}).json()["data"]["token"]
        token_admin = c.post("/auth/login", json={"username": "test_admin", "password": "adminpass"}).json()["data"]["token"]

        yield {
            "client": c,
            "officer_a_id": officer_a_id,
            "officer_b_id": officer_b_id,
            "uw_id": uw_id,
            "admin_id": admin_id,
            "token_a": token_a,
            "token_b": token_b,
            "token_uw": token_uw,
            "token_admin": token_admin,
        }

    # Cleanup is optional since we're using the main DB, but we clear token store.
    _token_store.clear()
    
    # Optional: cleanup test users from the database if desired
    # conn = sqlite3.connect(DATABASE_PATH)
    # cur = conn.cursor()
    # cur.execute("DELETE FROM users WHERE username IN ('officer_a', 'officer_b', 'test_uw', 'test_admin')")
    # conn.commit()
    # conn.close()


def _auth_header(token):
    return {"Authorization": f"Bearer {token}"}


# ─── Test 1: Loan Officer GET /history -> 200 ────────────────────────────────

def test_loan_officer_history_returns_200(client):
    token = client["token_a"]
    resp = client["client"].get("/history", headers=_auth_header(token))
    assert resp.status_code == 200


# ─── Test 2: Officer A prediction appears in Officer A history ───────────────

def test_officer_a_prediction_appears_in_own_history(client):
    token_a = client["token_a"]
    # Create prediction
    resp = client["client"].post("/analyze", json=LOAN_PAYLOAD, headers=_auth_header(token_a))
    assert resp.status_code == 200, f"Prediction failed: {resp.text}"

    # Fetch history
    resp = client["client"].get("/history", headers=_auth_header(token_a))
    assert resp.status_code == 200
    history = resp.json()["data"]
    assert len(history) >= 1, "Officer A should see at least their own prediction"
    # Verify user_id is set correctly
    for record in history:
        assert record["user_id"] == client["officer_a_id"], (
            f"Record user_id={record['user_id']} should be officer_a_id={client['officer_a_id']}"
        )


# ─── Test 3: Officer B cannot see Officer A's prediction ─────────────────────

def test_officer_b_cannot_see_officer_a_predictions(client):
    token_b = client["token_b"]
    resp = client["client"].get("/history", headers=_auth_header(token_b))
    assert resp.status_code == 200
    history = resp.json()["data"]
    # Officer B should not see any of Officer A's records
    for record in history:
        assert record["user_id"] != client["officer_a_id"], (
            "Officer B should not see Officer A's records"
        )


# ─── Test 4: Officer A cannot see Officer B's prediction ─────────────────────

def test_officer_a_cannot_see_officer_b_predictions(client):
    # First, Officer B creates a prediction
    token_b = client["token_b"]
    resp = client["client"].post("/analyze", json=LOAN_PAYLOAD, headers=_auth_header(token_b))
    assert resp.status_code == 200

    # Now Officer A checks history
    token_a = client["token_a"]
    resp = client["client"].get("/history", headers=_auth_header(token_a))
    assert resp.status_code == 200
    history = resp.json()["data"]
    for record in history:
        assert record["user_id"] != client["officer_b_id"], (
            "Officer A should not see Officer B's records"
        )


# ─── Test 5: Underwriter sees organization-wide history ──────────────────────

def test_underwriter_sees_all_history(client):
    token_uw = client["token_uw"]
    resp = client["client"].get("/history", headers=_auth_header(token_uw))
    assert resp.status_code == 200
    history = resp.json()["data"]
    # Underwriter should see records from both officers
    user_ids = {r["user_id"] for r in history if r["user_id"] is not None}
    assert client["officer_a_id"] in user_ids, "Underwriter should see Officer A's records"
    assert client["officer_b_id"] in user_ids, "Underwriter should see Officer B's records"


# ─── Test 6: Admin sees organization-wide history ────────────────────────────

def test_admin_sees_all_history(client):
    token_admin = client["token_admin"]
    resp = client["client"].get("/history", headers=_auth_header(token_admin))
    assert resp.status_code == 200
    history = resp.json()["data"]
    user_ids = {r["user_id"] for r in history if r["user_id"] is not None}
    assert client["officer_a_id"] in user_ids, "Admin should see Officer A's records"
    assert client["officer_b_id"] in user_ids, "Admin should see Officer B's records"


# ─── Test 7: Unauthenticated GET /history -> 401 ─────────────────────────────

def test_unauthenticated_history_returns_401(client):
    resp = client["client"].get("/history")
    assert resp.status_code == 401


# ─── Test 8: Invalid token -> 401 ────────────────────────────────────────────

def test_invalid_token_history_returns_401(client):
    resp = client["client"].get("/history", headers={"Authorization": "Bearer invalid-token-xyz"})
    assert resp.status_code == 401


# ─── Test 9: Prediction save stores canonical user_id ─────────────────────────

def test_prediction_stores_canonical_user_id(client):
    token_a = client["token_a"]
    resp = client["client"].post("/analyze", json=LOAN_PAYLOAD, headers=_auth_header(token_a))
    assert resp.status_code == 200

    # Verify directly in DB
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM decisions ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    assert row is not None
    assert row["user_id"] == client["officer_a_id"], (
        f"DB user_id={row['user_id']} should match officer_a_id={client['officer_a_id']}"
    )


# ─── Test 10: No newly-created prediction has NULL user_id ───────────────────

def test_authenticated_prediction_never_has_null_user_id(client):
    token_b = client["token_b"]
    resp = client["client"].post("/analyze", json=LOAN_PAYLOAD, headers=_auth_header(token_b))
    assert resp.status_code == 200

    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM decisions ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    assert row[0] is not None, "Authenticated prediction must have a non-NULL user_id"


# ─── Test 11: Admin delete audit does not raise KeyError ─────────────────────

def test_admin_delete_no_keyerror(client):
    """Admin delete_decision should not raise KeyError on admin['id']."""
    # Insert a dummy decision to delete
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO decisions (timestamp, income, loan_amount, credit_score, decision, risk_level, emi, user_id) "
        "VALUES (datetime('now'), 50000, 100000, 700, 'APPROVE', 'LOW', 3000, ?)",
        (client["officer_a_id"],)
    )
    conn.commit()
    decision_id = cur.lastrowid
    conn.close()

    token_admin = client["token_admin"]
    resp = client["client"].post(
        f"/api/data/delete/{decision_id}",
        headers=_auth_header(token_admin),
    )
    # Must not be 500 (KeyError). Should be 200 (success).
    assert resp.status_code == 200, f"Admin delete failed: {resp.text}"


# ─── Test 12: Regression — old id/user_id mismatch ──────────────────────────

def test_regression_token_dict_uses_user_id_not_id(client):
    """
    Regression test for the P1 KeyError: 'id' bug.

    The token store provides 'user_id', not 'id'. Any route that depends
    on get_current_user() and accesses user["id"] will raise KeyError.
    This test verifies the canonical key is 'user_id'.
    """
    token_a = client["token_a"]

    # Call an endpoint that returns the user object from the token
    resp = client["client"].get("/auth/me", headers=_auth_header(token_a))
    assert resp.status_code == 200, "Token should be valid"
    
    user_data = resp.json()["data"]
    assert "user_id" in user_data, "Response must contain 'user_id'"
    # Wait, /auth/me returns {"user_id": ..., "username": ..., "role": ...}
    assert user_data["user_id"] == client["officer_a_id"]


# ─── RBAC enforcement: Loan Officer blocked from admin routes ────────────────

def test_loan_officer_blocked_from_audit(client):
    token_a = client["token_a"]
    resp = client["client"].get("/audit", headers=_auth_header(token_a))
    assert resp.status_code == 403


def test_loan_officer_blocked_from_user_list(client):
    token_a = client["token_a"]
    resp = client["client"].get("/auth/users", headers=_auth_header(token_a))
    assert resp.status_code == 403
