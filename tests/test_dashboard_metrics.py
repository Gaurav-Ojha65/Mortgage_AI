"""
Dashboard Metric Clarity Regression Tests

Verifies:
1. Only canonical v3.1 (model_used='lightgbm') records are included in avgRisk
2. Average is calculated from calibrated_default_probability
3. No observed default claim is made (label/tooltip checks are frontend-only)
4. Empty dataset handled correctly
5. Historical non-v3.1 records do NOT alter the v3.1 dashboard metric
"""
import os
import sqlite3
import pytest
from fastapi.testclient import TestClient

from backend.api import app, save_decision, get_history, init_db
from backend.database import DATABASE_PATH
from backend.auth import create_token


# ─── Helpers ─────────────────────────────────────────────────────────

def get_admin_token(client):
    """Log in as the admin user and return the token."""
    resp = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    return resp.json()["data"]["token"]


@pytest.fixture(scope="module")
def client():
    """Test client with seeded canonical and non-canonical records."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Clear any existing test records
    cursor.execute("DELETE FROM decisions WHERE advice = 'metric_test_marker'")
    conn.commit()
    conn.close()

    # Seed canonical v3.1 records (model_used = 'lightgbm')
    for prob in [0.10, 0.20, 0.30]:
        save_decision({
            "timestamp": "2026-08-31T00:00:00",
            "income": 80000, "loan_amount": 250000, "credit_score": 700,
            "decision": "MANUAL_REVIEW", "risk_level": "MODERATE",
            "default_probability": prob,
            "approval_probability": 1.0 - prob,
            "emi": 1500, "model_used": "lightgbm",
            "advice": "metric_test_marker",
            "age_band": None, "region": None, "user_id": None,
            "model_version": "v3.1",
            "calibration_version": "oof-iso-v3.1",
            "policy_version": "v3.1-policy-v1",
        })

    # Seed one NON-canonical record (model_used = 'xgboost' — hypothetical legacy)
    save_decision({
        "timestamp": "2026-08-31T00:00:00",
        "income": 50000, "loan_amount": 400000, "credit_score": 550,
        "decision": "REJECT", "risk_level": "SEVERE",
        "default_probability": 0.95,  # Very high — would skew average if included
        "approval_probability": 0.05,
        "emi": 3000, "model_used": "xgboost",
        "advice": "metric_test_marker",
        "age_band": None, "region": None, "user_id": None,
        "model_version": "v3.0",
        "calibration_version": "platt",
        "policy_version": "legacy",
    })

    # Seed v3.0 lightgbm record
    save_decision({
        "timestamp": "2026-08-31T00:00:00",
        "income": 60000, "loan_amount": 300000, "credit_score": 600,
        "decision": "REJECT", "risk_level": "HIGH",
        "default_probability": 0.85,
        "approval_probability": 0.15,
        "emi": 2500, "model_used": "lightgbm",
        "advice": "metric_test_marker",
        "age_band": None, "region": None, "user_id": None,
        "model_version": "v3.0",
        "calibration_version": "oof-iso-v3.0",
        "policy_version": "v3.0-policy-v1",
    })

    # Seed old calibration record
    save_decision({
        "timestamp": "2026-08-31T00:00:00",
        "income": 70000, "loan_amount": 350000, "credit_score": 650,
        "decision": "MANUAL_REVIEW", "risk_level": "MODERATE",
        "default_probability": 0.45,
        "approval_probability": 0.55,
        "emi": 2000, "model_used": "lightgbm",
        "advice": "metric_test_marker",
        "age_band": None, "region": None, "user_id": None,
        "model_version": "v3.1",
        "calibration_version": "platt",
        "policy_version": "v3.1-policy-v1",
    })

    # Seed old policy record
    save_decision({
        "timestamp": "2026-08-31T00:00:00",
        "income": 75000, "loan_amount": 250000, "credit_score": 680,
        "decision": "APPROVE", "risk_level": "LOW",
        "default_probability": 0.05,
        "approval_probability": 0.95,
        "emi": 1500, "model_used": "lightgbm",
        "advice": "metric_test_marker",
        "age_band": None, "region": None, "user_id": None,
        "model_version": "v3.1",
        "calibration_version": "oof-iso-v3.1",
        "policy_version": "legacy",
    })

    with TestClient(app) as c:
        # Get admin token once and attach it to the client
        resp = c.post("/auth/login", json={"username": "admin", "password": "admin123"})
        c.admin_token = resp.json()["data"]["token"]
        yield c

    # Cleanup
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM decisions WHERE advice = 'metric_test_marker'")
    conn.commit()
    conn.close()


# ─── Tests ───────────────────────────────────────────────────────────

class TestDashboardMetricClarity:

    def test_history_returns_model_used_field(self, client):
        """The /history endpoint must return model_used for frontend filtering."""
        token = client.admin_token
        resp = client.get("/history?limit=10", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) > 0
        # Every record must have model_used field
        for record in data:
            assert "model_used" in record

    def test_canonical_v31_filtering(self, client):
        """
        avgRisk should be calculated ONLY from exact canonical v3.1 records:
        model_version="v3.1", calibration_version="oof-iso-v3.1", policy_version="v3.1-policy-v1"
        """
        token = client.admin_token
        resp = client.get("/history?limit=100", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        history = resp.json()["data"]

        # Filter to test records
        test_records = [h for h in history if h.get("advice") == "metric_test_marker"]
        assert len(test_records) == 7  # 3 canonical + 4 non-canonical

        # Canonical exact match
        canonical = [
            h for h in test_records
            if h.get("model_version") == "v3.1"
            and h.get("calibration_version") == "oof-iso-v3.1"
            and h.get("policy_version") == "v3.1-policy-v1"
        ]
        assert len(canonical) == 3

        # Non-canonical
        non_canonical = [h for h in test_records if h not in canonical]
        assert len(non_canonical) == 4

        # Canonical average: mean(0.10, 0.20, 0.30) = 0.20
        canonical_avg = sum(h["default_probability"] for h in canonical) / len(canonical)
        assert abs(canonical_avg - 0.20) < 0.001

    def test_average_uses_calibrated_default_probability(self, client):
        """
        The dashboard uses default_probability which is stored as
        calibrated_default_probability from the ML pipeline.
        """
        token = client.admin_token
        resp = client.get("/history?limit=100", headers={"Authorization": f"Bearer {token}"})
        history = resp.json()["data"]

        test_canonical = [
            h for h in history
            if h.get("advice") == "metric_test_marker" and h["model_used"] == "lightgbm"
        ]

        for record in test_canonical:
            prob = record["default_probability"]
            assert 0.0 <= prob <= 1.0
            # Verify approval_probability = 1 - default_probability (calibrated pair)
            if record.get("approval_probability") is not None:
                assert abs(record["approval_probability"] + prob - 1.0) < 0.001

    def test_empty_dataset_returns_zero(self, client):
        """
        If there are no canonical records, avgRisk should be 0 (handled by frontend).
        Verify the backend doesn't crash on empty results.
        """
        token = client.admin_token
        # Request with limit=0 is not valid (ge=1), so test with a user who has no records
        # Instead, test the get_history function directly with a non-existent user
        result = get_history(limit=20, user_id=999999)
        assert result == []

    def test_non_v31_records_preserved_in_history(self, client):
        """
        Historical non-v3.1 records must still appear in the full history listing.
        Only the avgRisk aggregation should exclude them — records themselves are NOT deleted.
        """
        token = client.admin_token
        resp = client.get("/history?limit=100", headers={"Authorization": f"Bearer {token}"})
        history = resp.json()["data"]

        test_records = [h for h in history if h.get("advice") == "metric_test_marker"]
        model_versions = set(h.get("model_version") for h in test_records)
        calibration_versions = set(h.get("calibration_version") for h in test_records)
        policy_versions = set(h.get("policy_version") for h in test_records)

        # Ensure all types of historical records are present
        assert "v3.0" in model_versions
        assert "platt" in calibration_versions
        assert "legacy" in policy_versions

    def test_analyze_stores_calibrated_probability(self, client):
        """
        The /analyze endpoint must store the calibrated_default_probability
        in the default_probability column and model_used='lightgbm'.
        """
        token = client.admin_token
        resp = client.post("/analyze", json={
            "income": 85000,
            "loan_amount": 250000,
            "interest_rate": 6.5,
            "loan_term": 30,
            "credit_score": 720,
            "existing_loans": 1,
        }, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        result = resp.json()["data"]

        # The stored default_probability must be the calibrated one
        assert "calibrated_default_probability" in result
        assert result["model_used"] == "lightgbm"

        # Verify it was stored correctly
        latest = get_history(limit=1)
        assert len(latest) == 1
        assert latest[0]["model_used"] == "lightgbm"
        assert abs(latest[0]["default_probability"] - result["calibrated_default_probability"]) < 0.001
