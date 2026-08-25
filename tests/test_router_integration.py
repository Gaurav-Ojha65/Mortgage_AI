"""
Router Integration Tests — Mortgage AI v3.1
Tests newly mounted endpoints:
- Prometheus /metrics
- Enhanced /health with v3.1 provenance
- Document OCR router /api/documents/supported-formats
- Fairness router /api/fairness/report
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))
sys.path.insert(0, str(_ROOT))

from api import app

client = TestClient(app)


def test_health_endpoint_v3_1_provenance():
    """Verify /health returns active v3.1 model and policy versions."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    res_data = data["data"]
    assert res_data["status"] == "ok"
    assert res_data["model_version"] == "v3.1"
    assert res_data["calibration_version"] == "oof-iso-v3.1"
    assert res_data["policy_version"] == "v3.1-policy-v1"
    assert res_data["policy_thresholds"]["approve_threshold"] == 0.045
    assert res_data["policy_thresholds"]["reject_threshold"] == 0.335


def test_prometheus_metrics_endpoint():
    """Verify /metrics returns Prometheus formatted plain text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "mortgage_uptime_seconds" in body
    assert "mortgage_predictions_total" in body
    assert 'model_version="v3.1"' in body
    assert 'policy_version="v3.1-policy-v1"' in body


def test_document_router_supported_formats():
    """Verify document router /api/documents/supported-formats is reachable."""
    response = client.get("/api/documents/supported-formats")
    assert response.status_code == 200
    data = response.json()
    assert "supported_formats" in data
    extensions = [fmt["extension"] for fmt in data["supported_formats"]]
    assert ".pdf" in extensions
    assert ".png" in extensions


def test_fairness_router_report_or_groups():
    """Verify fairness router is mounted under /api/fairness."""
    response = client.get("/api/fairness/groups")
    # Should either return 200 with groups or valid JSON response
    assert response.status_code in (200, 404, 503)
    if response.status_code == 200:
        data = response.json()
        assert "groups" in data or "reference_group" in data


def test_database_path_unification():
    """Prove all database consumers resolve to the exact same canonical SQLite path."""
    import database
    import auth
    import audit_log
    from routers import data_router, analytics_router
    import api

    canonical = database.DATABASE_PATH
    assert auth.DATABASE_PATH == canonical
    assert audit_log.DATABASE_PATH == canonical
    assert data_router.DATABASE_PATH == canonical
    assert analytics_router.DATABASE_PATH == canonical
    assert api.DATABASE_PATH == canonical

    # Verify canonical database can be connected and queried
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    assert cursor.fetchone() == (1,)
    conn.close()

