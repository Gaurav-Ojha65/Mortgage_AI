import pytest
import os
from fastapi.testclient import TestClient
from backend.api import app, init_db
from backend.auth import init_users_table, get_all_users

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_cors_headers(client):
    response = client.options("/api/analyze", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

def test_auth_rejection(client):
    response = client.post("/auth/login", json={"username": "nonexistent", "password": "wrongpassword"})
    assert response.status_code == 401
    assert "Invalid username or password" in response.json()["detail"]

def test_rbac_history(client):
    # Try history without token
    response = client.get("/history")
    assert response.status_code == 401

def test_input_validation(client):
    # Test bounds on LoanApplication
    response = client.post("/analyze", json={
        "income": 6000000, # over 5M
        "loan_amount": 100000,
        "interest_rate": 5,
        "loan_term": 30,
        "credit_score": 750
    })
    assert response.status_code == 422
    assert "le" in str(response.json()) or "less than or equal to" in str(response.json())

def test_fairness_api(client):
    response = client.get("/api/fairness/groups")
    # Should return empty response, not 404
    assert response.status_code == 200
    data = response.json()
    assert data["groups"] == []

def test_analyze_does_not_reference_removed_risk_level(client):
    payload = {
        "income": 70000,
        "loan_amount": 250000,
        "interest_rate": 5.5,
        "loan_term": 30,
        "credit_score": 680,
        "employment_years": 4,
        "existing_loans": 1,
        "home_ownership": 1,
        "purpose_encoded": 0
    }
    response = client.post("/analyze", json=payload)
    # The exact previous failure was a 500 error due to NameError: name 'risk_level' is not defined.
    # The test must fail on the old code (which returned 500) and pass on the fixed code (200).
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["decision"] in ["APPROVE", "MANUAL_REVIEW", "REJECT"]
    assert "calibrated_default_probability" in data
    assert "raw_default_probability" in data
    assert data["model_version"] == "v3.1"
