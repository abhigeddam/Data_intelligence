"""
Tests for FastAPI REST backend endpoints.
"""

from fastapi.testclient import TestClient
from data_intelligence.api.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"


def test_api_samples_list():
    res = client.get("/api/v1/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 3


def test_api_sample_audit():
    res = client.post("/api/v1/audit/sample/retail")
    assert res.status_code == 200
    audit = res.json()
    assert "audit_id" in audit
    assert audit["business_type"] == "retail"
    assert "executive_summary" in audit
    assert len(audit["model_results"]) >= 3


def test_api_query():
    res = client.post(
        "/api/v1/query",
        json={"question": "What is our total revenue?", "sample_dataset": "retail"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert data["metric_value"] is not None
