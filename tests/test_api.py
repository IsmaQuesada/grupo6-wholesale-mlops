import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_predict_200(sample_request):
    """Verifica que un request válido retorna HTTP 200 con schema válido."""
    resp = client.post("/predict", json=sample_request)
    if resp.status_code == 503:
        pytest.skip("Model artifacts not loaded")
    assert resp.status_code == 200

    data = resp.json()
    assert "cluster" in data
    assert data["cluster"] in [0, 1, 2]
    assert "distance_to_centroid" in data
    assert isinstance(data["distance_to_centroid"], float)
    assert "model_version" in data


def test_predict_campos_requeridos():
    """Verifica que campos faltantes retornan HTTP 422 (validation error)."""
    resp = client.post("/predict", json={"Fresh": 12669})
    assert resp.status_code == 422


def test_predict_gasto_negativo(sample_request):
    """Verifica que gasto negativo retorna HTTP 422."""
    sample_request["Fresh"] = -100
    resp = client.post("/predict", json=sample_request)
    assert resp.status_code == 422


# ------------------------------------------------------------------
# Tests de endpoints de soporte
# ------------------------------------------------------------------

def test_health_200():
    """GET /health retorna 200 con status 'ok'."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


def test_metrics_200():
    """GET /metrics retorna 200 con las 6 claves de métricas."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    claves = {"latency_avg_ms", "throughput_req_per_sec", "error_rate_pct",
              "availability_pct", "total_requests", "uptime_seconds"}
    assert set(data.keys()) == claves


def test_root_200():
    """GET / retorna 200 con info básica de la API."""
    resp = client.get("/")
    assert resp.status_code == 200
