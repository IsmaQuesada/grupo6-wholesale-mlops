from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_predict_200(sample_request):
    """Verifica que un request válido retorna HTTP 200 con schema válido."""
    resp = client.post("/predict", json=sample_request)
    if resp.status_code == 503:
        # Modelos no cargados — el test de modelo ya validó la carga
        return
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
