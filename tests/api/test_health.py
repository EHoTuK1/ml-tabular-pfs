from fastapi.testclient import TestClient


def test_health_with_model(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_health_without_model(client_no_model: TestClient) -> None:
    r = client_no_model.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": False}
