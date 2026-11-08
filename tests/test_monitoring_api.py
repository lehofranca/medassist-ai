from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_get_patient_anomalies():
    response = client.get("/patients/P001/anomalies")

    assert response.status_code == 200
    assert response.json()["total_anomalies"] == 3


def test_get_patient_alerts():
    response = client.get("/patients/P001/alerts")

    assert response.status_code == 200

    data = response.json()

    assert data["evaluation"]["priority"] == "high"
    assert data["alert"]["requires_human_review"] is True


def test_patient_without_anomalies():
    response = client.get("/patients/P002/anomalies")

    assert response.status_code == 200
    assert response.json()["total_anomalies"] == 0


def test_patient_without_alerts():
    response = client.get("/patients/P002/alerts")

    assert response.status_code == 200
    assert response.json()["alert"] is None


def test_unknown_patient_returns_404():
    response = client.get("/patients/P999/alerts")

    assert response.status_code == 404
    assert response.json()["detail"] == "Patient not found"
