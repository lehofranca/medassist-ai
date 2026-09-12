from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "running", "service": "MedAssist AI"}


def test_get_patient_context():
    response = client.get("/patients/P001/context")

    assert response.status_code == 200

    data = response.json()

    assert data["patient_id"] == "P001"
    assert "Patient ID: P001" in data["context"]
    assert "Conditions: hypertension" in data["context"]
    assert "Medications: losartan" in data["context"]


def test_get_unknown_patient_context():
    response = client.get("/patients/P999/context")

    assert response.status_code == 404
    assert response.json() == {"detail": "Patient not found"}
