from app.rag.patient_service import get_patient_by_id


def test_get_existing_patient():
    patient = get_patient_by_id("P001")

    assert patient is not None
    assert patient["patient_id"] == "P001"
    assert patient["age"] == 45
    assert patient["sex"] == "F"
    assert "hypertension" in patient["conditions"]
    assert "losartan" in patient["medications"]


def test_get_unknown_patient():
    patient = get_patient_by_id("P999")

    assert patient is None
