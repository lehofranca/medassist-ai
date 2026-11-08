from app.anomaly.vitals_loader import (
    load_vital_signs,
    get_patient_vital_signs,
)


def test_load_vital_signs_returns_records():
    records = load_vital_signs()

    assert isinstance(records, list)
    assert len(records) == 9


def test_vital_signs_have_required_fields():
    records = load_vital_signs()

    required_fields = {
        "patient_id",
        "timestamp",
        "heart_rate",
        "systolic_pressure",
        "diastolic_pressure",
        "oxygen_saturation",
        "respiratory_rate",
    }

    for record in records:
        assert required_fields.issubset(record.keys())


def test_get_patient_vital_signs_returns_correct_patient():
    records = get_patient_vital_signs("P001")

    assert len(records) == 4
    assert all(record["patient_id"] == "P001" for record in records)


def test_get_patient_vital_signs_returns_empty_for_unknown_patient():
    records = get_patient_vital_signs("P999")

    assert records == []
