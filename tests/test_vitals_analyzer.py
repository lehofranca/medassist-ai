from app.anomaly.vitals_analyzer import analyze_patient_vitals


def test_detects_anomalies_for_patient_p001():
    anomalies = analyze_patient_vitals("P001")

    assert len(anomalies) == 3

    indicators = {
        anomaly["indicator"]
        for anomaly in anomalies
    }

    assert indicators == {
        "heart_rate",
        "oxygen_saturation",
        "respiratory_rate",
    }


def test_patient_p002_has_no_anomalies():
    anomalies = analyze_patient_vitals("P002")

    assert anomalies == []


def test_patient_p003_has_no_anomalies():
    anomalies = analyze_patient_vitals("P003")

    assert anomalies == []


def test_unknown_patient_returns_no_anomalies():
    anomalies = analyze_patient_vitals("P999")

    assert anomalies == []


def test_anomaly_contains_required_information():
    anomalies = analyze_patient_vitals("P001")

    for anomaly in anomalies:
        assert anomaly["patient_id"] == "P001"
        assert anomaly["timestamp"]
        assert anomaly["indicator"]
        assert anomaly["observed_value"] is not None
        assert anomaly["reason"]
        assert anomaly["status"] == "anomaly_detected"
