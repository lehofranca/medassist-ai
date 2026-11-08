from app.alerts.alert_engine import generate_alert


def test_generate_alert_for_high_priority():
    evaluation = {
        "status": "review_required",
        "priority": "high",
        "total_anomalies": 3,
        "requires_human_review": True,
        "anomalies": [
            {
                "patient_id": "P001",
                "indicator": "heart_rate",
                "observed_value": 125,
            },
            {
                "patient_id": "P001",
                "indicator": "oxygen_saturation",
                "observed_value": 91,
            },
            {
                "patient_id": "P001",
                "indicator": "respiratory_rate",
                "observed_value": 25,
            },
        ],
    }

    alert = generate_alert(evaluation)

    assert alert is not None
    assert alert["patient_id"] == "P001"
    assert alert["priority"] == "high"
    assert alert["status"] == "pending_human_review"
    assert alert["requires_human_review"] is True


def test_alert_contains_all_detected_indicators():
    evaluation = {
        "priority": "high",
        "total_anomalies": 3,
        "requires_human_review": True,
        "anomalies": [
            {"patient_id": "P001", "indicator": "heart_rate"},
            {"patient_id": "P001", "indicator": "oxygen_saturation"},
            {"patient_id": "P001", "indicator": "respiratory_rate"},
        ],
    }

    alert = generate_alert(evaluation)

    assert set(alert["indicators"]) == {
        "heart_rate",
        "oxygen_saturation",
        "respiratory_rate",
    }


def test_no_alert_is_generated_without_human_review():
    evaluation = {
        "priority": "none",
        "total_anomalies": 0,
        "requires_human_review": False,
        "anomalies": [],
    }

    alert = generate_alert(evaluation)

    assert alert is None


def test_alert_contains_unique_identifier_and_timestamp():
    evaluation = {
        "priority": "low",
        "total_anomalies": 1,
        "requires_human_review": True,
        "anomalies": [
            {"patient_id": "P001", "indicator": "heart_rate"},
        ],
    }

    alert = generate_alert(evaluation)

    assert alert["alert_id"].startswith("ALT-P001-")
    assert alert["created_at"]
    assert alert["message"]


def test_alert_preserves_original_anomalies():
    anomalies = [
        {
            "patient_id": "P001",
            "indicator": "heart_rate",
            "observed_value": 125,
        }
    ]

    evaluation = {
        "priority": "low",
        "total_anomalies": 1,
        "requires_human_review": True,
        "anomalies": anomalies,
    }

    alert = generate_alert(evaluation)

    assert alert["anomalies"] == anomalies
