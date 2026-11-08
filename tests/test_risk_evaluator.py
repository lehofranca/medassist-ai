from app.anomaly.risk_evaluator import evaluate_anomalies


def test_three_different_indicators_generate_high_priority():
    anomalies = [
        {"indicator": "heart_rate"},
        {"indicator": "oxygen_saturation"},
        {"indicator": "respiratory_rate"},
    ]

    result = evaluate_anomalies(anomalies)

    assert result["priority"] == "high"
    assert result["total_anomalies"] == 3
    assert result["requires_human_review"] is True


def test_two_indicators_generate_medium_priority():
    anomalies = [
        {"indicator": "heart_rate"},
        {"indicator": "oxygen_saturation"},
    ]

    result = evaluate_anomalies(anomalies)

    assert result["priority"] == "medium"


def test_one_indicator_generates_low_priority():
    anomalies = [
        {"indicator": "heart_rate"},
    ]

    result = evaluate_anomalies(anomalies)

    assert result["priority"] == "low"


def test_empty_anomalies_return_no_priority():
    result = evaluate_anomalies([])

    assert result["status"] == "no_anomalies"
    assert result["priority"] == "none"
    assert result["total_anomalies"] == 0
    assert result["requires_human_review"] is False


def test_evaluation_preserves_original_anomalies():
    anomalies = [
        {"indicator": "heart_rate", "observed_value": 125},
    ]

    result = evaluate_anomalies(anomalies)

    assert result["anomalies"] == anomalies
