def evaluate_anomalies(anomalies: list[dict]) -> dict:
    """Organize detected anomalies for human review."""

    if not anomalies:
        return {
            "status": "no_anomalies",
            "priority": "none",
            "total_anomalies": 0,
            "requires_human_review": False,
            "anomalies": [],
        }

    indicators = {
        anomaly["indicator"]
        for anomaly in anomalies
    }

    if len(indicators) >= 3:
        priority = "high"
    elif len(indicators) == 2:
        priority = "medium"
    else:
        priority = "low"

    return {
        "status": "review_required",
        "priority": priority,
        "total_anomalies": len(anomalies),
        "requires_human_review": True,
        "anomalies": anomalies,
    }
