from datetime import datetime, timezone


def generate_alert(evaluation: dict) -> dict | None:
    """Generate a structured alert for human review."""

    if not evaluation["requires_human_review"]:
        return None

    anomalies = evaluation["anomalies"]

    if not anomalies:
        return None

    patient_id = anomalies[0]["patient_id"]

    indicators = sorted({
        anomaly["indicator"]
        for anomaly in anomalies
    })

    return {
        "alert_id": f"ALT-{patient_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "patient_id": patient_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "priority": evaluation["priority"],
        "status": "pending_human_review",
        "total_anomalies": evaluation["total_anomalies"],
        "indicators": indicators,
        "requires_human_review": True,
        "message": (
            f"{len(indicators)} different indicators require "
            f"human review for patient {patient_id}."
        ),
        "anomalies": anomalies,
    }
