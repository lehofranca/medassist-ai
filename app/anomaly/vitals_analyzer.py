from app.anomaly.vitals_loader import get_patient_vital_signs


# Experimental thresholds for synthetic dataset validation.
# These values are not medical diagnostic criteria.
ANOMALY_THRESHOLDS = {
    "heart_rate": {
        "min": 50,
        "max": 110,
    },
    "systolic_pressure": {
        "min": 90,
        "max": 160,
    },
    "diastolic_pressure": {
        "min": 60,
        "max": 100,
    },
    "oxygen_saturation": {
        "min": 94,
        "max": 100,
    },
    "respiratory_rate": {
        "min": 12,
        "max": 22,
    },
}


def analyze_patient_vitals(patient_id: str) -> list[dict]:
    """Identify measurements outside experimental thresholds."""

    records = get_patient_vital_signs(patient_id)
    anomalies = []

    for record in records:
        for indicator, limits in ANOMALY_THRESHOLDS.items():
            value = record[indicator]

            if value < limits["min"] or value > limits["max"]:
                anomalies.append({
                    "patient_id": patient_id,
                    "timestamp": record["timestamp"],
                    "indicator": indicator,
                    "observed_value": value,
                    "expected_min": limits["min"],
                    "expected_max": limits["max"],
                    "reason": (
                        f"{indicator} value {value} is outside "
                        f"the experimental range "
                        f"({limits['min']}-{limits['max']})."
                    ),
                    "status": "anomaly_detected",
                })

    return anomalies
