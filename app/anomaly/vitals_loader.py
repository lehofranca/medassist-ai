import json
from pathlib import Path


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "vitals"
    / "vital_signs.json"
)


REQUIRED_FIELDS = {
    "patient_id",
    "timestamp",
    "heart_rate",
    "systolic_pressure",
    "diastolic_pressure",
    "oxygen_saturation",
    "respiratory_rate",
}


def load_vital_signs() -> list[dict]:
    """Load and validate vital signs from the JSON dataset."""

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Dataset must contain a list.")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object.")

        missing_fields = REQUIRED_FIELDS - record.keys()

        if missing_fields:
            raise ValueError(
                f"Record {index} is missing: {sorted(missing_fields)}"
            )

    return records


def get_patient_vital_signs(patient_id: str) -> list[dict]:
    """Retrieve vital signs for a specific patient."""

    records = load_vital_signs()

    return [
        record
        for record in records
        if record["patient_id"] == patient_id
    ]