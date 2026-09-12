import json
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "patients.json"


def get_patient_by_id(patient_id: str) -> dict | None:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        patients = json.load(file)

    for patient in patients:
        if patient["patient_id"] == patient_id:
            return patient

    return None
