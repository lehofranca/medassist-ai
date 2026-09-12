from app.rag.patient_service import get_patient_by_id


def build_clinical_context(patient_id: str) -> str | None:
    patient = get_patient_by_id(patient_id)

    if patient is None:
        return None

    conditions = ", ".join(patient["conditions"]) or "None"
    medications = ", ".join(patient["medications"]) or "None"
    pending_exams = ", ".join(patient["pending_exams"]) or "None"

    recent_exams = []

    for exam in patient["recent_exams"]:
        recent_exams.append(
            f"- {exam['name']}: "
            f"status={exam['status']}, "
            f"result={exam['result']}"
        )

    exams_context = "\n".join(recent_exams) or "- None"

    return (
        f"Patient ID: {patient['patient_id']}\n"
        f"Age: {patient['age']}\n"
        f"Sex: {patient['sex']}\n"
        f"Conditions: {conditions}\n"
        f"Medications: {medications}\n"
        f"Pending exams: {pending_exams}\n"
        f"Recent exams:\n{exams_context}"
    )
