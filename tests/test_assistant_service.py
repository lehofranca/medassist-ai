from app.llm.assistant_service import ask_assistant


def test_safe_question_returns_success():
    result = ask_assistant(
        "P001",
        "What are the patient pending exams?"
    )

    assert result["status"] == "success"
    assert result["patient_id"] == "P001"
    assert "blood pressure monitoring" in result["response"]


def test_prescription_requires_human_validation():
    result = ask_assistant(
        "P001",
        "Prescribe a medication for this patient"
    )

    assert result["status"] == "human_validation_required"
    assert result["patient_id"] == "P001"
    assert result["message"] != ""


def test_unknown_patient_returns_not_found():
    result = ask_assistant(
        "P999",
        "What are the patient pending exams?"
    )

    assert result["status"] == "patient_not_found"
    assert result["patient_id"] == "P999"
    assert result["message"] == "Patient not found"
