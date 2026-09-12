from app.safety.safety_rules import validate_query


def test_safe_query_is_allowed():
    result = validate_query(
        "What are the patient pending exams?"
    )

    assert result["allowed"] is True
    assert result["requires_human_validation"] is False


def test_prescription_query_requires_human_validation():
    result = validate_query(
        "Prescribe a medication for this patient"
    )

    assert result["allowed"] is False
    assert result["requires_human_validation"] is True
    assert "human validation" in result["reason"].lower()
