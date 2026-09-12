from app.rag.clinical_context import build_clinical_context


def test_build_clinical_context():
    context = build_clinical_context("P001")

    assert context is not None
    assert "Patient ID: P001" in context
    assert "Age: 45" in context
    assert "Conditions: hypertension" in context
    assert "Medications: losartan" in context
    assert "Pending exams: blood pressure monitoring" in context
    assert "Complete Blood Count" in context
    assert "Within reference range" in context


def test_build_clinical_context_unknown_patient():
    context = build_clinical_context("P999")

    assert context is None
