from app.llm import assistant_service


def test_e2e_safe_question_with_patient(monkeypatch):

    fake_context = (
        "Patient ID: P001\n"
        "Age: 45\n"
        "Conditions: hypertension\n"
    )

    class FakeChain:
        def invoke(self, payload):
            assert payload["context"] == fake_context
            assert payload["question"] == "What are the symptoms of hypertension?"

            return (
                "Hypertension may not cause noticeable symptoms. "
                "Regular blood pressure monitoring is important."
            )

    monkeypatch.setattr(
        assistant_service,
        "build_clinical_context",
        lambda patient_id: fake_context,
    )

    monkeypatch.setattr(
        assistant_service,
        "build_assistant_chain",
        lambda: FakeChain(),
    )

    result = assistant_service.ask_assistant(
        "P001",
        "What are the symptoms of hypertension?",
    )

    assert result["status"] == "success"
    assert result["patient_id"] == "P001"
    assert result["question"] == "What are the symptoms of hypertension?"
    assert "Hypertension" in result["response"]


def test_e2e_unknown_patient_does_not_call_llm(monkeypatch):

    llm_called = False

    def fake_chain():
        nonlocal llm_called
        llm_called = True
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(
        assistant_service,
        "build_clinical_context",
        lambda patient_id: None,
    )

    monkeypatch.setattr(
        assistant_service,
        "build_assistant_chain",
        fake_chain,
    )

    result = assistant_service.ask_assistant(
        "P999",
        "What are the symptoms of hypertension?",
    )

    assert result["status"] == "patient_not_found"
    assert result["patient_id"] == "P999"
    assert result["message"] == "Patient not found"
    assert llm_called is False


def test_e2e_unsafe_question_stops_before_context_and_llm(monkeypatch):

    context_called = False
    llm_called = False

    def fake_context(patient_id):
        nonlocal context_called
        context_called = True
        raise AssertionError("Clinical context should not be called")

    def fake_chain():
        nonlocal llm_called
        llm_called = True
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(
        assistant_service,
        "build_clinical_context",
        fake_context,
    )

    monkeypatch.setattr(
        assistant_service,
        "build_assistant_chain",
        fake_chain,
    )

    result = assistant_service.ask_assistant(
        "P001",
        "What medication should I take?",
    )

    assert result["status"] == "human_validation_required"
    assert result["patient_id"] == "P001"
    assert context_called is False
    assert llm_called is False
