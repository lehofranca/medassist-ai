from app.llm.assistant_chain import build_assistant_chain
from app.rag.clinical_context import build_clinical_context
from app.safety.safety_rules import validate_query


def ask_assistant(patient_id: str, question: str) -> dict:
    safety_result = validate_query(question)

    if not safety_result["allowed"]:
        return {
            "status": "human_validation_required",
            "patient_id": patient_id,
            "question": question,
            "message": safety_result["reason"]
        }

    context = build_clinical_context(patient_id)

    if context is None:
        return {
            "status": "patient_not_found",
            "patient_id": patient_id,
            "question": question,
            "message": "Patient not found"
        }

    chain = build_assistant_chain()

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return {
        "status": "success",
        "patient_id": patient_id,
        "question": question,
        "response": response
    }
