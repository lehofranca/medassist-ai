from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.llm.assistant_service import ask_assistant


app = FastAPI(
    title="MedAssist AI",
    description="AI-assisted clinical information service",
    version="1.0.0",
)


class AssistantRequest(BaseModel):
    patient_id: str
    question: str


@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "MedAssist AI",
    }


@app.get("/patients/{patient_id}/context")
def get_patient_context(patient_id: str):
    from app.rag.clinical_context import build_clinical_context

    context = build_clinical_context(patient_id)

    if context is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found",
        )

    return {
        "patient_id": patient_id,
        "context": context,
    }


@app.post("/assistant/ask")
def assistant_ask(request: AssistantRequest):
    result = ask_assistant(
        request.patient_id,
        request.question,
    )

    if result["status"] == "patient_not_found":
        raise HTTPException(
            status_code=404,
            detail=result["message"],
        )

    return result
