from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.llm.assistant_service import ask_assistant
from app.anomaly.vitals_analyzer import analyze_patient_vitals
from app.anomaly.risk_evaluator import evaluate_anomalies
from app.alerts.alert_engine import generate_alert


app = FastAPI(
    title="MedAssist AI",
    description="AI-assisted clinical information and patient monitoring service",
    version="2.0.0",
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


@app.get("/patients/{patient_id}/anomalies")
def get_patient_anomalies(patient_id: str):
    from app.rag.patient_service import get_patient_by_id

    patient = get_patient_by_id(patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found",
        )

    anomalies = analyze_patient_vitals(patient_id)

    return {
        "patient_id": patient_id,
        "total_anomalies": len(anomalies),
        "anomalies": anomalies,
    }


@app.get("/patients/{patient_id}/alerts")
def get_patient_alerts(patient_id: str):
    from app.rag.patient_service import get_patient_by_id

    patient = get_patient_by_id(patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found",
        )

    anomalies = analyze_patient_vitals(patient_id)
    evaluation = evaluate_anomalies(anomalies)
    alert = generate_alert(evaluation)

    return {
        "patient_id": patient_id,
        "evaluation": evaluation,
        "alert": alert,
    }
