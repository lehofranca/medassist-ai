import re
import sys
from pathlib import Path

import torch
from langchain_core.runnables import RunnableLambda
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_PATH = PROJECT_ROOT / "app" / "rag"

if str(RAG_PATH) not in sys.path:
    sys.path.insert(0, str(RAG_PATH))


from retriever import load_rag, retrieve
from rag_qa import build_context, generate_answer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TOP_K = 3
MIN_RETRIEVAL_SCORE = 0.45


_rag_components = None
_model = None
_tokenizer = None


def _load_components():
    global _rag_components
    global _model
    global _tokenizer

    if _rag_components is None:
        print("Loading RAG components...")
        _rag_components = load_rag()

    if _tokenizer is None:
        print("Loading tokenizer...")

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

    if _model is None:
        print("Loading Qwen model...")

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
        )

        _model.eval()

    return _rag_components, _model, _tokenizer


def _extract_context_field(context: str, field: str):
    """
    Extract a structured field from the clinical patient context.
    """

    pattern = rf"^{re.escape(field)}:\s*(.+)$"

    match = re.search(
        pattern,
        context,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if match:
        return match.group(1).strip()

    return None


def _patient_context_answer(question: str, context: str):
    """
    Answer deterministic patient-context questions directly.

    These questions should not depend on LLM generation because
    the requested information already exists as structured data.
    """

    normalized = question.lower().strip()

    if (
        "pending exam" in normalized
        or "pending exams" in normalized
        or "pending test" in normalized
        or "pending tests" in normalized
    ):
        value = _extract_context_field(
            context,
            "Pending exams",
        )

        if value:
            return f"Pending exams: {value}"

    if (
        "medication" in normalized
        or "medications" in normalized
    ):
        value = _extract_context_field(
            context,
            "Medications",
        )

        if value:
            return f"Medications: {value}"

    if (
        "condition" in normalized
        or "conditions" in normalized
    ):
        value = _extract_context_field(
            context,
            "Conditions",
        )

        if value:
            return f"Conditions: {value}"

    if (
        "patient id" in normalized
        or "patient identifier" in normalized
    ):
        value = _extract_context_field(
            context,
            "Patient ID",
        )

        if value:
            return f"Patient ID: {value}"

    return None


def _is_patient_context_question(question: str) -> bool:
    normalized = question.lower().strip()

    patient_keywords = [
        "patient",
        "pending exam",
        "pending exams",
        "pending test",
        "pending tests",
        "recent exam",
        "recent exams",
        "medication",
        "medications",
        "condition",
        "conditions",
        "patient id",
    ]

    return any(
        keyword in normalized
        for keyword in patient_keywords
    )


def _generate_response(inputs: dict) -> str:
    question = inputs["question"]
    patient_context = inputs["context"]

    # Deterministic extraction for structured patient data.
    direct_answer = _patient_context_answer(
        question,
        patient_context,
    )

    if direct_answer:
        return direct_answer

    # Load RAG and LLM components for medical questions.
    rag, model, tokenizer = _load_components()

    index, metadata, embedding_model = rag

    results = retrieve(
        question,
        index,
        metadata,
        embedding_model,
        top_k=TOP_K,
    )

    if (
        results
        and results[0]["score"] >= MIN_RETRIEVAL_SCORE
    ):
        rag_context, _ = build_context(results)

        context = (
            "PATIENT CLINICAL CONTEXT:\n"
            f"{patient_context}\n\n"
            "RETRIEVED MEDICAL EVIDENCE:\n"
            f"{rag_context}"
        )

    else:
        context = (
            "PATIENT CLINICAL CONTEXT:\n"
            f"{patient_context}"
        )

    return generate_answer(
        model,
        tokenizer,
        question,
        context,
    )


def build_assistant_chain():
    return RunnableLambda(_generate_response)
