import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_PATH = PROJECT_ROOT / "app" / "rag"

sys.path.insert(0, str(RAG_PATH))

from retriever import load_rag, retrieve


TEST_CASES = [
    {
        "name": "Hypertension",
        "query": "What are the symptoms of hypertension?",
        "expected": "What are the symptoms of High Blood Pressure ?",
    },
    {
        "name": "High Blood Pressure",
        "query": "What are the symptoms of high blood pressure?",
        "expected": "What are the symptoms of High Blood Pressure ?",
    },
    {
        "name": "Pulmonary Hypertension",
        "query": "What are the symptoms of pulmonary hypertension?",
        "expected": "What are the symptoms of Pulmonary Hypertension ?",
    },
    {
        "name": "Kidney Disease",
        "query": "What are the symptoms of kidney disease?",
        "expected": "What are the symptoms of Kidney Disease ?",
    },
    {
        "name": "Diabetic Kidney Disease",
        "query": "What are the symptoms of diabetic kidney disease?",
        "expected": "What are the symptoms of Diabetic Kidney Disease ?",
    },
]


def test_entity_detection():
    from retriever import detect_entity

    assert detect_entity(
        "What are the symptoms of pulmonary hypertension?"
    ) == "pulmonary hypertension"

    assert detect_entity(
        "What are the symptoms of hypertension?"
    ) == "hypertension"

    assert detect_entity(
        "What are the symptoms of high blood pressure?"
    ) == "high blood pressure"


def test_retrieval_targets():
    rag = load_rag()

    for case in TEST_CASES:
        index, metadata, embedding_model = rag

        results = retrieve(
            case["query"],
            index,
            metadata,
            embedding_model,
            top_k=3,
        )

        assert results, f"No results returned for: {case['name']}"

        top_question = results[0]["document"]["question"]

        assert (
            top_question.strip().lower()
            == case["expected"].strip().lower()
        ), (
            f"\n{case['name']} failed."
            f"\nExpected: {case['expected']}"
            f"\nActual:   {top_question}"
        )


def test_specificity_penalty():
    from retriever import (
        calculate_specificity_penalty,
        detect_entity,
    )

    pulmonary_document = {
        "question": "What are the symptoms of Pulmonary Hypertension ?"
    }

    hypertension_document = {
        "question": "What are the symptoms of High Blood Pressure ?"
    }

    assert detect_entity(
        "What are the symptoms of pulmonary hypertension?"
    ) == "pulmonary hypertension"

    assert (
        calculate_specificity_penalty(
            "What are the symptoms of pulmonary hypertension?",
            pulmonary_document,
        )
        == 0.0
    )

    assert (
        calculate_specificity_penalty(
            "What are the symptoms of pulmonary hypertension?",
            hypertension_document,
        )
        == 0.25
    )
