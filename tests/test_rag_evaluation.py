from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_PATH = PROJECT_ROOT / "app" / "rag"

if str(RAG_PATH) not in sys.path:
    sys.path.insert(0, str(RAG_PATH))

from retriever import load_rag, retrieve, detect_entity


EVALUATION_CASES = [
    {
        "name": "Hypertension",
        "query": "What are the symptoms of hypertension?",
        "expected_entity": "hypertension",
        "expected_question": "What are the symptoms of High Blood Pressure ?",
    },
    {
        "name": "High Blood Pressure",
        "query": "What are the symptoms of high blood pressure?",
        "expected_entity": "high blood pressure",
        "expected_question": "What are the symptoms of High Blood Pressure ?",
    },
    {
        "name": "Pulmonary Hypertension",
        "query": "What are the symptoms of pulmonary hypertension?",
        "expected_entity": "pulmonary hypertension",
        "expected_question": "What are the symptoms of Pulmonary Hypertension ?",
    },
]


def test_rag_evaluation_suite():
    index, metadata, embedding_model = load_rag()

    passed = 0

    print("\n")
    print("=" * 80)
    print("MEDASSIST AI - RAG EVALUATION")
    print("=" * 80)

    for case in EVALUATION_CASES:

        print("\n" + "-" * 80)
        print(f"CASE: {case['name']}")
        print(f"QUERY: {case['query']}")

        entity = detect_entity(case["query"])

        assert entity == case["expected_entity"], (
            f"\nEntity mismatch"
            f"\nExpected: {case['expected_entity']}"
            f"\nActual:   {entity}"
        )

        results = retrieve(
            case["query"],
            index,
            metadata,
            embedding_model,
            top_k=3,
        )

        assert results, "No documents retrieved"

        top_result = results[0]
        top_question = top_result["document"]["question"]

        assert top_question == case["expected_question"], (
            f"\nTop-1 retrieval mismatch"
            f"\nExpected: {case['expected_question']}"
            f"\nActual:   {top_question}"
        )

        assert top_result["score"] >= 0.45, (
            f"\nRetrieval score below threshold"
            f"\nScore: {top_result['score']:.4f}"
        )

        print(f"Entity:       {entity}")
        print(f"Top-1 score:  {top_result['score']:.4f}")
        print(f"Top-1 result: {top_question}")
        print("Result:       PASS")

        passed += 1

    print("\n" + "=" * 80)
    print(f"RAG EVALUATION: {passed}/{len(EVALUATION_CASES)} PASSED")
    print("=" * 80)

    assert passed == len(EVALUATION_CASES)
