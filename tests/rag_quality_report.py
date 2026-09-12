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


MIN_SCORE = 0.45


def run_evaluation():
    print("\n" + "=" * 70)
    print("MEDASSIST AI - RAG QUALITY REPORT")
    print("=" * 70)

    index, metadata, embedding_model = load_rag()

    total = len(EVALUATION_CASES)

    entity_hits = 0
    top1_hits = 0
    top3_hits = 0
    scores = []

    for case in EVALUATION_CASES:

        entity = detect_entity(case["query"])

        results = retrieve(
            case["query"],
            index,
            metadata,
            embedding_model,
            top_k=3,
        )

        if not results:
            print(f"\n{case['name']}: NO RESULTS")
            continue

        top1 = results[0]
        score = float(top1["score"])

        scores.append(score)

        if entity == case["expected_entity"]:
            entity_hits += 1

        retrieved_questions = [
            result["document"]["question"]
            for result in results
        ]

        if top1["document"]["question"] == case["expected_question"]:
            top1_hits += 1

        if case["expected_question"] in retrieved_questions:
            top3_hits += 1

        print("\n" + "-" * 70)
        print(f"Case:             {case['name']}")
        print(f"Entity:           {entity}")
        print(f"Top-1 score:      {score:.4f}")
        print(f"Top-1 result:     {top1['document']['question']}")
        print(
            f"Entity match:     "
            f"{'PASS' if entity == case['expected_entity'] else 'FAIL'}"
        )
        print(
            f"Top-1 match:      "
            f"{'PASS' if top1['document']['question'] == case['expected_question'] else 'FAIL'}"
        )
        print(
            f"Top-3 contains:   "
            f"{'PASS' if case['expected_question'] in retrieved_questions else 'FAIL'}"
        )

    entity_accuracy = entity_hits / total if total else 0
    top1_accuracy = top1_hits / total if total else 0
    top3_accuracy = top3_hits / total if total else 0

    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0
    )

    minimum_score = (
        min(scores)
        if scores
        else 0
    )

    quality_gate = (
        entity_accuracy >= 1.0
        and top1_accuracy >= 1.0
        and top3_accuracy >= 1.0
        and minimum_score >= MIN_SCORE
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Evaluation cases:          {total}")
    print(f"Entity accuracy:           {entity_accuracy:.2%}")
    print(f"Top-1 retrieval accuracy:  {top1_accuracy:.2%}")
    print(f"Top-3 retrieval accuracy:  {top3_accuracy:.2%}")
    print(f"Average retrieval score:   {average_score:.4f}")
    print(f"Minimum retrieval score:   {minimum_score:.4f}")
    print(f"Minimum score threshold:   {MIN_SCORE:.2f}")

    print("\n" + "=" * 70)

    if quality_gate:
        print("QUALITY GATE: PASS")
    else:
        print("QUALITY GATE: FAIL")

    print("=" * 70)

    return quality_gate


if __name__ == "__main__":
    success = run_evaluation()

    if not success:
        raise SystemExit(1)
