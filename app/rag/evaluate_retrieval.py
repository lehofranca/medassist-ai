import json
from pathlib import Path

from retriever import retrieve, load_index, load_metadata, load_embedding_model


EVALUATION_PATH = Path("data/evaluation/medquad_eval.json")
OUTPUT_PATH = Path("data/evaluation/rag_retrieval_results.json")


def main():
    print("=== MedAssist AI - RAG RETRIEVAL EVALUATION ===")

    with open(EVALUATION_PATH, "r", encoding="utf-8-sig") as file:
        questions = json.load(file)

    print(f"Evaluation samples: {len(questions)}")

    print("\nLoading FAISS index...")
    index = load_index()

    print("Loading metadata...")
    metadata = load_metadata()

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print(f"\nIndexed documents: {index.ntotal}")

    results = []

    for i, item in enumerate(questions, start=1):

        question = item["question"]

        print("\n" + "=" * 80)
        print(f"QUESTION {i}")
        print(question)

        retrieved = retrieve(
            question,
            index,
            metadata,
            embedding_model,
        )

        question_results = []

        for rank, result in enumerate(retrieved, start=1):

            document = result["document"]

            print(f"\nTOP {rank}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Question: {document['question']}")
            print(f"Type: {document.get('qtype')}")

            question_results.append(
                {
                    "rank": rank,
                    "similarity": float(result["score"]),
                    "question": document["question"],
                    "qtype": document.get("qtype"),
                }
            )

        results.append(
            {
                "question": question,
                "reference_answer": item.get("reference_answer"),
                "results": question_results,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== RETRIEVAL EVALUATION COMPLETED ===")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
