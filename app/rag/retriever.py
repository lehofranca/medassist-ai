import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("data/processed/faiss_index/medquad.index")
METADATA_PATH = Path("data/processed/faiss_index/metadata.json")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 3
CANDIDATE_K = 30


MEDICAL_ALIASES = {
    "hypertension": [
        "high blood pressure",
        "hypertension",
    ],
    "high blood pressure": [
        "high blood pressure",
        "hypertension",
    ],
    "type 2 diabetes": [
        "type 2 diabetes",
        "diabetes",
    ],
    "diabetes": [
        "diabetes",
    ],
    "anemia": [
        "anemia",
    ],
}


QUESTION_TYPES = {
    "symptoms": [
        "symptom",
        "symptoms",
        "sign",
        "signs",
    ],
    "risk": [
        "risk",
        "at risk",
    ],
    "cause": [
        "cause",
        "causes",
        "why",
    ],
    "treatment": [
        "treatment",
        "treatments",
        "therapy",
    ],
    "diagnosis": [
        "diagnose",
        "diagnosis",
        "diagnostic",
    ],
    "prevention": [
        "prevent",
        "prevention",
    ],
    "information": [
        "what is",
        "what are",
        "information",
    ],
}


def load_index():
    return faiss.read_index(str(INDEX_PATH))


def load_metadata():
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def load_rag():
    index = load_index()
    metadata = load_metadata()
    embedding_model = load_embedding_model()

    return index, metadata, embedding_model


def normalize_text(text):
    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def detect_entity(query):
    normalized = normalize_text(query)

    # Specific medical entities must be evaluated
    # before generic terms because specific terms
    # may contain generic entity names.
    if "pulmonary hypertension" in normalized:
        return "pulmonary hypertension"

    if "high blood pressure" in normalized:
        return "high blood pressure"

    if "type 2 diabetes" in normalized:
        return "type 2 diabetes"

    if "hypertension" in normalized:
        return "hypertension"

    if "diabetes" in normalized:
        return "diabetes"

    if "anemia" in normalized:
        return "anemia"

    return None

def detect_question_type(query):
    normalized = normalize_text(query)

    for question_type, keywords in QUESTION_TYPES.items():

        for keyword in keywords:

            if keyword in normalized:
                return question_type

    return None


def calculate_entity_score(query, document):
    entity = detect_entity(query)

    if not entity:
        return 0.0

    document_question = normalize_text(
        document.get("question", "")
    )

    aliases = MEDICAL_ALIASES.get(
        entity,
        [entity],
    )

    # Strong match: exact target entity appears
    # in the document question.
    for alias in aliases:

        if alias in document_question:

            # When the user explicitly asks about
            # high blood pressure, prefer documents
            # explicitly referring to high blood pressure.
            if entity == "high blood pressure":
                if "high blood pressure" in document_question:
                    return 1.0

                if "pulmonary hypertension" in document_question:
                    return 0.0

            return 1.0

    return 0.0


def calculate_question_type_score(query, document):
    question_type = detect_question_type(query)

    if not question_type:
        return 0.0

    document_qtype = normalize_text(
        document.get("qtype", "")
    )

    document_question = normalize_text(
        document.get("question", "")
    )

    # Prefer the structured qtype from the dataset.
    if question_type == "symptoms":
        if document_qtype == "symptoms":
            return 1.0

        if "symptoms" in document_question:
            return 0.8

    if question_type == "risk":
        if document_qtype == "susceptibility":
            return 1.0

        if "risk" in document_question:
            return 0.8

    if question_type == "cause":
        if document_qtype == "causes":
            return 1.0

        if "cause" in document_question:
            return 0.8

    if question_type == "treatment":
        if document_qtype == "treatment":
            return 1.0

        if "treatment" in document_question:
            return 0.8

    if question_type == "diagnosis":
        if document_qtype == "diagnosis":
            return 1.0

        if "diagnos" in document_question:
            return 0.8

    if question_type == "prevention":
        if document_qtype == "prevention":
            return 1.0

        if "prevent" in document_question:
            return 0.8

    return 0.0


def calculate_specificity_penalty(query, document):
    entity = detect_entity(query)

    document_question = normalize_text(
        document.get("question", "")
    )

    # Generic hypertension should prefer the direct
    # High Blood Pressure document.
    if entity == "hypertension":

        if "pulmonary hypertension" in document_question:
            return 0.40

        if (
            "high blood pressure and kidney disease"
            in document_question
        ):
            return 0.20

    # Explicit high blood pressure query.
    if entity == "high blood pressure":

        if "pulmonary hypertension" in document_question:
            return 0.40

        if (
            "high blood pressure and kidney disease"
            in document_question
        ):
            return 0.20

    # Explicit pulmonary hypertension query.
    # The pulmonary hypertension document is the
    # specific and correct source, so it must NOT
    # receive the generic hypertension penalty.
    if entity == "pulmonary hypertension":

        if "pulmonary hypertension" in document_question:
            return 0.0

        if "symptoms of high blood pressure" in document_question:
            return 0.25

        if (
            "high blood pressure and kidney disease"
            in document_question
        ):
            return 0.25

    return 0.0

def rerank_results(query, candidates):

    reranked = []

    for result in candidates:

        document = result["document"]

        semantic_score = result["score"]

        entity_score = calculate_entity_score(
            query,
            document,
        )

        question_type_score = calculate_question_type_score(
            query,
            document,
        )

        specificity_penalty = calculate_specificity_penalty(
            query,
            document,
        )

        final_score = (
            semantic_score * 0.55
            + entity_score * 0.30
            + question_type_score * 0.15
            - specificity_penalty
        )

        reranked.append(
            {
                "score": final_score,
                "semantic_score": semantic_score,
                "entity_score": entity_score,
                "question_type_score": question_type_score,
                "specificity_penalty": specificity_penalty,
                "document": document,
            }
        )

    reranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return reranked


def retrieve(
    query,
    index,
    metadata,
    embedding_model,
    top_k=DEFAULT_TOP_K,
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_embedding,
        CANDIDATE_K,
    )

    candidates = []

    for score, index_id in zip(
        scores[0],
        indices[0],
    ):

        if index_id < 0:
            continue

        candidates.append(
            {
                "score": float(score),
                "document": metadata[index_id],
            }
        )

    reranked = rerank_results(
        query,
        candidates,
    )

    return reranked[:top_k]


def main():

    print("=== MedAssist AI - RAG Retriever Test ===")

    index = load_index()
    metadata = load_metadata()
    embedding_model = load_embedding_model()

    print(
        f"Indexed documents: {index.ntotal}"
    )

    query = input(
        "\nQuery: "
    ).strip()

    if not query:
        print("Query cannot be empty.")
        return

    print(
        f"\nDetected entity: {detect_entity(query)}"
    )

    print(
        f"Detected question type: "
        f"{detect_question_type(query)}"
    )

    results = retrieve(
        query,
        index,
        metadata,
        embedding_model,
    )

    print(
        "\nTop retrieved documents after reranking:"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        document = result["document"]

        print("\n" + "=" * 80)
        print(f"RESULT {rank}")

        print(
            f"Final score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Semantic score: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"Entity score: "
            f"{result['entity_score']:.4f}"
        )

        print(
            f"Question type score: "
            f"{result['question_type_score']:.4f}"
        )

        print(
            f"Specificity penalty: "
            f"{result['specificity_penalty']:.4f}"
        )

        print(
            f"Question: "
            f"{document['question']}"
        )

        print(
            f"Type: "
            f"{document.get('qtype')}"
        )

        print(
            f"Answer: "
            f"{document['answer']}"
        )


if __name__ == "__main__":
    main()
