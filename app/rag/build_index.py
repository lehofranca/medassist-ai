from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATASET_PATH = Path("data/processed/medquad_full.json")
INDEX_DIR = Path("data/processed/faiss_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("=== MedAssist AI - Building RAG Index ===")

    print(f"Loading dataset: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Documents loaded: {len(data)}")

    texts = [
        f"Question: {item['question']}\nAnswer: {item['answer']}"
        for item in data
    ]

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype("float32")

    print(f"Embedding shape: {embeddings.shape}")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "medquad.index"
    metadata_path = INDEX_DIR / "metadata.json"

    faiss.write_index(index, str(index_path))

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== RAG INDEX CREATED ===")
    print(f"Vectors: {index.ntotal}")
    print(f"Dimension: {dimension}")
    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
