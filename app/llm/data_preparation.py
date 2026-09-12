import json
from pathlib import Path


RAW_DATASET = Path("data/raw/medquad_sample.json")
PROCESSED_DATASET = Path("data/processed/medquad_processed.json")


def load_dataset() -> list[dict]:
    with RAW_DATASET.open("r", encoding="utf-8") as file:
        return json.load(file)


def preprocess_dataset(data: list[dict]) -> list[dict]:
    processed = []

    for item in data:
        question = item["question"].strip()
        answer = item["answer"].strip()

        if not question or not answer:
            continue

        processed.append(
            {
                "question": question,
                "answer": answer
            }
        )

    return processed


def save_dataset(data: list[dict]) -> None:
    PROCESSED_DATASET.parent.mkdir(parents=True, exist_ok=True)

    with PROCESSED_DATASET.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def prepare_dataset() -> list[dict]:
    data = load_dataset()
    processed = preprocess_dataset(data)
    save_dataset(processed)

    return processed


if __name__ == "__main__":
    dataset = prepare_dataset()

    print("Processed dataset created")
    print("Samples:", len(dataset))
