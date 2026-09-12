import json
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "prithvi1029/medquad-medical-qa"
OUTPUT_PATH = Path("data/processed/medquad_full.json")

MAX_SAMPLES = 5000


def main():
    print("Loading MedQuAD from Hugging Face...")

    dataset = load_dataset(DATASET_NAME)

    full_dataset = dataset["train"]

    print(f"Total examples available: {len(full_dataset)}")

    # Keep only valid QA records
    full_dataset = full_dataset.filter(
        lambda x: (
            x["Question"] is not None
            and x["Answer"] is not None
            and str(x["Question"]).strip() != ""
            and str(x["Answer"]).strip() != ""
        )
    )

    print(f"Valid examples: {len(full_dataset)}")

    # Remove duplicated questions
    seen = set()
    unique_examples = []

    for example in full_dataset:
        question = str(example["Question"]).strip()

        normalized = question.lower()

        if normalized not in seen:
            seen.add(normalized)

            unique_examples.append(
                {
                    "question": question,
                    "answer": str(example["Answer"]).strip(),
                    "qtype": example.get("qtype"),
                }
            )

    print(f"Unique examples: {len(unique_examples)}")

    # Keep the project lightweight for local training/evaluation
    if len(unique_examples) > MAX_SAMPLES:
        unique_examples = unique_examples[:MAX_SAMPLES]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            unique_examples,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("MedQuAD dataset prepared successfully")
    print(f"Samples exported: {len(unique_examples)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
