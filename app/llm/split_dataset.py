import json
from pathlib import Path

from datasets import Dataset


INPUT_PATH = Path("data/processed/medquad_full.json")
OUTPUT_DIR = Path("data/processed/dataset_split")

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1


def main():
    print("Loading prepared MedQuAD dataset...")

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Total samples: {len(data)}")

    dataset = Dataset.from_list(data)

    # First split: 80% train / 20% temporary
    first_split = dataset.train_test_split(
        test_size=0.2,
        seed=42,
    )

    train_dataset = first_split["train"]
    temporary_dataset = first_split["test"]

    # Second split: 50% validation / 50% test
    second_split = temporary_dataset.train_test_split(
        test_size=0.5,
        seed=42,
    )

    validation_dataset = second_split["train"]
    test_dataset = second_split["test"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset.to_json(
        str(OUTPUT_DIR / "train.json"),
        force_ascii=False,
    )

    validation_dataset.to_json(
        str(OUTPUT_DIR / "validation.json"),
        force_ascii=False,
    )

    test_dataset.to_json(
        str(OUTPUT_DIR / "test.json"),
        force_ascii=False,
    )

    print()
    print("Dataset split completed")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(validation_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
