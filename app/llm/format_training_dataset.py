import json
from pathlib import Path

from datasets import Dataset


DATASET_DIR = Path("data/processed/dataset_split")
OUTPUT_DIR = Path("data/training")


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def format_example(example):
    return {
        "text": (
            "<|im_start|>user\n"
            f"{example['question']}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{example['answer']}"
            "<|im_end|>"
        )
    }


def main():
    print("Loading dataset splits...")

    train_data = load_jsonl(DATASET_DIR / "train.json")
    validation_data = load_jsonl(DATASET_DIR / "validation.json")

    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)

    train_dataset = train_dataset.map(format_example)
    validation_dataset = validation_dataset.map(format_example)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset.to_json(
        str(OUTPUT_DIR / "train_formatted.json"),
        force_ascii=False,
    )

    validation_dataset.to_json(
        str(OUTPUT_DIR / "validation_formatted.json"),
        force_ascii=False,
    )

    print("Training datasets prepared")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
