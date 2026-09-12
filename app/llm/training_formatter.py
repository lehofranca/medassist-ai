import json
from pathlib import Path


PROCESSED_DATASET = Path("data/processed/medquad_processed.json")
TRAINING_DATASET = Path("data/training/medquad_training.json")


def load_processed_dataset() -> list[dict]:
    with PROCESSED_DATASET.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_for_training(data: list[dict]) -> list[dict]:
    training_data = []

    for item in data:
        training_data.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": item["question"]
                    },
                    {
                        "role": "assistant",
                        "content": item["answer"]
                    }
                ]
            }
        )

    return training_data


def save_training_dataset(data: list[dict]) -> None:
    TRAINING_DATASET.parent.mkdir(parents=True, exist_ok=True)

    with TRAINING_DATASET.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


def prepare_training_dataset() -> list[dict]:
    data = load_processed_dataset()
    training_data = format_for_training(data)
    save_training_dataset(training_data)

    return training_data


if __name__ == "__main__":
    dataset = prepare_training_dataset()

    print("Training dataset created")
    print("Samples:", len(dataset))
