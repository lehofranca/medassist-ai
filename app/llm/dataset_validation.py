import json
from pathlib import Path


TRAINING_DATASET = Path("data/training/medquad_training.json")
REPORT_PATH = Path("data/processed/validation_report.json")


def load_dataset() -> list[dict]:
    with TRAINING_DATASET.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_dataset(data: list[dict]) -> dict:
    errors = []
    warnings = []

    seen_questions = set()
    duplicates = 0

    for index, item in enumerate(data):
        if "messages" not in item:
            errors.append(f"Sample {index}: missing messages")
            continue

        messages = item["messages"]

        if len(messages) != 2:
            errors.append(f"Sample {index}: expected exactly 2 messages")
            continue

        user_message = messages[0]
        assistant_message = messages[1]

        if user_message.get("role") != "user":
            errors.append(f"Sample {index}: first message must be user")

        if assistant_message.get("role") != "assistant":
            errors.append(f"Sample {index}: second message must be assistant")

        question = user_message.get("content", "").strip()
        answer = assistant_message.get("content", "").strip()

        if not question:
            errors.append(f"Sample {index}: empty question")

        if not answer:
            errors.append(f"Sample {index}: empty answer")

        if len(answer) < 20:
            warnings.append(f"Sample {index}: answer is very short")

        if question in seen_questions:
            duplicates += 1

        seen_questions.add(question)

    report = {
        "dataset": str(TRAINING_DATASET),
        "total_samples": len(data),
        "unique_questions": len(seen_questions),
        "duplicates": duplicates,
        "errors": errors,
        "warnings": warnings,
        "valid": len(errors) == 0
    }

    return report


def save_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    dataset = load_dataset()
    report = validate_dataset(dataset)
    save_report(report)

    print("Dataset validation completed")
    print("Samples:", report["total_samples"])
    print("Unique questions:", report["unique_questions"])
    print("Duplicates:", report["duplicates"])
    print("Errors:", len(report["errors"]))
    print("Warnings:", len(report["warnings"]))
    print("Valid:", report["valid"])
