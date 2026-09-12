import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

EVAL_PATH = Path("data/evaluation/medquad_eval.json")
ADAPTER_PATH = Path("data/processed/medassist-qwen-smoke")
OUTPUT_PATH = Path("data/evaluation/evaluation_results.json")


def load_evaluation_dataset():
    with open(EVAL_PATH, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction, reference):
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    if not prediction_tokens or not reference_tokens:
        return 0.0

    prediction_counts = {}
    reference_counts = {}

    for token in prediction_tokens:
        prediction_counts[token] = prediction_counts.get(token, 0) + 1

    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1

    common = 0

    for token in prediction_counts:
        if token in reference_counts:
            common += min(
                prediction_counts[token],
                reference_counts[token],
            )

    if common == 0:
        return 0.0

    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def generate_response(model, tokenizer, question):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs.shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


def load_base_model(tokenizer):
    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    model.eval()

    return model


def load_lora_model(tokenizer):
    print("Loading base model for LoRA...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.eval()

    return model


def main():
    print("=== MEDASSIST AI EVALUATION ===")
    print()

    dataset = load_evaluation_dataset()

    print(f"Evaluation samples: {len(dataset)}")
    print()

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # BEFORE - BASE MODEL
    # ---------------------------------------------------------

    print("=== BEFORE: BASE MODEL ===")

    base_model = load_base_model(tokenizer)

    before_results = []

    for index, example in enumerate(dataset, start=1):
        question = example["question"]
        reference = example["reference_answer"]

        print(f"[BEFORE {index}/{len(dataset)}]")
        print(f"Question: {question}")

        response = generate_response(
            base_model,
            tokenizer,
            question,
        )

        score = token_f1(
            response,
            reference,
        )

        print(f"F1: {score:.4f}")
        print()

        before_results.append(
            {
                "question": question,
                "reference_answer": reference,
                "response": response,
                "token_f1": round(score, 4),
            }
        )

    del base_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # AFTER - BASE MODEL + LORA
    # ---------------------------------------------------------

    print("=== AFTER: BASE MODEL + LORA ===")

    lora_model = load_lora_model(tokenizer)

    after_results = []

    for index, example in enumerate(dataset, start=1):
        question = example["question"]
        reference = example["reference_answer"]

        print(f"[AFTER {index}/{len(dataset)}]")
        print(f"Question: {question}")

        response = generate_response(
            lora_model,
            tokenizer,
            question,
        )

        score = token_f1(
            response,
            reference,
        )

        print(f"F1: {score:.4f}")
        print()

        after_results.append(
            {
                "question": question,
                "reference_answer": reference,
                "response": response,
                "token_f1": round(score, 4),
            }
        )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    before_average = sum(
        item["token_f1"] for item in before_results
    ) / len(before_results)

    after_average = sum(
        item["token_f1"] for item in after_results
    ) / len(after_results)

    improvement = after_average - before_average

    results = {
        "model": MODEL_NAME,
        "adapter": str(ADAPTER_PATH),
        "evaluation_samples": len(dataset),
        "metric": "token_f1",
        "before": {
            "model": "base",
            "average_token_f1": round(before_average, 4),
            "results": before_results,
        },
        "after": {
            "model": "base + LoRA",
            "average_token_f1": round(after_average, 4),
            "results": after_results,
        },
        "comparison": {
            "absolute_change": round(improvement, 4),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("========================================")
    print("FINAL EVALUATION")
    print("========================================")
    print(f"Before average F1: {before_average:.4f}")
    print(f"After average F1:  {after_average:.4f}")
    print(f"Change:            {improvement:+.4f}")
    print()
    print("Evaluation completed successfully.")
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
