import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = Path("data/processed/medassist-qwen-lora")
EVAL_PATH = Path("data/evaluation/medquad_eval.json")


def load_eval_dataset():
    with open(EVAL_PATH, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def generate(model, tokenizer, question):
    prompt = (
        "<|im_start|>user\n"
        f"{question}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()


def main():

    print("=== MEDASSIST AI EVALUATION ===")

    dataset = load_eval_dataset()

    print(f"Evaluation samples: {len(dataset)}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading BASE model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    print("\nLoading LORA model...")
    lora_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    lora_model = PeftModel.from_pretrained(
        lora_base,
        ADAPTER_PATH,
    )

    results = []

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        reference = item["reference_answer"]

        print("\n" + "=" * 80)
        print(f"QUESTION {index}/{len(dataset)}")
        print(question)

        print("\nGenerating BASE...")
        before = generate(
            base_model,
            tokenizer,
            question,
        )

        print("\nGenerating BASE + LORA...")
        after = generate(
            lora_model,
            tokenizer,
            question,
        )

        print("\n--- REFERENCE ---")
        print(reference)

        print("\n--- BASE ---")
        print(before)

        print("\n--- BASE + LORA ---")
        print(after)

        results.append(
            {
                "question": question,
                "reference_answer": reference,
                "base_response": before,
                "lora_response": after,
            }
        )

    output_path = Path(
        "data/evaluation/medassist_evaluation_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("=== EVALUATION COMPLETED ===")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
