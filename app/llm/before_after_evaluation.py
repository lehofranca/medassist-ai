import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "data/processed/medassist-qwen-smoke"


QUESTION = "What are the symptoms of hypertension?"


def generate_response(model, tokenizer):
    messages = [
        {
            "role": "user",
            "content": QUESTION,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


def main():
    print("=== BEFORE × AFTER EVALUATION ===")
    print(f"Question: {QUESTION}\n")

    # ---------------------------------------------------------
    # BEFORE: Base model without LoRA
    # ---------------------------------------------------------
    print("Loading base tokenizer...")

    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )

    base_model.eval()

    print("Generating BEFORE response...")

    before_response = generate_response(
        base_model,
        base_tokenizer,
    )

    print("\n=== BEFORE: BASE MODEL ===")
    print(before_response)

    # ---------------------------------------------------------
    # AFTER: Base model + LoRA adapter
    # ---------------------------------------------------------
    print("\nLoading tokenizer from adapter...")

    adapter_tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH
    )

    print("Loading second base model...")

    adapter_base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )

    print("Loading LoRA adapter...")

    adapted_model = PeftModel.from_pretrained(
        adapter_base_model,
        ADAPTER_PATH,
    )

    adapted_model.eval()

    print("Generating AFTER response...")

    after_response = generate_response(
        adapted_model,
        adapter_tokenizer,
    )

    print("\n=== AFTER: BASE + LORA ===")
    print(after_response)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n=== EVALUATION COMPLETED ===")
    print("Before: Base model")
    print("After: Base model + LoRA adapter")


if __name__ == "__main__":
    main()
