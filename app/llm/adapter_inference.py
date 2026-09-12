import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "data/processed/medassist-qwen-lora"


def main():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
    )

    model.eval()

    prompt = "What are the symptoms of hypertension?"

    messages = [
        {
            "role": "user",
            "content": prompt,
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

    print("Generating response...")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\n=== MODEL RESPONSE ===")
    print(response)


if __name__ == "__main__":
    main()
