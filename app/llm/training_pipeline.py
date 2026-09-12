import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_PATH = Path("data/training/train_formatted.json")
VALIDATION_PATH = Path("data/training/validation_formatted.json")

OUTPUT_DIR = Path("data/processed/medassist-qwen-lora")

# ---------------------------------------------------------
# Local CPU training configuration
# ---------------------------------------------------------

MAX_TRAIN_SAMPLES = 20
MAX_VALIDATION_SAMPLES = 10
MAX_STEPS = 5
MAX_LENGTH = 256


def load_jsonl(path: Path, max_samples: int | None = None) -> Dataset:

    with open(path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    if max_samples is not None:
        data = data[:max_samples]

    return Dataset.from_list(data)


def main():

    print("=== MedAssist AI LoRA Training ===")

    print("\nTraining configuration:")
    print(f"Model: {MODEL_NAME}")
    print(f"Train samples: {MAX_TRAIN_SAMPLES}")
    print(f"Validation samples: {MAX_VALIDATION_SAMPLES}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print("Device: CPU")
    print()

    print("Loading datasets...")

    train_dataset = load_jsonl(
        TRAIN_PATH,
        MAX_TRAIN_SAMPLES,
    )

    validation_dataset = load_jsonl(
        VALIDATION_PATH,
        MAX_VALIDATION_SAMPLES,
    )

    print(f"Train samples loaded: {len(train_dataset)}")
    print(f"Validation samples loaded: {len(validation_dataset)}")

    print(f"\nLoading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading model: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )

    training_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),

        # Controlled local training
        max_steps=MAX_STEPS,
        num_train_epochs=1,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=1,

        learning_rate=2e-4,

        logging_steps=1,

        eval_strategy="steps",
        eval_steps=5,

        save_strategy="steps",
        save_steps=5,
        save_total_limit=1,

        report_to="none",

        max_length=MAX_LENGTH,
        packing=False,

        use_cpu=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=training_config,
        peft_config=lora_config,
    )

    print("\nStarting controlled LoRA training...")
    print("This run is intentionally small for CPU validation.\n")

    train_result = trainer.train()

    print("\n=== TRAINING RESULTS ===")
    print(f"Train loss: {train_result.training_loss:.4f}")
    print(f"Training steps: {train_result.global_step}")

    print("\nRunning validation...")

    evaluation_result = trainer.evaluate()

    print("\n=== VALIDATION RESULTS ===")

    for key, value in evaluation_result.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

        else:
            print(f"{key}: {value}")

    print("\nSaving LoRA adapter...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\n=== TRAINING COMPLETED ===")
    print(f"Adapter: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
