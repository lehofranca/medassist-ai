from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset_path: Path = Path("data/training/medquad_training.json")
    output_dir: Path = Path("data/processed/medassist-llama")
    
    # Fine-tuning eficiente com LoRA/PEFT
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Configurações iniciais para treinamento
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512


TRAINING_CONFIG = TrainingConfig()


if __name__ == "__main__":
    config = TRAINING_CONFIG

    print("Training configuration loaded")
    print("Model:", config.model_name)
    print("Dataset:", config.dataset_path)
    print("Output:", config.output_dir)
    print("LoRA rank:", config.lora_r)
    print("LoRA alpha:", config.lora_alpha)
    print("Epochs:", config.num_train_epochs)
