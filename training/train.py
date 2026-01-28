from typing import cast

import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer

from config import (
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_OUTPUT_DIR,
    LORA_R,
    LORA_TARGET_MODULES,
    NUM_EPOCHS,
    STUDENT_MODEL_ID,
    USE_BF16,
    USE_FP16,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)

from .dataset import load_dataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(device: torch.device):
    print(f"Loading model: {STUDENT_MODEL_ID}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_MODEL_ID,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if USE_FP16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        trust_remote_code=True,
        dtype=dtype,
        device_map={"": device},
    )

    return model, tokenizer


def create_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def create_training_args(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=USE_FP16,
        bf16=USE_BF16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_pin_memory=False,
    )


def train():
    device = get_device()
    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(device)

    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config, mixed=False)
    model.print_trainable_parameters()

    train_dataset, val_dataset = load_dataset()

    training_args = create_training_args(str(LORA_OUTPUT_DIR))

    trainer = SFTTrainer(
        model=cast(PeftModel, model),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving adapter to {LORA_OUTPUT_DIR}")
    trainer.save_model(str(LORA_OUTPUT_DIR))
    tokenizer.save_pretrained(str(LORA_OUTPUT_DIR))

    print("Training complete!")


def main():
    train()


if __name__ == "__main__":
    main()
