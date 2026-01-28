import json
from pathlib import Path
from typing import Optional

from datasets import Dataset

from config import TRAINING_DATA_PATH, VALIDATION_DATA_PATH


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def format_for_sft(sample: dict) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Convert this process description to a Mermaid flowchart:\n\n{sample['description']}",
            },
            {
                "role": "assistant",
                "content": sample["mermaid"],
            },
        ]
    }


def load_dataset(
    train_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
) -> tuple[Dataset, Dataset]:
    train_path = train_path or TRAINING_DATA_PATH
    val_path = val_path or VALIDATION_DATA_PATH

    train_samples = load_jsonl(train_path)
    val_samples = load_jsonl(val_path)

    print(f"Loaded {len(train_samples)} training samples")
    print(f"Loaded {len(val_samples)} validation samples")

    train_formatted = [format_for_sft(s) for s in train_samples]
    val_formatted = [format_for_sft(s) for s in val_samples]

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)

    return train_dataset, val_dataset


def preview_sample(sample: dict) -> None:
    formatted = format_for_sft(sample)
    print("=" * 60)
    print("USER:")
    print(formatted["messages"][0]["content"])
    print("-" * 60)
    print("ASSISTANT:")
    print(formatted["messages"][1]["content"])
    print("=" * 60)
