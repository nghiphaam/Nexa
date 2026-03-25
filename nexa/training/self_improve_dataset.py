"""Dataset writer for self-improvement training data."""
import json
import os
from pathlib import Path


def append_jsonl(filepath: str, sample: dict):
    """
    Append a sample to JSONL file.

    Args:
        filepath: Path to JSONL file
        sample: Dict containing training sample
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_batch(filepath: str, samples: list[dict]):
    """
    Save multiple samples to JSONL file.

    Args:
        filepath: Path to JSONL file
        samples: List of training samples
    """
    for sample in samples:
        append_jsonl(filepath, sample)


def load_jsonl(filepath: str) -> list[dict]:
    """
    Load all samples from JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of samples
    """
    if not os.path.exists(filepath):
        return []

    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def count_samples(filepath: str) -> int:
    """
    Count number of samples in JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        Number of samples
    """
    if not os.path.exists(filepath):
        return 0

    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def format_for_training(sample: dict) -> str:
    """
    Format sample for training (convert to text).

    Args:
        sample: Sample dict with messages, thought, answer

    Returns:
        Formatted text string
    """
    text_parts = []

    # Add conversation history
    for msg in sample.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text_parts.append(f"<|user|>{content}")
        elif role == "assistant":
            text_parts.append(f"<|assistant|>{content}")

    # Add thought if present
    if sample.get("thought"):
        text_parts.append(f"Thought: {sample['thought']}")

    # Add answer
    if sample.get("answer"):
        text_parts.append(f"Answer: {sample['answer']}")

    text_parts.append("<|endoftext|>")
    return "\n".join(text_parts)
