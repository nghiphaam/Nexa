"""Prepare reasoning datasets for training."""
import json
import os
import importlib


def require_pkg(name, pip_name=None):
    try:
        importlib.import_module(name)
    except ImportError as exc:
        package = pip_name or name
        raise RuntimeError(
            f"Missing required dependency '{package}'. Install it before running prepare_reasoning.py."
        ) from exc


require_pkg("datasets")


def download_and_convert(repo_id: str, output_path: str, format_fn):
    """
    Download dataset from HuggingFace and convert to Nexa format.

    Args:
        repo_id: HuggingFace dataset repo
        output_path: Output JSONL path
        format_fn: Function to format each sample
    """
    from datasets import load_dataset

    print(f"Loading {repo_id}...")
    ds = load_dataset(repo_id, split="train")

    print(f"Converting {len(ds)} samples...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(ds):
            try:
                formatted = format_fn(sample)
                if formatted:
                    f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[warn] Skip sample {i}: {e}")

            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(ds)} processed", end="\r")

    print(f"\nSaved to {output_path}")


def format_opus_reasoning(sample: dict) -> dict | None:
    """Format nohurry/Opus-4.6-Reasoning-3000x-filtered."""
    if "conversations" not in sample:
        return None

    messages = []
    for msg in sample["conversations"]:
        role = msg.get("from", "")
        content = msg.get("value", "")

        if role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": content})

    if not messages:
        return None

    return {"messages": messages}


def format_teichai_reasoning(sample: dict) -> dict | None:
    """Format TeichAI/claude-4.5-opus-high-reasoning-250x."""
    if "messages" not in sample:
        return None

    messages = []
    for msg in sample["messages"]:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ["user", "assistant"]:
            messages.append({"role": role, "content": content})

    if not messages:
        return None

    return {"messages": messages}


def format_qwen_reasoning(sample: dict) -> dict | None:
    """Format Jackrong/Qwen3.5-reasoning-700x."""
    if "conversations" not in sample:
        return None

    messages = []
    for msg in sample["conversations"]:
        role = msg.get("from", "")
        content = msg.get("value", "")

        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})

    if not messages:
        return None

    return {"messages": messages}


def merge_datasets(input_files: list[str], output_path: str):
    """
    Merge multiple JSONL files into one.

    Args:
        input_files: List of input JSONL paths
        output_path: Output merged JSONL path
    """
    total = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"[warn] {input_file} not found, skipping")
                continue

            count = 0
            with open(input_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:
                        out_f.write(line + "\n")
                        count += 1

            print(f"  {input_file}: {count} samples")
            total += count

    print(f"\nMerged {total} samples to {output_path}")


def main():
    data_dir = "data/reasoning"
    os.makedirs(data_dir, exist_ok=True)

    datasets = [
        ("nohurry/Opus-4.6-Reasoning-3000x-filtered", "opus_reasoning.jsonl", format_opus_reasoning),
        ("TeichAI/claude-4.5-opus-high-reasoning-250x", "teichai_reasoning.jsonl", format_teichai_reasoning),
        ("Jackrong/Qwen3.5-reasoning-700x", "qwen_reasoning.jsonl", format_qwen_reasoning),
    ]

    output_files = []
    for repo_id, filename, format_fn in datasets:
        output_path = os.path.join(data_dir, filename)
        try:
            download_and_convert(repo_id, output_path, format_fn)
            output_files.append(output_path)
        except Exception as e:
            print(f"[error] Failed to process {repo_id}: {e}")

    # Merge all datasets
    merged_path = os.path.join(data_dir, "reasoning_merged.jsonl")
    merge_datasets(output_files, merged_path)

    print(f"\n✓ Done! Merged dataset: {merged_path}")


if __name__ == "__main__":
    main()
