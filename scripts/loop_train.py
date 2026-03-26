"""Closed-loop training script: train -> generate -> filter -> retrain."""
import argparse
import os
import sys
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nexa.inference.chat_session import ChatSession
from nexa.inference.self_improve import batch_generate_and_score
from nexa.model.config import Config
from nexa.model.nexa_model import NexaModel
from nexa.tokenizer.tokenizer import load_tokenizer
from nexa.training.self_improve_dataset import append_jsonl, count_samples


def _normalize_config(raw_config, device: str) -> Config:
    if raw_config is None:
        config = Config()
    elif isinstance(raw_config, Config):
        config = raw_config
    elif isinstance(raw_config, dict):
        config = Config(**raw_config)
    elif is_dataclass(raw_config):
        config = Config(**asdict(raw_config))
    else:
        allowed = {f.name for f in fields(Config)}
        config_kwargs = {name: getattr(raw_config, name) for name in allowed if hasattr(raw_config, name)}
        config = Config(**config_kwargs)
    config.device = device
    return config


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint using the checkpoint's saved Config."""
    print(f"Loading model from {checkpoint_path}...")
    map_location = 'cpu' if device.startswith('xla') else device
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    config = _normalize_config(ckpt.get('config'), device)
    model = NexaModel(config)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing or unexpected:
        print(f"[warn] checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)
    model.eval()
    return model, config


def generate_training_data(model, config, prompts: list[str], output_path: str, min_score: float = 0.7):
    print(f"\nGenerating data for {len(prompts)} prompts...")
    tokenizer = load_tokenizer()
    session = ChatSession(model, tokenizer, config)
    samples = batch_generate_and_score(session, prompts, min_score)
    print(f"Generated {len(samples)} high-quality samples (score >= {min_score})")
    for sample in samples:
        append_jsonl(output_path, sample)
    return len(samples)


def get_default_prompts():
    return [
        'Explain the concept of recursion in programming.',
        'What are the key differences between supervised and unsupervised learning?',
        'How does a neural network learn?',
        'Explain the time complexity of quicksort.',
        'What is the difference between a stack and a queue?',
        'How does gradient descent work?',
        'Explain the concept of overfitting in machine learning.',
        'What is the purpose of regularization?',
        'How does backpropagation work?',
        'Explain the difference between classification and regression.',
    ]


def main():
    parser = argparse.ArgumentParser(description='Closed-loop self-improvement training')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt', help='Model checkpoint path')
    parser.add_argument('--output', default='data/self_improve.jsonl', help='Output dataset path')
    parser.add_argument('--prompts-file', default=None, help='File with prompts (one per line)')
    parser.add_argument('--min-score', type=float, default=0.7, help='Minimum quality score')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu/xla)')
    parser.add_argument('--iterations', type=int, default=1, help='Number of loop iterations')
    parser.add_argument('--train-after-gen', action='store_true', help='Run training after generation')
    args = parser.parse_args()

    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = get_default_prompts()
        print(f"Using {len(prompts)} default prompts")

    for iteration in range(args.iterations):
        print(f"\n{'=' * 65}")
        print(f"  ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'=' * 65}")

        model, config = load_model(args.checkpoint, args.device)
        initial_count = count_samples(args.output)
        new_samples = generate_training_data(model, config, prompts, args.output, args.min_score)
        final_count = count_samples(args.output)

        print("\nDataset stats:")
        print(f"  Before: {initial_count} samples")
        print(f"  Added:  {new_samples} samples")
        print(f"  Total:  {final_count} samples")

        if args.train_after_gen and iteration < args.iterations - 1:
            print('\n[info] Training not implemented in this script.')
            print('[info] Run train.py separately with the updated dataset.')
            break

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif args.device.startswith('xla'):
            from nexa.utils.device import sync_xla
            sync_xla()

    print(f"\nDone. Self-improvement dataset: {args.output}")


if __name__ == '__main__':
    main()
