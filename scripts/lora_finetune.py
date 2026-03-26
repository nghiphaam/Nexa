#!/usr/bin/env python3
"""LoRA fine-tuning script for Nexa models."""
import argparse
import json
import os
import sys
from dataclasses import asdict, fields, is_dataclass

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexa.inference.lora import LoRALinear, apply_lora
from nexa.model.config import Config
from nexa.model.nexa_model import NexaModel
from nexa.tokenizer.tokenizer import EOS_TOKEN, load_tokenizer


def normalize_config(raw_config) -> Config:
    if raw_config is None:
        return Config()
    if isinstance(raw_config, Config):
        return raw_config
    if isinstance(raw_config, dict):
        return Config(**raw_config)
    if is_dataclass(raw_config):
        return Config(**asdict(raw_config))
    allowed = {f.name for f in fields(Config)}
    config_kwargs = {name: getattr(raw_config, name) for name in allowed if hasattr(raw_config, name)}
    return Config(**config_kwargs)


def freeze_non_lora_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = 'lora_' in name


def collect_lora_params(model):
    return [param for name, param in model.named_parameters() if param.requires_grad and 'lora_' in name]


def load_sft_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if 'messages' in item:
                text = ''
                for msg in item['messages']:
                    role = msg['role']
                    content = msg['content']
                    if role == 'user':
                        text += f"<|user|>{content}<|assistant|>"
                    elif role == 'assistant':
                        text += f"{content}<|endoftext|>"
                if text:
                    data.append(text)
    return data


def main():
    p = argparse.ArgumentParser(description='LoRA fine-tuning for Nexa')
    p.add_argument('--checkpoint', type=str, required=True, help='Base model checkpoint')
    p.add_argument('--data', type=str, required=True, help='SFT data (JSONL)')
    p.add_argument('--output', type=str, default='lora_checkpoints', help='Output directory')
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-alpha', type=int, default=16)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--target-modules', nargs='*', default=None, help='Optional override for LoRA target module attribute names')
    args = p.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = normalize_config(ckpt.get('config'))
    config.device = device

    tokenizer = load_tokenizer()
    config.vocab_size = tokenizer.get_vocab_size()
    config.eos_id = tokenizer.token_to_id(EOS_TOKEN)

    model = NexaModel(config)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing or unexpected:
        print(f"[warn] checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)

    print(f"Adding LoRA layers (rank={args.lora_rank}, alpha={args.lora_alpha})")
    apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, target_modules=args.target_modules)
    freeze_non_lora_params(model)
    lora_params = collect_lora_params(model)
    if not lora_params:
        raise RuntimeError('No LoRA parameters were attached. Check target modules and model architecture.')

    print(f"Loading SFT data: {args.data}")
    texts = load_sft_data(args.data)
    if not texts:
        raise RuntimeError('No SFT examples found in the provided dataset.')
    print(f"Loaded {len(texts)} examples")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0.0
        step_count = 0

        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            if not batch_texts:
                continue

            token_rows = []
            for text in batch_texts:
                ids = tokenizer.encode(text).ids
                if len(ids) > config.block_size:
                    ids = ids[:config.block_size]
                token_rows.append(ids)

            max_len = max(len(ids) for ids in token_rows)
            padded_ids = []
            padded_labels = []
            for ids in token_rows:
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [config.eos_id] * pad_len)
                padded_labels.append(ids + [-100] * pad_len)

            input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
            labels = torch.tensor(padded_labels, dtype=torch.long, device=device)

            _, loss = model(input_ids, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            global_step += 1
            step_count += 1

            if global_step % 10 == 0:
                print(f"  Step {global_step} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(1, step_count)
        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, 'lora_weights.pt')

    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[name] = {
                'lora_A': module.lora_A.detach().cpu(),
                'lora_B': module.lora_B.detach().cpu(),
                'rank': module.rank,
                'alpha': module.alpha,
            }

    torch.save({
        'lora_state': lora_state,
        'rank': args.lora_rank,
        'alpha': args.lora_alpha,
        'base_checkpoint': args.checkpoint,
    }, output_path)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")
    print(f"\nLoRA weights saved to: {output_path}")


if __name__ == '__main__':
    main()
