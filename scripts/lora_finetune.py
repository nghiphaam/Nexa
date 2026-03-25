#!/usr/bin/env python3
"""LoRA fine-tuning script for Nexa models."""
import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexa.model.nexa_model import NexaModel
from nexa.model.config import Config
from nexa.tokenizer.tokenizer import load_tokenizer, EOS_TOKEN


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


def add_lora_to_model(model, rank=8, alpha=16, target_modules=None):
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']

    lora_params = []
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                lora = LoRALayer(module.in_features, module.out_features, rank, alpha)
                setattr(module, 'lora', lora)
                lora_params.extend(lora.parameters())

                original_forward = module.forward
                def new_forward(self, x, original_forward=original_forward):
                    base_out = original_forward(x)
                    if hasattr(self, 'lora'):
                        base_out = base_out + self.lora(x)
                    return base_out
                module.forward = new_forward.__get__(module, nn.Linear)

    for param in model.parameters():
        param.requires_grad = False

    for param in lora_params:
        param.requires_grad = True

    return lora_params


def load_sft_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if 'messages' in item:
                text = ""
                for msg in item['messages']:
                    role = msg['role']
                    content = msg['content']
                    if role == 'user':
                        text += f"<|user|>{content}<|assistant|>"
                    elif role == 'assistant':
                        text += f"{content}<|endoftext|>"
                data.append(text)
    return data


def main():
    p = argparse.ArgumentParser(description="LoRA fine-tuning for Nexa")
    p.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint")
    p.add_argument("--data", type=str, required=True, help="SFT data (JSONL)")
    p.add_argument("--output", type=str, default="lora_checkpoints", help="Output directory")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt['config']

    tokenizer = load_tokenizer()
    config.vocab_size = tokenizer.get_vocab_size()
    config.eos_id = tokenizer.token_to_id(EOS_TOKEN)

    model = NexaModel(config)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)

    print(f"Adding LoRA layers (rank={args.lora_rank}, alpha={args.lora_alpha})")
    lora_params = add_lora_to_model(model, rank=args.lora_rank, alpha=args.lora_alpha)

    print(f"Loading SFT data: {args.data}")
    texts = load_sft_data(args.data)
    print(f"Loaded {len(texts)} examples")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0.0

        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]

            input_ids = []
            for text in batch_texts:
                ids = tokenizer.encode(text).ids
                if len(ids) > config.block_size:
                    ids = ids[:config.block_size]
                input_ids.append(ids)

            max_len = max(len(ids) for ids in input_ids)
            padded_ids = []
            for ids in input_ids:
                padded = ids + [config.eos_id] * (max_len - len(ids))
                padded_ids.append(padded)

            input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
            labels = input_ids.clone()

            logits, loss = model(input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                print(f"  Step {global_step} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / (len(texts) // args.batch_size)
        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "lora_weights.pt")

    lora_state = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state[name] = {
                'lora_A': module.lora.lora_A.data.cpu(),
                'lora_B': module.lora.lora_B.data.cpu(),
            }

    torch.save({
        'lora_state': lora_state,
        'rank': args.lora_rank,
        'alpha': args.lora_alpha,
        'base_checkpoint': args.checkpoint,
    }, output_path)

    print(f"\nLoRA weights saved to: {output_path}")


if __name__ == "__main__":
    main()
