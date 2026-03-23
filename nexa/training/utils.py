"""Training utilities: data loading, optimizer, lr schedule, etc."""
import os
import math
import contextlib
import numpy as np
import torch
import torch.nn.functional as F
from nexa.utils.device import is_cuda_device, is_xla_device


def get_random_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i: i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix])
    if is_cuda_device(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def make_amp_context(device, dtype):
    if not is_cuda_device(device):
        return contextlib.nullcontext()
    if dtype not in (torch.float16, torch.bfloat16):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        return torch.amp.autocast(device_type=device, dtype=dtype)
    except Exception:
        return contextlib.nullcontext()


def safe_load_model_state(model, state_dict, label="checkpoint"):
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] {label} loaded with missing={len(missing)} unexpected={len(unexpected)}")
        return True
    except Exception as e:
        print(f"[warn] {label} incompatible -> partial load skipped: {e}")
        return False


def apply_preset_args(args):
    if args.preset == "low":
        args.block_size = 128
        args.batch_size = 8
        args.n_layer = 4
        args.n_embd = 256
        args.n_head = 4
        args.n_kv_head = 1
        args.use_grad_ckpt = False
        args.compile = False
    elif args.preset == "mid":
        args.block_size = 384
        args.batch_size = 32
        args.n_layer = 12
        args.n_embd = 768
        args.n_head = 12
        args.n_kv_head = 4
    elif args.preset == "high":
        args.block_size = 512
        args.batch_size = 128
        args.n_layer = 16
        args.n_embd = 1024
        args.n_head = 16
        args.n_kv_head = 4
        args.use_grad_ckpt = True
    return args


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, amp_ctx):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_random_batch(data, config.block_size, config.batch_size, config.device)
            with amp_ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def generate_sample(model, tokenizer, prompt, max_tokens, config, amp_ctx):
    idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=config.device)
    model.eval()
    with amp_ctx:
        out = model.generate(
            idx, max_new_tokens=max_tokens, temperature=config.temperature,
            top_k=config.top_k, top_p=config.top_p, min_p=config.min_p,
            repetition_penalty=config.repetition_penalty,
        )
    model.train()
    return tokenizer.decode(out[0].tolist())
