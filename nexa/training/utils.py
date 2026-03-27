"""Training utilities: data loading, optimizer, lr schedule, etc."""
import os
import math
import sys
import contextlib
import numpy as np
import torch
import torch.nn.functional as F
from nexa.utils.device import is_cuda_device, is_xla_device


def get_random_batch(data, block_size, batch_size, device):
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError(
            f"Dataset too small for block_size={block_size}: got {len(data)} tokens"
        )
    ix = torch.randint(max_start, (batch_size,))
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
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
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


def _collect_explicit_cli_fields(argv):
    option_to_field = {
        "--block-size": "block_size",
        "--batch-size": "batch_size",
        "--n-layer": "n_layer",
        "--n-embd": "n_embd",
        "--n-head": "n_head",
        "--n-kv-head": "n_kv_head",
        "--use-grad-ckpt": "use_grad_ckpt",
        "--no-use-grad-ckpt": "use_grad_ckpt",
        "--compile": "compile",
        "--no-compile": "compile",
    }
    explicit_fields = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        field = option_to_field.get(token.split("=", 1)[0])
        if field is not None:
            explicit_fields.add(field)
    return explicit_fields


def apply_preset_args(args, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    explicit_fields = _collect_explicit_cli_fields(argv)
    preset_values = None

    if args.preset == "low":
        preset_values = {
            "block_size": 128,
            "batch_size": 8,
            "n_layer": 4,
            "n_embd": 256,
            "n_head": 4,
            "n_kv_head": 1,
            "use_grad_ckpt": False,
            "compile": False,
        }
    elif args.preset == "mid":
        preset_values = {
            "block_size": 384,
            "batch_size": 32,
            "n_layer": 12,
            "n_embd": 768,
            "n_head": 12,
            "n_kv_head": 4,
        }
    elif args.preset == "high":
        preset_values = {
            "block_size": 512,
            "batch_size": 128,
            "n_layer": 16,
            "n_embd": 1024,
            "n_head": 16,
            "n_kv_head": 4,
            "use_grad_ckpt": True,
        }

    if preset_values is None:
        return args

    for field, value in preset_values.items():
        if field not in explicit_fields:
            setattr(args, field, value)
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
