#!/usr/bin/env python3
"""Train Nexa model with multi-GPU and TPU support."""
import argparse
import os
import sys

# Keep the repo root importable when the script is launched from outside the repo.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    p = argparse.ArgumentParser(description="Train Nexa Language Model")

    # Data
    p.add_argument("--data-dir", type=str, default="data")

    # Model architecture
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--n-embd", type=int, default=2048)
    p.add_argument("--n-head", type=int, default=16)
    p.add_argument("--n-kv-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=24)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=2e-5)
    p.add_argument("--max-iters", type=int, default=10000)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=1337)

    # Optimization
    p.add_argument("--use-grad-ckpt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sliding-window", type=int, default=None)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )

    # Device
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "xla", "tpu"],
        help="Device to use. 'auto' will select best available. For multi-GPU, use torchrun."
    )
    p.add_argument(
        "--preset",
        type=str,
        default="auto",
        choices=["auto", "low", "mid", "high"],
        help="Model size preset"
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"]
    )

    args = p.parse_args()

    # Import here to avoid loading heavy dependencies at startup
    from nexa.model.config import Config
    from nexa.training.trainer import train
    from nexa.training.utils import apply_preset_args
    from nexa.tokenizer.tokenizer import get_vocab_size
    from nexa.utils.device import (
        safe_cuda_alloc,
        safe_xla_alloc,
        get_xla_device,
        auto_select_device,
    )

    args = apply_preset_args(args)

    # Device selection
    if args.device == "auto":
        device_str = auto_select_device(prefer_cuda=True)
    elif args.device == "cpu":
        device_str = "cpu"
    elif args.device in ("xla", "tpu"):
        device_str = "xla"
        if not safe_xla_alloc():
            raise RuntimeError("XLA/TPU requested but not available")
        get_xla_device()  # initialize and cache once
        device_str = "xla"
    elif args.device == "cuda":
        if not safe_cuda_alloc(0):
            raise RuntimeError("CUDA requested but not available")
        device_str = "cuda:0"
    else:
        device_str = args.device

    # Dtype selection
    if args.dtype != "auto":
        dtype_map = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
        dtype_str = dtype_map[args.dtype]
    else:
        import torch
        if device_str.startswith("cuda"):
            dtype_str = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        elif device_str.startswith("xla"):
            dtype_str = "bfloat16"
        else:
            dtype_str = "float32"

    vocab_size = get_vocab_size()
    config = Config(
        data_dir=args.data_dir,
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        min_lr=args.min_lr,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=device_str,
        dtype=dtype_str,
        use_grad_ckpt=args.use_grad_ckpt,
        sliding_window=args.sliding_window,
        preset=args.preset,
        seed=args.seed,
    )

    # Print distributed info if applicable
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(f"Distributed training: rank {local_rank}/{world_size}")
        print("Tip: Use 'torchrun --nproc_per_node=N train.py ...' for multi-GPU")

    train(config)


if __name__ == "__main__":
    main()
