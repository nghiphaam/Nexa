"""Main training loop for Nexa model with multi-GPU and TPU support."""
import os
import time
import math
import random
import contextlib
import numpy as np
import torch
from nexa.model.config import Config
from nexa.model.nexa_model import NexaModel
from nexa.tokenizer.tokenizer import load_tokenizer, EOS_TOKEN
from nexa.utils.device import (
    is_cuda_device,
    is_xla_device,
    configure_tf32_runtime,
    setup_distributed_cuda,
    setup_distributed_xla,
    get_rank,
    get_world_size,
    barrier,
    is_distributed,
)
from nexa.training.data import DataLoaderLite, CUDAPrefetcher
from nexa.training.optimizer import configure_optimizer, get_lr
from nexa.training.utils import (
    make_amp_context,
    safe_load_model_state,
    estimate_loss,
    generate_sample,
)
from nexa.training.auto_config import auto_config


def train(config: Config):
    local_rank = 0
    world_size = 1
    is_main_process = True

    if is_xla_device(config.device):
        local_rank, world_size, config.device = setup_distributed_xla()
        is_main_process = (local_rank == 0)
    elif is_cuda_device(config.device) and "LOCAL_RANK" in os.environ:
        local_rank, world_size, config.device = setup_distributed_cuda()
        is_main_process = (local_rank == 0)

    random.seed(config.seed + local_rank)
    np.random.seed(config.seed + local_rank)
    torch.manual_seed(config.seed + local_rank)
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + local_rank)
        torch.cuda.manual_seed_all(config.seed + local_rank)
        configure_tf32_runtime()
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    config = auto_config(config)

    device = config.device
    tokenizer = load_tokenizer()
    config.vocab_size = tokenizer.get_vocab_size()
    config.eos_id = tokenizer.token_to_id(EOS_TOKEN)

    if is_main_process:
        print("=" * 65)
        print("  Nexa 1.5  (~1B, Multi-GPU/TPU)")
        print("=" * 65)
        print(f"\nDevice       : {device}")
        print(f"World size   : {world_size}")
        print(f"Local rank   : {local_rank}")

        if is_cuda_device(device):
            props = torch.cuda.get_device_properties(local_rank)
            print(f"GPU          : {props.name} (sm_{props.major}{props.minor})")
            print(f"VRAM         : {props.total_memory / (1024**3):.1f} GB")
            if props.major < 8:
                print("SDPA backend : memory-efficient (no flash_attn)")
            else:
                print("SDPA backend : flash_attention_2")
        elif is_xla_device(device):
            print("Accelerator  : TPU/XLA")

    train_path = os.path.join(config.data_dir, "train.bin")
    val_path = os.path.join(config.data_dir, "val.bin")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise RuntimeError(
            f"Missing {train_path} or {val_path}. Run pre_train.py first."
        )

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    if is_main_process:
        print(f"\nTrain tokens : {len(train_data):,}")
        print(f"Val tokens   : {len(val_data):,}")
        print(f"Tokenizer    : Nexa BPE (vocab={config.vocab_size})")
        print(
            f"\nArchitecture : block={config.block_size}, embd={config.n_embd}, "
            f"q_head={config.n_head}, kv_head={config.n_kv_head}, layer={config.n_layer}"
        )
        print(
            f"GQA ratio    : {config.n_head}:{config.n_kv_head} "
            f"({config.n_head // config.n_kv_head} Q per KV group)"
        )

    model = NexaModel(config).to(device)

    if world_size > 1:
        if is_xla_device(device):
            pass  # XLA handles distribution automatically
        elif is_distributed():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

    resume_iter = 0
    ckpt_path = os.path.join(config.checkpoint_dir, "best.pt")
    if os.path.exists(ckpt_path):
        if is_main_process:
            print(f"\nResuming from {ckpt_path}...")
        map_location = device if not is_xla_device(device) else "cpu"
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

        model_to_load = model.module if hasattr(model, "module") else model
        safe_load_model_state(model_to_load, ckpt["model"], label="resume checkpoint")
        resume_iter = ckpt.get("iter", 0)

        if is_main_process:
            print(f"  Resumed at iter {resume_iter}, val_loss={ckpt.get('val_loss', '?')}")

    if config.compile_model and hasattr(torch, "compile") and not is_xla_device(device):
        if is_main_process:
            print(f"torch.compile enabled (mode={config.compile_mode})")
        try:
            model = torch.compile(model, mode=config.compile_mode)
        except Exception as e:
            if is_main_process:
                print(f"[warn] torch.compile failed: {e}. Falling back to eager mode.")

    optimizer, fused_str = configure_optimizer(model, config, device)

    if resume_iter > 0 and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        if is_main_process:
            print("  Restored optimizer state")

    micro_bs = config.batch_size
    tok_per_step = micro_bs * config.grad_accum_steps * config.block_size * world_size

    if is_main_process:
        print(f"Optimizer    : AdamW ({fused_str}, lr={config.lr}, min_lr={config.min_lr})")
        print(
            f"Batch        : {micro_bs} x {config.grad_accum_steps} accum x {config.block_size} x {world_size} GPUs = {tok_per_step:,} tok/step"
        )
        print(f"Max iters    : {config.max_iters:,}")
        print(
            f"LR schedule  : warmup({config.warmup_iters}) -> cosine -> floor({config.min_lr})"
        )

    use_amp = is_cuda_device(device)
    dtype_obj = getattr(torch, config.dtype, torch.float32)
    amp_ctx = make_amp_context(device, dtype_obj) if use_amp else contextlib.nullcontext()
    scaler = (
        torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))
        if use_amp
        else None
    )

    if resume_iter > 0 and scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
        if is_main_process:
            print("  Restored scaler state")

    if is_main_process:
        print(f"\n{'\u2500' * 65}")
        print("  TRAINING")
        print(f"{'\u2500' * 65}")

    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    base_loader = DataLoaderLite(
        train_path, config.batch_size, config.block_size, device, eos_id=eos_id
    )
    train_loader = CUDAPrefetcher(base_loader)

    t0 = time.time()
    best_val = ckpt.get("val_loss", float("inf")) if resume_iter > 0 else float("inf")

    for it in range(resume_iter, config.max_iters + 1):
        lr = get_lr(it, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if (it % config.eval_interval == 0 or it == config.max_iters) and is_main_process:
            losses = estimate_loss(model, train_data, val_data, config, amp_ctx)
            elapsed = time.time() - t0
            ppl = math.exp(min(losses["val"], 20))
            print(
                f"\n  step {it:>5d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"ppl {ppl:.2f} | "
                f"lr {lr:.2e} | "
                f"{elapsed:.0f}s"
            )

            if losses["val"] < best_val:
                best_val = losses["val"]
                os.makedirs(config.checkpoint_dir, exist_ok=True)

                model_to_save = model.module if hasattr(model, "module") else model
                ckpt_data = {
                    "model": model_to_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": it,
                    "val_loss": best_val,
                    "config": config,
                    "meta": {
                        "tokenizer": "nexa",
                        "vocab_size": config.vocab_size,
                        "dtype": config.dtype,
                        "world_size": world_size,
                    },
                }
                if scaler is not None:
                    ckpt_data["scaler"] = scaler.state_dict()
                torch.save(ckpt_data, os.path.join(config.checkpoint_dir, "best.pt"))
                print(f"  ** best model saved (val={best_val:.4f})")

            if world_size > 1 and is_distributed():
                best_val_tensor = torch.tensor([best_val], device=device)
                torch.distributed.broadcast(best_val_tensor, src=0)
                best_val = best_val_tensor.item()

            sample = generate_sample(
                model, tokenizer, "The meaning of life is", 100, config, amp_ctx
            )
            print(f"  >> {sample[:200]}...\n")

        if world_size > 1:
            barrier()

        if it == config.max_iters:
            break

        accum_loss = 0.0
        skip_step = False
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(config.grad_accum_steps):
            x, y = train_loader.next_batch()

            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / config.grad_accum_steps

            if not torch.isfinite(loss):
                if is_main_process:
                    print(f"\n[warn] NaN loss at step={it}. Resetting optimizer state.")
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad = None
                        state = optimizer.state.get(p, {})
                        if 'exp_avg' in state:
                            state['exp_avg'].zero_()
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'].zero_()
                skip_step = True
                break

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

        if world_size > 1 and is_distributed():
            barrier()

        norm = -1.0

        if not skip_step:
            if is_xla_device(device):
                if config.grad_clip > 0:
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    ).item()
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(optimizer, barrier=False)
                xm.mark_step()
            elif scaler is not None:
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    ).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                if config.grad_clip > 0:
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    ).item()
                optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if it > 0 and it % config.log_interval == 0 and is_main_process:
            elapsed = time.time() - t0
            tps = (it * tok_per_step) / elapsed
            rem_iters = config.max_iters - it
            eta_s = rem_iters * (elapsed / it)
            eta_str = f"{eta_s / 60:.1f}m" if eta_s < 3600 else f"{eta_s / 3600:.1f}h"

            gpu_str = ""
            if is_cuda_device(device):
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_res = torch.cuda.memory_reserved() / 1e9
                gpu_str = f" | VRAM: {mem_alloc:.1f}/{mem_res:.1f}G"

            norm_str = f" | norm {norm:.2f}" if norm >= 0 else ""

            print(
                f"  step {it:>5d}/{config.max_iters} | loss {accum_loss:.4f} | lr {lr:.2e}{norm_str} | {tps:,.0f} tok/s | ETA: {eta_str}{gpu_str}"
            )

    if is_main_process:
        total = time.time() - t0
        print(f"\n{'\u2500' * 65}")
        print(f"  DONE in {total:.0f}s ({total / 60:.1f}min)")
        print(f"  Best val loss: {best_val:.4f} | PPL: {math.exp(min(best_val, 20)):.2f}")
        print(f"{'\u2500' * 65}")
