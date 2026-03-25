"""Auto-configuration based on hardware capabilities."""
import torch
from nexa.model.config import Config
from nexa.utils.device import is_cuda_device, is_xla_device, is_rocm


def auto_config(config: Config) -> Config:
    if is_xla_device(config.device):
        config.dtype = "bfloat16"
        config.compile_model = False

        # Only auto-adjust grad_accum if user didn't explicitly set batch_size or grad_accum
        # NOTE: This heuristic checks if values differ from defaults (batch_size=2, grad_accum_steps=16)
        # Limitation: If user explicitly sets batch_size=2, it will be treated as default
        user_set_batch = config.batch_size != 2
        user_set_accum = config.grad_accum_steps != 16

        if not user_set_batch and not user_set_accum:
            # Auto-configure both
            target_tokens = (50000 if config.preset == "auto"
                            else {"low": 16000, "mid": 50000, "high": 120000}.get(config.preset, 50000))
            config.grad_accum_steps = max(1, target_tokens // max(1, config.batch_size * config.block_size))

        if config.sliding_window is None:
            config.sliding_window = max(config.block_size, 512)
        print(f"[auto_config] TPU/XLA device={config.device}")
        print(f"[auto_config] dtype={config.dtype}, compile=disabled, "
              f"batch={config.batch_size}, accum={config.grad_accum_steps}, "
              f"sliding_window={config.sliding_window}, grad_ckpt={config.use_grad_ckpt}")
        return config

    if not torch.cuda.is_available():
        config.dtype = "float32"
        config.compile_model = False
        config.batch_size = min(config.batch_size, 1)
        target_tokens = 50000
        config.grad_accum_steps = max(1, target_tokens // max(1, config.batch_size * config.block_size))
        return config

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    sm = props.major

    # ROCm and CUDA dtype selection
    if is_rocm():
        # AMD ROCm: prefer bfloat16 for MI200+, float16 otherwise
        config.dtype = "bfloat16" if vram_gb >= 64 else "float16"
        config.compile_mode = "reduce-overhead"  # ROCm compile support varies
    else:
        # NVIDIA CUDA
        config.dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        config.compile_mode = "max-autotune" if sm >= 8 else "reduce-overhead"

    if config.preset == "auto":
        if vram_gb < 16:
            config.batch_size = 1
            config.block_size = min(config.block_size, 256)
            config.use_grad_ckpt = True
        elif vram_gb < 40:
            config.batch_size = 2
        else:
            config.batch_size = 4

    target_tokens = (20000 if config.preset == "auto"
                    else {"low": 8000, "mid": 20000, "high": 50000}.get(config.preset, 20000))
    config.grad_accum_steps = max(1, target_tokens // max(1, config.batch_size * config.block_size))

    if config.block_size * config.n_layer > 4000:
        config.use_grad_ckpt = True

    if config.sliding_window is None:
        head_dim = config.n_embd // config.n_head
        bytes_per_elem = 2 if config.dtype in ("float16", "bfloat16") else 4
        bytes_per_token = config.n_kv_head * head_dim * 2 * bytes_per_elem
        kv_budget = vram_gb * (1024**3) * 0.3
        max_tokens = kv_budget // (bytes_per_token * config.n_layer * config.batch_size)
        config.sliding_window = int(min(max_tokens, config.block_size * 2))

    device_type = "ROCm" if is_rocm() else "CUDA"
    print(f"[auto_config] {device_type} GPU={props.name} ({vram_gb:.1f}GB, sm_{props.major}{props.minor})")
    print(f"[auto_config] dtype={config.dtype}, compile={config.compile_mode}, "
          f"batch={config.batch_size}, accum={config.grad_accum_steps}, "
          f"sliding_window={config.sliding_window}, grad_ckpt={config.use_grad_ckpt}")

    return config
