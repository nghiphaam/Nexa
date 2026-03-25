"""Device utilities for CUDA, ROCm (AMD), XLA/TPU, and CPU."""
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    _HAS_XLA = True
except (ImportError, RuntimeError):
    torch_xla = None
    xm = None
    xr = None
    _HAS_XLA = False

_XLA_DEVICE_CACHE = None


def is_rocm():
    """Check if running on AMD ROCm."""
    try:
        return torch.version.hip is not None
    except Exception:
        return False


def get_gpu_count():
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def get_tpu_count():
    if not _HAS_XLA:
        return 0
    try:
        return xr.world_size()
    except Exception:
        return 0


def safe_cuda_alloc(device_id=0):
    """Test if CUDA/ROCm device can allocate memory safely."""
    try:
        if not torch.cuda.is_available():
            return False
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0 or device_id >= gpu_count:
            return False
        try:
            free, total = torch.cuda.mem_get_info(device_id)
            if free < 100 * 1024 * 1024:
                return False
        except Exception:
            pass  # Ignore memory check if unavailable
        test = torch.empty((1, 1024, 1024), device=f"cuda:{device_id}", dtype=torch.float32)
        del test
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False


def safe_xla_alloc():
    """Test if XLA/TPU device can allocate memory safely."""
    global _XLA_DEVICE_CACHE
    if not _HAS_XLA:
        return False
    try:
        if _XLA_DEVICE_CACHE is None:
            _XLA_DEVICE_CACHE = xm.xla_device()
        test = torch.zeros((10, 10), device=_XLA_DEVICE_CACHE)
        xm.mark_step()
        del test
        return True
    except (RuntimeError, OSError) as e:
        print(f"Warning: XLA allocation test failed: {e}")
        _XLA_DEVICE_CACHE = None
        return False


def get_xla_device():
    global _XLA_DEVICE_CACHE
    if not _HAS_XLA:
        raise RuntimeError("torch_xla is not available")
    if _XLA_DEVICE_CACHE is None:
        _XLA_DEVICE_CACHE = xm.xla_device()
    return _XLA_DEVICE_CACHE


def has_xla_device():
    return safe_xla_alloc()


def clear_xla_device_cache():
    global _XLA_DEVICE_CACHE
    _XLA_DEVICE_CACHE = None
    return None


def setup_distributed_cuda():
    """Setup distributed training for CUDA/ROCm GPUs."""
    if not torch.cuda.is_available():
        return 0, 1, "cpu"
    if "LOCAL_RANK" in os.environ:
        from datetime import timedelta
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if not torch.distributed.is_initialized():
            backend = "nccl"
            if is_rocm():
                try:
                    torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
                except Exception:
                    backend = "gloo"
                    torch.distributed.init_process_group(backend="gloo", timeout=timedelta(seconds=60))
            else:
                torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        return local_rank, world_size, device
    else:
        return 0, 1, "cuda:0"


def setup_distributed_xla():
    if not _HAS_XLA:
        raise RuntimeError("torch_xla is not available")
    device = get_xla_device()
    local_rank = xm.get_local_ordinal()
    world_size = xr.world_size()
    return local_rank, world_size, device


def get_device_info(include_all_devices=False):
    """
    Get comprehensive device information.

    Args:
        include_all_devices: If True, query all GPU properties (slower on multi-GPU).
                           If False, only query device 0 (faster).
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": get_gpu_count(),
        "is_rocm": is_rocm(),
        "xla_available": _HAS_XLA,
        "tpu_count": get_tpu_count(),
        "cpu_count": os.cpu_count() or 1,
    }
    if info["cuda_available"] and include_all_devices:
        info["cuda_devices"] = []
        for i in range(info["cuda_count"]):
            props = torch.cuda.get_device_properties(i)
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:
                free, total = 0, props.total_memory
            info["cuda_devices"].append({
                "id": i, "name": props.name,
                "total_memory_gb": total / (1024**3),
                "free_memory_gb": free / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    return info


def is_cuda_device(device):
    return str(device).startswith("cuda")


def is_xla_device(device):
    return str(device).startswith("xla")


def is_distributed():
    return torch.distributed.is_initialized()


def get_rank():
    if is_distributed():
        return torch.distributed.get_rank()
    elif _HAS_XLA:
        return xm.get_local_ordinal()
    return 0


def get_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    elif _HAS_XLA:
        return xr.world_size()
    return 1


def barrier():
    if is_distributed():
        torch.distributed.barrier()
    elif _HAS_XLA:
        xm.rendezvous("barrier")


def configure_tf32_runtime():
    if not torch.cuda.is_available():
        return
    if getattr(torch.backends.cuda, "_tf32_configured", False):
        return
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass  # Ignore memory check if unavailable
    torch.backends.cuda._tf32_configured = True


def auto_select_device(prefer_cuda=True):
    """
    Auto-select best available device.

    IMPORTANT: Respects LOCAL_RANK for multi-GPU/DDP to avoid all processes using cuda:0.
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            return f"cuda:{local_rank}"

    if prefer_cuda and torch.cuda.is_available() and safe_cuda_alloc(0):
        return "cuda:0"
    if safe_xla_alloc():
        return "xla"
    return "cpu"


def empty_cache(device=None):
    """Clear device cache for CUDA/ROCm."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_cuda_device(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_xla():
    """Synchronize XLA/TPU operations (flush graph, not memory clear)."""
    if _HAS_XLA:
        xm.mark_step()
