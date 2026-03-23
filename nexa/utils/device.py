"""Device utilities for CUDA (single/multi-GPU), XLA/TPU, and CPU."""
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    _HAS_XLA = True
except Exception:
    torch_xla = None
    xm = None
    _HAS_XLA = False


def get_gpu_count():
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def get_tpu_count():
    if not _HAS_XLA:
        return 0
    try:
        return xm.xrt_world_size()
    except Exception:
        return 0


def safe_cuda_alloc(device_id=0):
    try:
        if not torch.cuda.is_available():
            return False
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0 or device_id >= gpu_count:
            return False
        free, total = torch.cuda.mem_get_info(device_id)
        if free < 100 * 1024 * 1024:
            return False
        test = torch.empty((1, 1024, 1024), device=f"cuda:{device_id}", dtype=torch.float32)
        del test
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False


def safe_xla_alloc():
    if not _HAS_XLA:
        return False
    try:
        device = xm.xla_device()
        torch.zeros((1,), device=device)
        return True
    except Exception:
        return False


def get_xla_device():
    if not _HAS_XLA:
        raise RuntimeError("torch_xla is not available")
    return xm.xla_device()


def setup_distributed_cuda():
    if not torch.cuda.is_available():
        return 0, 1, "cpu"
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        return local_rank, world_size, device
    else:
        return 0, 1, "cuda:0"


def setup_distributed_xla():
    if not _HAS_XLA:
        raise RuntimeError("torch_xla is not available")
    device = xm.xla_device()
    local_rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    return local_rank, world_size, device


def get_device_info():
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": get_gpu_count(),
        "xla_available": _HAS_XLA,
        "tpu_count": get_tpu_count(),
        "cpu_count": os.cpu_count() or 1,
    }
    if info["cuda_available"]:
        info["cuda_devices"] = []
        for i in range(info["cuda_count"]):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
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
        return xm.get_ordinal()
    return 0


def get_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    elif _HAS_XLA:
        return xm.xrt_world_size()
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
            pass
    torch.backends.cuda._tf32_configured = True


def auto_select_device(prefer_cuda=True):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            return f"cuda:{local_rank}"
    if prefer_cuda and safe_cuda_alloc(0):
        return "cuda:0"
    if safe_xla_alloc():
        return "xla"
    return "cpu"
