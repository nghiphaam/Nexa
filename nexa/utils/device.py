"""Device utilities."""
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch_xla.core.xla_model as xm
    _HAS_XLA = True
except (ImportError, RuntimeError):
    xm = None
    _HAS_XLA = False

_XLA_DEVICE_CACHE = None


def safe_cuda_alloc(device_id=0):
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
            pass
        test = torch.empty((1, 1024, 1024), device=f"cuda:{device_id}", dtype=torch.float32)
        del test
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False


def safe_xla_alloc():
    """Test whether an XLA device can allocate memory safely."""
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


def is_cuda_device(device):
    return str(device).startswith("cuda")


def is_xla_device(device):
    return str(device).startswith("xla")


def auto_select_device(prefer_cuda=True):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            return f"cuda:{local_rank}"

    if prefer_cuda and torch.cuda.is_available() and safe_cuda_alloc(0):
        return "cuda:0"
    if safe_xla_alloc():
        return "xla"
    return "cpu"
