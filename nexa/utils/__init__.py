"""Device utilities."""
from nexa.utils.device import (
    auto_select_device,
    is_cuda_device,
    is_xla_device,
    safe_cuda_alloc,
    safe_xla_alloc,
    get_xla_device,
)

__all__ = [
    "auto_select_device",
    "is_cuda_device",
    "is_xla_device",
    "safe_cuda_alloc",
    "safe_xla_alloc",
    "get_xla_device",
]
