"""Model backends. Importing this package registers the built-in adapters."""

from .base import (
    ModelAdapter,
    UnsupportedModelError,
    adapter_for,
    cache_indices,
    register_adapter,
    registered_adapters,
)
from .ultralytics import UltralyticsAdapter

__all__ = [
    "ModelAdapter",
    "UltralyticsAdapter",
    "UnsupportedModelError",
    "adapter_for",
    "cache_indices",
    "register_adapter",
    "registered_adapters",
]
