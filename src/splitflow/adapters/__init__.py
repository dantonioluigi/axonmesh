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

# Imported last so the generic fallback is tried after purpose-built adapters.
from .fx import FxAdapter, TraceError  # isort: skip

__all__ = [
    "FxAdapter",
    "ModelAdapter",
    "TraceError",
    "UltralyticsAdapter",
    "UnsupportedModelError",
    "adapter_for",
    "cache_indices",
    "register_adapter",
    "registered_adapters",
]
