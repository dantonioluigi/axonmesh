"""The contract a model family must satisfy to be splittable.

Everything else in this package — the planner, the transports and codecs, the
wire protocol, the adaptive policy, the operator — works on a *graph of layers*
and a *wire set*, not on any particular architecture. An adapter is the thin
piece that knows how to read one family's graph and run a span of it; implement
it and the whole toolchain applies.

Four questions an adapter answers:

- ``graph()`` — the layers and how they feed each other (resolved indices).
- ``default_cut()`` — the natural place to cut (e.g. where the backbone ends).
- ``probe_shapes()`` — each layer's output shape, so a cut can be priced.
- ``run_span()`` — execute layers ``start..stop``, honouring skip connections.

Adapters register a detector so :func:`adapter_for` can pick one from a bare
model object, which is what makes ``SplitModel(model)`` work without the caller
naming a backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from ..topology import LayerInfo


class UnsupportedModelError(TypeError):
    """No registered adapter recognises this model."""


@runtime_checkable
class ModelAdapter(Protocol):
    """What the split machinery needs from a model family."""

    @property
    def module(self) -> nn.Module:
        """The underlying torch model (for device moves and fingerprinting)."""
        ...

    def graph(self) -> list[LayerInfo]:
        """Layers with their resolved source indices, in execution order."""
        ...

    def default_cut(self) -> int:
        """The natural cut point when the caller does not choose one."""
        ...

    def probe_shapes(self, imgsz: int = 640) -> list[tuple[int, ...] | None]:
        """Output shape of every layer for a square ``imgsz`` input."""
        ...

    def run_span(self, start: int, stop: int, x: Any, cache: dict[int, torch.Tensor]) -> Any:
        """Run layers ``start..stop`` inclusive, reading/writing ``cache``."""
        ...


def cache_indices(graph: list[LayerInfo]) -> set[int]:
    """Layers whose output a *non-adjacent* consumer needs, so must be cached.

    Derived from the graph rather than a framework-specific attribute, so every
    adapter gets the same rule for free.
    """
    return {
        source
        for layer in graph
        for source in layer.sources
        if source >= 0 and source != layer.index - 1
    }


Detector = Callable[[Any], bool]
Builder = Callable[[Any], ModelAdapter]
_REGISTRY: list[tuple[str, Detector, Builder]] = []


def register_adapter(name: str, detects: Detector, build: Builder) -> None:
    """Register a backend: ``detects(model)`` decides, ``build(model)`` wraps."""
    _REGISTRY.append((name, detects, build))


def registered_adapters() -> list[str]:
    """Names of the registered backends, in resolution order."""
    return [name for name, _, _ in _REGISTRY]


def adapter_for(model: Any) -> ModelAdapter:
    """Wrap ``model`` in the first adapter that claims it.

    Passing an adapter through is allowed, so callers can bypass detection.
    """
    if isinstance(model, ModelAdapter) and not isinstance(model, nn.Module):
        return model
    for _name, detects, build in _REGISTRY:
        try:
            claimed = detects(model)
        except Exception:  # a probe must never break resolution
            claimed = False
        if claimed:
            return build(model)
    raise UnsupportedModelError(
        f"no adapter for {type(model).__name__}; registered: {registered_adapters()}"
    )
