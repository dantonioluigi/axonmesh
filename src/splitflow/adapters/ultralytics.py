"""Adapter for ultralytics models (YOLO detection and friends).

Ultralytics keeps the network as a flat module list where each module carries
``m.i`` (its index) and ``m.f`` (what it consumes). This adapter turns that
into the generic graph the rest of the toolchain expects, and replays a span of
it exactly as ``BaseModel._predict_once`` would — so a split run is
bit-identical to an unsplit one.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..topology import LayerInfo, backbone_cut, build_graph, probe_output_shapes
from .base import ModelAdapter, cache_indices, register_adapter


class UltralyticsAdapter:
    """Reads the ``m.f``/``m.i`` wiring of an ultralytics model."""

    name = "ultralytics"

    def __init__(self, det_model: nn.Module) -> None:
        self._module = det_model
        self._graph = build_graph(det_model)
        self._cache_set = cache_indices(self._graph)

    @property
    def module(self) -> nn.Module:
        return self._module

    def graph(self) -> list[LayerInfo]:
        return self._graph

    def default_cut(self) -> int:
        """End of the backbone — the layer before the neck's first upsample."""
        return backbone_cut(self._graph)

    def probe_shapes(self, imgsz: int = 640) -> list[tuple[int, ...] | None]:
        return probe_output_shapes(self._module, imgsz=imgsz)

    def run_span(self, start: int, stop: int, x: Any, cache: dict[int, torch.Tensor]) -> Any:
        layers = self._module.model[start : stop + 1]
        for m, info in zip(layers, self._graph[start : stop + 1], strict=True):
            if not info.is_sequential:
                if len(info.sources) == 1:
                    x = cache[info.sources[0]]
                else:
                    x = [x if s == info.index - 1 else cache[s] for s in info.sources]
            x = m(x)
            if info.index in self._cache_set:
                cache[info.index] = x
        return x


def _is_ultralytics(model: Any) -> bool:
    """A flat module list whose entries carry the ``i``/``f`` wiring."""
    inner = getattr(model, "model", None)
    if not isinstance(inner, nn.Module) or len(inner) == 0:  # type: ignore[arg-type]
        return False
    first = inner[0]
    return hasattr(first, "i") and hasattr(first, "f")


register_adapter("ultralytics", _is_ultralytics, UltralyticsAdapter)

# Fail fast if the class ever drifts from the protocol.
_: type[ModelAdapter] = UltralyticsAdapter
