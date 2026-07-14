"""Graph introspection for ultralytics detection models.

Ultralytics models are a flat list of modules. Each module ``m`` carries
``m.i`` (its index) and ``m.f`` (the index or indices of the layers it
consumes, where ``-1`` means "the previous layer"). Skip connections
(``Concat``, ``Detect``) reference non-adjacent layers, so cutting the model
at layer ``k`` is *not* a matter of slicing ``model.model[:k]``: every cached
tensor that a layer after the cut still consumes must cross the wire too.

This module resolves the topology into absolute indices and computes, for any
cut point, the exact set of tensors the edge half must transmit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

#: Sentinel index for the model input image.
MODEL_INPUT = -1


class UnsupportedTopologyError(RuntimeError):
    """The model wiring cannot be handled by this splitter."""


@dataclass(frozen=True)
class LayerInfo:
    """One layer of the flat ultralytics graph, with resolved source indices."""

    index: int
    name: str
    sources: tuple[int, ...]
    params: int

    @property
    def is_sequential(self) -> bool:
        """True when the layer only consumes the output of the previous layer."""
        return self.sources == (self.index - 1,)


def _resolve_sources(module: nn.Module) -> tuple[int, ...]:
    raw = module.f if isinstance(module.f, (list, tuple)) else [module.f]
    sources = []
    for f in raw:
        sources.append(module.i + f if f < 0 else f)
    return tuple(sources)


def build_graph(det_model: nn.Module) -> list[LayerInfo]:
    """Resolve ``m.f``/``m.i`` wiring of a DetectionModel into absolute indices."""
    graph: list[LayerInfo] = []
    for m in det_model.model:
        sources = _resolve_sources(m)
        for s in sources:
            if not (MODEL_INPUT <= s < m.i):
                raise UnsupportedTopologyError(
                    f"layer {m.i} ({type(m).__name__}) references layer {s}, "
                    "which is not an earlier layer or the model input"
                )
        name = getattr(m, "type", type(m).__name__).rsplit(".", 1)[-1]
        graph.append(LayerInfo(index=m.i, name=name, sources=sources, params=int(m.np)))
    return graph


def wire_indices(graph: list[LayerInfo], cut: int) -> tuple[int, ...]:
    """Layer outputs that must cross the wire when the edge runs layers ``0..cut``.

    Returns the sorted indices of every layer ``i <= cut`` whose output is
    consumed by some layer ``j > cut``.
    """
    if not 0 <= cut < len(graph) - 1:
        raise ValueError(f"cut must be in [0, {len(graph) - 2}], got {cut}")
    needed: set[int] = set()
    for layer in graph[cut + 1 :]:
        for s in layer.sources:
            if s == MODEL_INPUT:
                raise UnsupportedTopologyError(
                    f"layer {layer.index} consumes the raw model input; "
                    f"cutting at {cut} would require shipping the input image"
                )
            if s <= cut:
                needed.add(s)
    return tuple(sorted(needed))


def backbone_cut(graph: list[LayerInfo]) -> int:
    """Index of the last backbone layer (the layer before the first upsample).

    In YOLO detection models the neck starts with an ``nn.Upsample``; everything
    before it is the backbone (for YOLO11 this is layer 10, the ``C2PSA``).
    """
    for layer in graph:
        if "Upsample" in layer.name:
            return layer.index - 1
    raise UnsupportedTopologyError("no Upsample layer found; cannot infer where the neck starts")


def probe_output_shapes(det_model: nn.Module, imgsz: int = 640) -> list[tuple[int, ...] | None]:
    """Output shape of every layer for a ``1x3x{imgsz}x{imgsz}`` input.

    The final layer (Detect) may return a non-tensor structure; its entry is
    ``None``. Used to price each candidate cut in bytes.
    """
    graph = build_graph(det_model)
    shapes: list[tuple[int, ...] | None] = []
    cache: dict[int, torch.Tensor] = {}
    device = next(det_model.parameters()).device
    x: torch.Tensor | list[torch.Tensor] = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for m, info in zip(det_model.model, graph, strict=True):
            if not info.is_sequential:
                if len(info.sources) == 1:
                    x = cache[info.sources[0]]
                else:
                    x = [x if s == info.index - 1 else cache[s] for s in info.sources]
            x = m(x)
            if isinstance(x, torch.Tensor):
                cache[info.index] = x
                shapes.append(tuple(x.shape))
            else:
                shapes.append(None)
    return shapes
