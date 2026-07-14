"""Run an ultralytics DetectionModel as two halves: edge (backbone) and cloud.

The edge half runs layers ``0..cut`` and produces the *wire set*: every cached
tensor a later layer still needs (see :mod:`yolosplit.topology`). An optional
transport (see :mod:`yolosplit.transport`) simulates shipping those tensors —
quantising, serialising and counting bytes. The cloud half resumes from the
received tensors and produces the exact output the unsplit model would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.nn as nn

from .topology import LayerInfo, backbone_cut, build_graph, wire_indices


class Transport(Protocol):
    """Simulates the edge→cloud link for the wire tensors.

    Returns the tensors as reconstructed on the cloud side, plus the number of
    bytes that crossed the wire.
    """

    def __call__(
        self, wire: dict[int, torch.Tensor]
    ) -> tuple[dict[int, torch.Tensor], int]: ...  # pragma: no cover


@dataclass
class WireStats:
    """Bytes shipped across the wire, accumulated over calls."""

    frames: int = 0
    total_bytes: int = 0
    bytes_per_frame: list[int] = field(default_factory=list)

    def record(self, nbytes: int) -> None:
        self.frames += 1
        self.total_bytes += nbytes
        self.bytes_per_frame.append(nbytes)

    @property
    def mean_bytes(self) -> float:
        return self.total_bytes / self.frames if self.frames else 0.0


def raw_nbytes(wire: dict[int, torch.Tensor]) -> int:
    """Size of the wire set shipped as-is (no quantisation)."""
    return sum(t.numel() * t.element_size() for t in wire.values())


class SplitRunner:
    """Split a DetectionModel at ``cut`` and run the two halves.

    Calling the runner is functionally identical to calling the model's own
    ``_predict_once``: same modules, same order, same output. With
    ``transport=None`` the result is numerically identical; with a lossy
    transport (e.g. INT8) the cloud half runs on reconstructed tensors.

    Args:
        det_model: an ultralytics DetectionModel (``YOLO(...).model``).
        cut: index of the last edge layer. Defaults to the end of the backbone.
        transport: optional wire simulation applied between the halves.
    """

    def __init__(
        self,
        det_model: nn.Module,
        cut: int | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.det_model = det_model
        self.graph: list[LayerInfo] = build_graph(det_model)
        self.cut = backbone_cut(self.graph) if cut is None else cut
        self.wire = wire_indices(self.graph, self.cut)
        self.transport = transport
        self.stats = WireStats()

    def _run_span(
        self,
        start: int,
        stop: int,
        x: Any,
        cache: dict[int, torch.Tensor],
    ) -> Any:
        """Run modules ``start..stop`` inclusive, mirroring BaseModel._predict_once."""
        save = self.det_model.save
        span = zip(
            self.det_model.model[start : stop + 1], self.graph[start : stop + 1], strict=True
        )
        for m, info in span:
            if not info.is_sequential:
                if len(info.sources) == 1:
                    x = cache[info.sources[0]]
                else:
                    x = [x if s == info.index - 1 else cache[s] for s in info.sources]
            x = m(x)
            if info.index in save:
                cache[info.index] = x
        return x

    def edge(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run layers ``0..cut`` and return the wire set, keyed by layer index."""
        cache: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            out = self._run_span(0, self.cut, x, cache)
        cache[self.cut] = out
        return {i: cache[i] for i in self.wire}

    def cloud(self, wire: dict[int, torch.Tensor]) -> Any:
        """Resume from the received wire set and return the model output."""
        cache = dict(wire)
        # Seed with the last edge output; only consumed if layer cut+1 is
        # sequential, in which case it is part of the wire set by construction.
        x = wire.get(self.cut)
        with torch.no_grad():
            return self._run_span(self.cut + 1, len(self.graph) - 1, x, cache)

    def __call__(self, x: torch.Tensor) -> Any:
        wire = self.edge(x)
        if self.transport is not None:
            wire, nbytes = self.transport(wire)
        else:
            nbytes = raw_nbytes(wire)
        self.stats.record(nbytes)
        return self.cloud(wire)
