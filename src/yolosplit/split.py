"""Run a model as two halves: edge (up to the cut) and cloud (the rest).

The edge half runs layers ``0..cut`` and produces the *wire set*: every cached
tensor a later layer still needs (see :mod:`yolosplit.topology`). An optional
transport (see :mod:`yolosplit.transport`) simulates shipping those tensors —
quantising, serialising and counting bytes. The cloud half resumes from the
received tensors and produces the exact output the unsplit model would.

Which layers exist and how to run them comes from a
:class:`~yolosplit.adapters.ModelAdapter`, so this module is architecture-
agnostic; passing a bare model resolves an adapter automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.nn as nn

from .adapters import ModelAdapter, adapter_for
from .topology import LayerInfo, wire_indices


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
        model: nn.Module | ModelAdapter,
        cut: int | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.adapter: ModelAdapter = adapter_for(model)
        self.graph: list[LayerInfo] = self.adapter.graph()
        self.cut = self.adapter.default_cut() if cut is None else cut
        self.wire = wire_indices(self.graph, self.cut)
        self.transport = transport
        self.stats = WireStats()

    @property
    def det_model(self) -> nn.Module:
        """The underlying torch model (kept for backwards compatibility)."""
        return self.adapter.module

    def _run_span(
        self,
        start: int,
        stop: int,
        x: Any,
        cache: dict[int, torch.Tensor],
    ) -> Any:
        """Run layers ``start..stop`` inclusive via the adapter."""
        return self.adapter.run_span(start, stop, x, cache)

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
