"""Dynamic split point selection: pick the cut that fits the live constraints.

Every cut point has a price in two currencies: bytes on the wire (which tensors
must cross, at which dtype) and edge compute (how much of the network runs
before the cut). Given a bandwidth budget per frame — derived from the measured
link and the target FPS — the planner picks the cut that fits the budget while
off-loading as much compute as possible to the cloud.

This is the decision logic only: feeding it *live* bandwidth/GPU measurements
(and re-planning when they change) is the integration step that follows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch.nn as nn

from .adapters import ModelAdapter, adapter_for
from .topology import wire_indices

#: Serialisation overhead per INT8 tensor (header + per-tensor scale/zero-point).
INT8_TENSOR_OVERHEAD = 40


@dataclass(frozen=True)
class CutOption:
    """One candidate cut with its wire and edge-compute price."""

    cut: int
    wire: tuple[int, ...]
    wire_elements: int  # per-frame elements crossing the wire
    edge_params_share: float  # fraction of model parameters run on the edge

    def wire_bytes(self, transport: str = "int8") -> int:
        """Estimated bytes/frame for ``transport`` in {int8, fp16, fp32}.

        INT8 includes serialisation overhead; zlib is content-dependent and
        deliberately not estimated — measure it with ``splitflow measure``.
        """
        if transport == "int8":
            return self.wire_elements + INT8_TENSOR_OVERHEAD * len(self.wire)
        if transport == "fp16":
            return 2 * self.wire_elements
        if transport == "fp32":
            return 4 * self.wire_elements
        raise ValueError(f"unknown transport {transport!r}")


def enumerate_cuts(model: nn.Module | ModelAdapter, imgsz: int = 640) -> list[CutOption]:
    """Price every candidate cut of the model at the given input size.

    Works off the adapter, so any supported architecture can be planned for —
    pass a model and one is resolved, or pass an adapter directly.
    """
    adapter = adapter_for(model)
    graph = adapter.graph()
    shapes = adapter.probe_shapes(imgsz)
    total_params = sum(layer.params for layer in graph) or 1
    options = []
    edge_params = 0
    for cut in range(len(graph) - 1):
        edge_params += graph[cut].params
        try:
            wire = wire_indices(graph, cut)
        except Exception:  # a cut that would need the raw input is not a cut
            continue
        elements = sum(math.prod(shapes[i][1:]) for i in wire if shapes[i] is not None)
        options.append(
            CutOption(
                cut=cut,
                wire=wire,
                wire_elements=elements,
                edge_params_share=edge_params / total_params,
            )
        )
    return options


def budget_bytes_per_frame(bandwidth_mbps: float, fps: float) -> int:
    """Bytes each frame may spend on a link of ``bandwidth_mbps`` at ``fps``."""
    if bandwidth_mbps <= 0 or fps <= 0:
        raise ValueError("bandwidth and fps must be positive")
    return int(bandwidth_mbps * 1_000_000 / 8 / fps)


def plan_cut(
    options: list[CutOption],
    budget: int,
    transport: str = "int8",
) -> CutOption | None:
    """The cut that fits ``budget`` bytes/frame with the least edge compute.

    Returns ``None`` when no cut fits — meaning raw feature shipping cannot
    meet the link and a learned bottleneck (or plain JPEG) is required.
    Ties on edge share break toward fewer wire bytes.
    """
    feasible = [o for o in options if o.wire_bytes(transport) <= budget]
    if not feasible:
        return None
    return min(feasible, key=lambda o: (o.edge_params_share, o.wire_bytes(transport)))
