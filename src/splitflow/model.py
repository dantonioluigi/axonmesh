"""``SplitModel`` — the one object you need to split, run and deploy a model.

Everything under this facade already worked, but as separate pieces you had to
wire yourself: resolve an adapter, build a runner, price the cuts, pick a
transport, emit Kubernetes config. ``SplitModel`` is the front door:

    model = SplitModel(YOLO("yolo11l.pt").model)
    model.split(transport=Int8Transport(compress=True))   # configure the cut
    detections = model.run(x)                             # edge -> wire -> cloud
    cr = model.deploy(name="detector", image="ghcr.io/you/cloud:0.6.0",
                      model_url="https://store/model.pt")

It is architecture-agnostic: the model is resolved through the adapter registry
(:mod:`splitflow.adapters`), so anything with an adapter works the same way.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .adapters import ModelAdapter, adapter_for
from .planner import CutOption, budget_bytes_per_frame, enumerate_cuts, plan_cut
from .split import SplitRunner, Transport
from .topology import LayerInfo

_CR_API_VERSION = "split.dev/v1alpha1"
_CR_KIND = "SplitInference"


class SplitModel:
    """A model plus a split configuration, with everything you do to it."""

    def __init__(
        self,
        model: nn.Module | ModelAdapter,
        cut: int | None = None,
        transport: Transport | None = None,
        imgsz: int = 640,
    ) -> None:
        self.adapter: ModelAdapter = adapter_for(model)
        self.imgsz = imgsz
        self._runner = SplitRunner(self.adapter, cut=cut, transport=transport)

    # -- introspection ---------------------------------------------------

    @property
    def graph(self) -> list[LayerInfo]:
        return self._runner.graph

    @property
    def cut(self) -> int:
        """Index of the last layer that runs on the edge."""
        return self._runner.cut

    @property
    def wire(self) -> tuple[int, ...]:
        """Layer outputs that must cross the wire at the current cut."""
        return self._runner.wire

    @property
    def transport(self) -> Transport | None:
        return self._runner.transport

    @property
    def stats(self):
        """Bytes shipped so far (see :class:`~splitflow.split.WireStats`)."""
        return self._runner.stats

    def cut_options(self, imgsz: int | None = None) -> list[CutOption]:
        """Every candidate cut, priced in wire bytes and edge compute."""
        return enumerate_cuts(self.adapter.module, imgsz=imgsz or self.imgsz)

    # -- configuration ---------------------------------------------------

    def split(
        self,
        cut: int | None = None,
        transport: Transport | None = None,
    ) -> SplitModel:
        """Set the cut and/or the wire codec. Returns self, so it chains."""
        self._runner = SplitRunner(
            self.adapter,
            cut=self.cut if cut is None else cut,
            transport=self.transport if transport is None else transport,
        )
        return self

    def plan(
        self,
        bandwidth_mbps: float,
        fps: float,
        transport: str = "int8",
        imgsz: int | None = None,
    ) -> CutOption | None:
        """Choose the cut that fits a link budget, and apply it.

        Returns the chosen option, or ``None`` when nothing fits (in which case
        the current cut is left alone — see :mod:`splitflow.planner`).
        """
        budget = budget_bytes_per_frame(bandwidth_mbps, fps)
        choice = plan_cut(self.cut_options(imgsz), budget, transport=transport)
        if choice is not None:
            self.split(cut=choice.cut)
        return choice

    # -- execution -------------------------------------------------------

    def edge(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run the edge half; returns the wire set."""
        return self._runner.edge(x)

    def cloud(self, wire: dict[int, torch.Tensor]) -> Any:
        """Run the cloud half from a received wire set."""
        return self._runner.cloud(wire)

    def run(self, x: torch.Tensor) -> Any:
        """Full split inference: edge → wire (codec) → cloud."""
        return self._runner(x)

    __call__ = run

    # -- deployment ------------------------------------------------------

    def deploy(
        self,
        name: str,
        image: str,
        model_url: str,
        bottleneck_url: str | None = None,
        replicas: int = 1,
        namespace: str | None = None,
        auto: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Render the ``SplitInference`` custom resource for this configuration.

        Emits declarative config rather than talking to a cluster: apply it with
        ``kubectl`` (or commit it) and the operator reconciles the rest. By
        default the CR pins the current cut; pass ``auto=(mbps, fps)`` to let the
        edge re-plan against a live budget instead.
        """
        cut: dict[str, Any] = (
            {"mode": "auto", "auto": {"bandwidthMbps": auto[0], "fps": auto[1]}}
            if auto
            else {"mode": "fixed", "fixed": self.cut}
        )
        metadata: dict[str, Any] = {"name": name}
        if namespace:
            metadata["namespace"] = namespace
        spec: dict[str, Any] = {
            "model": {"url": model_url},
            "cut": cut,
            "cloud": {"image": image, "replicas": replicas, "imgsz": self.imgsz},
        }
        if bottleneck_url:
            spec["bottleneck"] = {"url": bottleneck_url}
        return {
            "apiVersion": _CR_API_VERSION,
            "kind": _CR_KIND,
            "metadata": metadata,
            "spec": spec,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        codec = type(self.transport).__name__ if self.transport else "none"
        return f"SplitModel(cut={self.cut}, wire={list(self.wire)}, transport={codec})"
