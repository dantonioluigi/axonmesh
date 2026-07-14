"""End-to-end accuracy of split inference: baseline mAP vs split+quantised mAP.

``split_inference`` transparently reroutes a DetectionModel's forward pass
through a :class:`~yolosplit.split.SplitRunner`, so the standard ultralytics
``val()``/``predict()`` pipelines (dataloading, letterboxing, NMS, metrics)
run unchanged on top of the split model.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch.nn as nn

from .split import SplitRunner, Transport


@contextmanager
def split_inference(
    det_model: nn.Module,
    cut: int | None = None,
    transport: Transport | None = None,
):
    """Patch ``det_model._predict_once`` to run split inference; restore on exit.

    Yields the :class:`SplitRunner`, whose ``stats`` accumulate the bytes each
    frame shipped across the simulated wire.
    """
    runner = SplitRunner(det_model, cut=cut, transport=transport)
    previous = det_model.__dict__.get("_predict_once")

    def patched(x, profile=False, visualize=False, embed=None):
        return runner(x)

    det_model._predict_once = patched
    try:
        yield runner
    finally:
        if previous is None:
            det_model.__dict__.pop("_predict_once", None)
        else:
            det_model._predict_once = previous


@dataclass(frozen=True)
class MapComparison:
    """Baseline vs split validation results on the same dataset."""

    cut: int
    baseline_map50: float
    baseline_map50_95: float
    split_map50: float
    split_map50_95: float
    frames: int
    wire_mean_bytes: float

    @property
    def delta_map50(self) -> float:
        return self.split_map50 - self.baseline_map50

    @property
    def delta_map50_95(self) -> float:
        return self.split_map50_95 - self.baseline_map50_95

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "delta_map50": self.delta_map50,
            "delta_map50_95": self.delta_map50_95,
        }


def compare_map(  # pragma: no cover - requires model weights and a dataset
    weights: str,
    data: str,
    cut: int | None = None,
    transport: Transport | None = None,
    imgsz: int = 640,
    device: str = "cpu",
    **val_kwargs: Any,
) -> MapComparison:
    """Validate ``weights`` on ``data`` twice: unsplit, then split at ``cut``.

    Args:
        weights: path to a ``.pt`` checkpoint.
        data: ultralytics dataset YAML.
        cut: last edge layer; defaults to the end of the backbone.
        transport: wire simulation (e.g. ``Int8Transport()``); ``None`` is lossless.
        val_kwargs: forwarded to ``YOLO.val`` (``batch``, ``split``, ...).
    """
    from ultralytics import YOLO

    baseline = YOLO(weights).val(data=data, imgsz=imgsz, device=device, verbose=False, **val_kwargs)

    yolo = YOLO(weights)
    with split_inference(yolo.model, cut=cut, transport=transport) as runner:
        split_res = yolo.val(data=data, imgsz=imgsz, device=device, verbose=False, **val_kwargs)

    return MapComparison(
        cut=runner.cut,
        baseline_map50=float(baseline.box.map50),
        baseline_map50_95=float(baseline.box.map),
        split_map50=float(split_res.box.map50),
        split_map50_95=float(split_res.box.map),
        frames=runner.stats.frames,
        wire_mean_bytes=runner.stats.mean_bytes,
    )
