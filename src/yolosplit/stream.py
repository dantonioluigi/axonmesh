"""Simulate the adaptive edge→cloud stream over a directory of frames.

For every frame the detector runs locally (edge), the policy picks a
transmission mode, and the simulator counts the bytes that mode would ship:
serialised detections, (bottlenecked) feature tensors, or the full JPEG.
The all-JPEG cost is recorded per frame too, so the summary can state exactly
how much bandwidth the adaptive scheme saves over always-streaming frames.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .measure import IMAGE_SUFFIXES, jpeg_nbytes, to_input_tensor
from .policy import AdaptivePolicy, Detection, Mode, serialize_detections
from .split import SplitRunner

Inferer = Callable[[np.ndarray], list[Detection]]
FeatureBytes = Callable[[np.ndarray], int]


@dataclass(frozen=True)
class FrameReport:
    name: str
    mode: Mode
    nbytes: int
    jpeg_bytes: int  # what always-JPEG would have shipped for this frame
    frame_conf: float | None
    retrain: bool
    reason: str


def simulate_stream(
    frames: Iterable[tuple[str, np.ndarray]],
    infer: Inferer,
    policy: AdaptivePolicy,
    feature_bytes: FeatureBytes,
    quality: int = 85,
) -> list[FrameReport]:
    """Run the policy over frames and price each transmission."""
    reports = []
    for name, image in frames:
        detections = infer(image)
        decision = policy.decide(detections)
        jpeg = jpeg_nbytes(image, quality)
        if decision.mode is Mode.DETECTIONS:
            nbytes = len(serialize_detections(detections))
        elif decision.mode is Mode.FEATURES:
            nbytes = feature_bytes(image)
        else:
            nbytes = jpeg
        reports.append(
            FrameReport(
                name=name,
                mode=decision.mode,
                nbytes=nbytes,
                jpeg_bytes=jpeg,
                frame_conf=decision.frame_conf,
                retrain=decision.retrain,
                reason=decision.reason,
            )
        )
    return reports


def summarize_stream(reports: list[FrameReport]) -> dict[str, float]:
    """Totals per mode plus the bandwidth ratio vs always-JPEG."""
    total = sum(r.nbytes for r in reports)
    baseline = sum(r.jpeg_bytes for r in reports)
    summary: dict[str, float] = {
        "frames": len(reports),
        "total_bytes": total,
        "baseline_jpeg_bytes": baseline,
        "saved_vs_jpeg": 1 - total / baseline if baseline else 0.0,
        "retrain_frames": sum(r.retrain for r in reports),
    }
    for mode in Mode:
        summary[f"frames_{mode.value}"] = sum(r.mode is mode for r in reports)
    return summary


def iter_image_frames(
    images_dir: str | Path, limit: int | None = None
) -> Iterable[tuple[str, np.ndarray]]:
    """Yield ``(name, BGR image)`` for every image in a directory, sorted."""
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[
        :limit
    ]
    if not paths:
        raise FileNotFoundError(f"no images found in {images_dir}")
    for p in paths:
        image = cv2.imread(str(p))
        if image is None:
            raise ValueError(f"could not read image {p}")
        yield p.name, image


def yolo_inferer(yolo, imgsz: int = 640, conf: float = 0.25) -> Inferer:
    """Adapt an ultralytics ``YOLO`` wrapper to the ``Inferer`` interface."""

    def infer(image: np.ndarray) -> list[Detection]:
        result = yolo.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
        boxes = result.boxes
        return [
            Detection(int(c), float(p), tuple(map(float, xyxyn)))
            for c, p, xyxyn in zip(boxes.cls, boxes.conf, boxes.xyxyn, strict=True)
        ]

    return infer


def transport_feature_bytes(runner: SplitRunner, imgsz: int = 640) -> FeatureBytes:
    """Price the FEATURES mode with the runner's transport (must not be None)."""
    if runner.transport is None:
        raise ValueError("runner needs a transport to price the FEATURES mode")

    def feature_bytes(image: np.ndarray) -> int:
        wire = runner.edge(to_input_tensor(image, imgsz))
        _, nbytes = runner.transport(wire)
        return nbytes

    return feature_bytes
