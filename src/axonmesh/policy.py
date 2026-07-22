"""Adaptive transmission policy: send only the information the frame deserves.

The edge runs the full detector locally (as it already does in production).
Per frame, the policy picks one of three modes:

- ``DETECTIONS`` — confident frame: ship the final boxes (11 bytes each).
- ``FEATURES`` — uncertain frame: ship the (bottlenecked) wire tensors so the
  cloud can re-run the heavy half in full precision.
- ``FRAME`` — drift or very low confidence: ship the full JPEG, which is also
  enqueued as a retraining candidate (the hard cases that will become the next
  dataset version).

The frame confidence is the *minimum* detection confidence: one uncertain
object is enough to warrant escalation. Drift detection is pluggable; the
default tracks an EMA of frame confidence and fires when it sinks below a
threshold — a deliberately simple stand-in for the production drift detector.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum


class Mode(str, Enum):
    DETECTIONS = "detections"
    FEATURES = "features"
    FRAME = "frame"


@dataclass(frozen=True)
class Detection:
    """One detection with normalised xyxy coordinates in [0, 1]."""

    cls_id: int
    conf: float
    xyxyn: tuple[float, float, float, float]


_DET_STRUCT = struct.Struct("<BH4H")
_COUNT_STRUCT = struct.Struct("<H")
_U16 = 65535


def serialize_detections(detections: list[Detection]) -> bytes:
    """Compact wire format: 2-byte count + 11 bytes per detection.

    Class fits u8, confidence and coordinates are u16 fixed-point in [0, 1].
    """
    parts = [_COUNT_STRUCT.pack(len(detections))]
    for d in detections:
        coords = [round(min(max(c, 0.0), 1.0) * _U16) for c in d.xyxyn]
        parts.append(_DET_STRUCT.pack(d.cls_id, round(d.conf * _U16), *coords))
    return b"".join(parts)


def deserialize_detections(payload: bytes) -> list[Detection]:
    (count,) = _COUNT_STRUCT.unpack_from(payload, 0)
    detections = []
    for k in range(count):
        cls_id, conf, *coords = _DET_STRUCT.unpack_from(
            payload, _COUNT_STRUCT.size + k * _DET_STRUCT.size
        )
        detections.append(Detection(cls_id, conf / _U16, tuple(c / _U16 for c in coords)))
    return detections


@dataclass
class ConfidenceEMADrift:
    """Drift heuristic: EMA of frame confidence sinking below a threshold.

    Frames with no detections count as confidence 0 — an industrial station
    that suddenly sees nothing is more likely drifting than idle. ``warmup`` frames are needed
    before the detector may fire, so a cold start is not misread as drift.
    """

    alpha: float = 0.05
    threshold: float = 0.5
    warmup: int = 20
    ema: float | None = field(default=None, init=False)
    frames: int = field(default=0, init=False)

    def update(self, frame_conf: float | None) -> bool:
        observed = 0.0 if frame_conf is None else frame_conf
        self.ema = (
            observed if self.ema is None else ((1 - self.alpha) * self.ema + self.alpha * observed)
        )
        self.frames += 1
        return self.frames >= self.warmup and self.ema < self.threshold


@dataclass(frozen=True)
class Decision:
    mode: Mode
    frame_conf: float | None
    drifting: bool
    retrain: bool
    reason: str


class AdaptivePolicy:
    """Threshold policy over frame confidence, with drift override."""

    def __init__(
        self,
        conf_high: float = 0.75,
        conf_low: float = 0.4,
        drift: ConfidenceEMADrift | None = None,
    ) -> None:
        if not 0.0 <= conf_low <= conf_high <= 1.0:
            raise ValueError(f"need 0 <= conf_low <= conf_high <= 1, got {conf_low}, {conf_high}")
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.drift = drift if drift is not None else ConfidenceEMADrift()

    def decide(self, detections: list[Detection]) -> Decision:
        """Route a frame by its least confident detection."""
        return self.decide_confidence(min((d.conf for d in detections), default=None))

    def decide_confidence(self, frame_conf: float | None) -> Decision:
        """Route a frame by a confidence already summarised from its detections.

        How to reduce a frame's detections to one number is scene-dependent —
        the minimum suits a station with a few known objects and is close to a
        constant on a crowded one, where some box is always marginal. Callers
        that know their scenes pick the statistic (see
        :mod:`axonmesh.cascade`); the thresholds live here either way.
        """
        drifting = self.drift.update(frame_conf)
        if drifting:
            return Decision(Mode.FRAME, frame_conf, True, True, "drift detected")
        if frame_conf is None:
            return Decision(Mode.FEATURES, None, False, False, "no detections")
        if frame_conf >= self.conf_high:
            return Decision(Mode.DETECTIONS, frame_conf, False, False, "confident")
        if frame_conf >= self.conf_low:
            return Decision(Mode.FEATURES, frame_conf, False, False, "medium confidence")
        return Decision(Mode.FRAME, frame_conf, False, True, "low confidence")
