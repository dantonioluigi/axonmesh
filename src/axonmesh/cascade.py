"""Edge-first inference: consult the cloud only for the frames that need it.

The bandwidth case for shipping *compressed features* does not hold at these
cuts — a JPEG frame is smaller and costs no accuracy (``docs/validation.md``).
What is left is not sending anything most of the time.

A small model runs on the edge. When it is confident, the wire carries its
detections — eleven bytes each — and that is the answer. When it is not, the
frame is escalated and a larger cloud model produces the answer instead. The
saving does not come from a better codec; it comes from the frames that never
travel.

Both axes have to be measured together, which is the lesson of the codec work:
a policy that saves 99% of the bandwidth by discarding a third of the
detections is the same failure wearing different clothes. :func:`run_cascade`
reports bytes *and* routes predictions through ultralytics' own validator, so
the mAP is the real one.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn

from .policy import AdaptivePolicy, Decision, Detection, Mode, serialize_detections
from .split import primary_output

FrameConfidence = Callable[[list[Detection]], float | None]


def min_confidence(detections: list[Detection]) -> float | None:
    """The least confident detection in the frame — escalate if any object is doubtful.

    Right when a frame holds a few known objects and every one of them matters.
    On a crowded scene it is close to a constant: there is always one marginal
    box near the detector's own threshold, so the frame is never confident and
    the policy escalates everything.
    """
    return min((d.conf for d in detections), default=None)


def mean_confidence(detections: list[Detection]) -> float | None:
    """Average confidence — robust to one marginal box, blind to one bad one."""
    return sum(d.conf for d in detections) / len(detections) if detections else None


def quantile_confidence(q: float = 0.25) -> FrameConfidence:
    """The ``q``-quantile of detection confidence: min's intent, without its fragility.

    A single weak box no longer condemns the frame, but a frame where a quarter
    of the objects are doubtful still escalates.
    """

    def confidence(detections: list[Detection]) -> float | None:
        if not detections:
            return None
        confidences = sorted(d.conf for d in detections)
        return confidences[min(int(q * len(confidences)), len(confidences) - 1)]

    return confidence


@dataclass
class CascadeStats:
    """Per-frame routing and wire cost."""

    modes: list[Mode] = field(default_factory=list)
    frame_bytes: list[int] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return len(self.modes)

    @property
    def mean_bytes(self) -> float:
        return sum(self.frame_bytes) / self.frames if self.frames else 0.0

    @property
    def escalated(self) -> int:
        return sum(mode is not Mode.DETECTIONS for mode in self.modes)

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.frames if self.frames else 0.0


def _detections_from(pred: torch.Tensor, imgsz: int, conf: float, iou: float) -> list[Detection]:
    from .server import _import_nms

    boxes = _import_nms()(pred, conf_thres=conf, iou_thres=iou)[0]
    return [
        Detection(int(b[5]), float(b[4]), (b[0] / imgsz, b[1] / imgsz, b[2] / imgsz, b[3] / imgsz))
        for b in boxes.tolist()
    ]


def jpeg_roundtrip(frame: torch.Tensor, quality: int) -> tuple[int, torch.Tensor]:
    """Encode one frame as JPEG and decode it back: its wire size, and what arrives.

    Both halves of the pair matter, and charging for one without applying the
    other is the easiest way to flatter an escalation path. A cloud model
    scoring an escalated frame must score the *decoded* image, because that is
    what a q50 wire actually delivers — not the pristine tensor the edge held.
    """
    image = (frame.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    ok, buf = cv2.imencode(".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:  # pragma: no cover - cv2 encodes any well-formed array
        raise RuntimeError("could not JPEG-encode a frame")
    decoded = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    received = torch.from_numpy(decoded.copy()).permute(2, 0, 1).float().div(255)
    return int(buf.nbytes), received.to(frame.device)


class Cascade(nn.Module):
    """Route each frame to the edge model or the cloud model, and price the wire.

    Callable like a detection model: given a batch it returns predictions of
    the same shape, so ultralytics' validator scores the cascade end to end
    without knowing one exists. Both halves share an image size, hence an
    anchor grid, which is what makes the routed rows concatenable.
    """

    def __init__(
        self,
        edge: nn.Module,
        cloud: nn.Module,
        policy: AdaptivePolicy | None = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        jpeg_quality: int = 50,
        frame_confidence: FrameConfidence = min_confidence,
    ) -> None:
        super().__init__()
        self.edge = edge.eval()
        self.cloud = cloud.eval()
        self.policy = policy if policy is not None else AdaptivePolicy()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.jpeg_quality = jpeg_quality
        self.frame_confidence = frame_confidence
        self.stats = CascadeStats()
        self.decisions: list[Decision] = []

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *_: object, **__: object) -> torch.Tensor:
        # ultralytics hands its backend augment/visualize/embed; a cascade has
        # no meaning for any of them, and refusing them would only mean the
        # validator cannot call it.
        edge_out = primary_output(self.edge(x))
        routed = edge_out.clone()
        escalate: list[int] = []
        received: list[torch.Tensor] = []

        for i in range(x.shape[0]):
            detections = _detections_from(edge_out[i : i + 1], self.imgsz, self.conf, self.iou)
            # The policy owns the thresholds; which statistic they are applied
            # to is the caller's, because summarising a frame's confidence is a
            # scene-dependent question.
            decision = self.policy.decide_confidence(self.frame_confidence(detections))
            self.decisions.append(decision)
            self.stats.modes.append(decision.mode)
            if decision.mode is Mode.DETECTIONS:
                self.stats.frame_bytes.append(len(serialize_detections(detections)))
            else:
                nbytes, arrived = jpeg_roundtrip(x[i], self.jpeg_quality)
                self.stats.frame_bytes.append(nbytes)
                escalate.append(i)
                received.append(arrived)

        if escalate:
            # The cloud scores what the wire delivered, not what the edge held:
            # an escalated frame has been through the codec it was charged for.
            index = torch.tensor(escalate, device=x.device)
            routed[index] = primary_output(self.cloud(torch.stack(received)))
        return routed


@contextmanager
def cascade_inference(cloud_model: nn.Module, cascade: Cascade):
    """Route a YOLO wrapper's forward pass through ``cascade``; restore on exit.

    Patching ``_predict_once`` rather than swapping the model out keeps every
    part of ultralytics' pipeline — dataloading, letterboxing, NMS, metrics —
    running exactly as it does for the two endpoint measurements, which is what
    makes the three mAP numbers comparable.
    """
    previous = cloud_model.__dict__.get("_predict_once")

    def patched(x, profile=False, visualize=False, embed=None):
        return cascade(x)

    cloud_model._predict_once = patched
    try:
        yield cascade
    finally:
        if previous is None:
            cloud_model.__dict__.pop("_predict_once", None)
        else:
            cloud_model._predict_once = previous


@dataclass(frozen=True)
class CascadeResult:
    """What the cascade cost and what it was worth, against both endpoints."""

    edge_only_map50_95: float
    cloud_only_map50_95: float
    cascade_map50_95: float
    edge_only_bytes: float
    cloud_only_bytes: float
    cascade_bytes: float
    escalation_rate: float
    frames: int

    @property
    def map_kept(self) -> float:
        """Share of the cloud-only accuracy the cascade retains."""
        return self.cascade_map50_95 / self.cloud_only_map50_95 if self.cloud_only_map50_95 else 0.0

    @property
    def bytes_saved(self) -> float:
        """Share of the always-escalate bandwidth the cascade avoids."""
        return 1 - self.cascade_bytes / self.cloud_only_bytes if self.cloud_only_bytes else 0.0

    def to_dict(self) -> dict[str, float]:
        from dataclasses import asdict

        return asdict(self) | {"map_kept": self.map_kept, "bytes_saved": self.bytes_saved}


def mean_jpeg_bytes(images: list[np.ndarray], imgsz: int, quality: int) -> float:
    """Baseline wire cost: every frame shipped, letterboxed, as JPEG."""
    from .measure import jpeg_nbytes, letterbox

    return sum(jpeg_nbytes(letterbox(im, imgsz), quality) for im in images) / len(images)
