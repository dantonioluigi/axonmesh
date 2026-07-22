"""Choose the routing threshold by measuring it, on unlabelled frames.

``conf_high`` decides whether the edge answers a frame or escalates it, and it
is compared against a detector's confidence score — which is not a probability.
0.6 does not mean "right six times in ten", the mapping differs between models
and shifts with the scene, so a threshold picked by intuition is a guess that
does not transfer.

The fix is not to calibrate the score. It is to stop needing it calibrated: for
every candidate threshold, measure what actually happens. The question a
cascade is really asking is *would the cloud have disagreed?*, and that can be
answered without a single label — run both models on frames from the
deployment, and compare their answers to each other.

That matters practically. mAP needs an annotated dataset, which a site rarely
has; agreement needs an hour of unlabelled footage from the camera that will be
running, which is the distribution the threshold has to hold on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import torch.nn as nn

from .cascade import FrameConfidence, jpeg_roundtrip, mean_confidence
from .measure import to_input_tensor
from .policy import Detection


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detection_agreement(
    edge: list[Detection], cloud: list[Detection], threshold: float = 0.5
) -> float:
    """How much two detection sets say the same thing, in [0, 1].

    Symmetric F1 over greedily IoU-matched boxes of the same class: 1.0 when
    the two agree completely, 0.0 when nothing matches. Two empty sets agree —
    a frame where both models see nothing is a frame the cascade handles
    perfectly, and scoring it 0 would push the threshold toward escalating
    exactly the frames that need it least.
    """
    if not edge and not cloud:
        return 1.0
    if not edge or not cloud:
        return 0.0
    unmatched = list(range(len(cloud)))
    matched = 0
    for detection in sorted(edge, key=lambda d: -d.conf):
        best, best_iou = None, threshold
        for index in unmatched:
            other = cloud[index]
            if other.cls_id != detection.cls_id:
                continue
            overlap = iou(detection.xyxyn, other.xyxyn)
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best is not None:
            unmatched.remove(best)
            matched += 1
    return 2 * matched / (len(edge) + len(cloud))


@dataclass(frozen=True)
class FrameProbe:
    """One calibration frame: what the edge said, and what it cost to be wrong."""

    confidence: float | None  # the statistic the threshold is compared against
    agreement: float  # edge vs cloud, if the edge answers this frame alone
    detection_bytes: int  # wire cost of answering locally
    frame_bytes: int  # wire cost of escalating


@dataclass(frozen=True)
class ThresholdPoint:
    """What one candidate threshold would cost and retain over the probe set."""

    threshold: float
    mean_bytes: float
    agreement: float  # expected agreement with an always-escalate deployment
    escalation_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "mean_bytes": self.mean_bytes,
            "agreement": self.agreement,
            "escalation_rate": self.escalation_rate,
        }


@torch.no_grad()
def probe_frames(
    edge: nn.Module,
    cloud: nn.Module,
    images: list[Path],
    imgsz: int = 640,
    conf: float = 0.25,
    iou_threshold: float = 0.45,
    jpeg_quality: int = 50,
    frame_confidence: FrameConfidence = mean_confidence,
    device: str = "cpu",
) -> list[FrameProbe]:
    """Run both models over the calibration frames once, recording both costs.

    Both models run per frame, which is the expensive part and is why this is a
    calibration step rather than something the edge does live. The cloud sees
    the frame as the wire would deliver it — decoded from JPEG — so the
    agreement is against the answer an escalation would really have produced.
    """
    from .cascade import _detections_from
    from .policy import serialize_detections

    edge, cloud = edge.eval(), cloud.eval()
    dev = torch.device(device)
    probes = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not read image {path}")
        x = to_input_tensor(image, imgsz).to(dev)

        edge_dets = _detections_from(_primary(edge(x)), imgsz, conf, iou_threshold)
        frame_bytes, received = jpeg_roundtrip(x[0], jpeg_quality)
        cloud_dets = _detections_from(
            _primary(cloud(received.unsqueeze(0))), imgsz, conf, iou_threshold
        )
        probes.append(
            FrameProbe(
                confidence=frame_confidence(edge_dets),
                agreement=detection_agreement(edge_dets, cloud_dets),
                detection_bytes=len(serialize_detections(edge_dets)),
                frame_bytes=frame_bytes,
            )
        )
    return probes


def _primary(raw: object) -> torch.Tensor:
    from .split import primary_output

    return primary_output(raw)


def sweep_thresholds(probes: list[FrameProbe], steps: int = 21) -> list[ThresholdPoint]:
    """What every threshold from 0 to 1 would have cost on the probed frames.

    A frame below the threshold escalates: it costs a frame and returns the
    cloud's own answer, so it agrees with an always-escalate deployment by
    construction. A frame above answers locally, cheaply, and agrees only as
    much as it measurably did.
    """
    if not probes:
        raise ValueError("no frames probed; calibration needs frames from the deployment")
    points = []
    for step in range(steps):
        threshold = step / (steps - 1)
        agreements, costs, escalated = [], [], 0
        for probe in probes:
            local = probe.confidence is not None and probe.confidence >= threshold
            agreements.append(probe.agreement if local else 1.0)
            costs.append(probe.detection_bytes if local else probe.frame_bytes)
            escalated += not local
        points.append(
            ThresholdPoint(
                threshold=threshold,
                mean_bytes=sum(costs) / len(costs),
                agreement=sum(agreements) / len(agreements),
                escalation_rate=escalated / len(probes),
            )
        )
    return points


def choose_threshold(
    points: list[ThresholdPoint],
    max_bytes: float | None = None,
    min_agreement: float | None = None,
) -> ThresholdPoint:
    """The best threshold meeting the given constraint.

    With a bandwidth ceiling, the one that fits and agrees most. With an
    agreement floor, the cheapest that clears it. With both, the cheapest
    option that satisfies each. A constraint nothing satisfies is an error
    rather than a silent fallback — a returned threshold implies its budget was
    met, and quietly returning the closest miss makes a deployment believe a
    promise nobody kept.
    """
    if max_bytes is None and min_agreement is None:
        raise ValueError("give a constraint: max_bytes, min_agreement, or both")
    fits = [
        point
        for point in points
        if (max_bytes is None or point.mean_bytes <= max_bytes)
        and (min_agreement is None or point.agreement >= min_agreement)
    ]
    if not fits:
        best_bytes = min(points, key=lambda p: p.mean_bytes)
        best_agreement = max(points, key=lambda p: p.agreement)
        raise ValueError(
            "no threshold satisfies the constraint; the cheapest option costs "
            f"{best_bytes.mean_bytes:.0f} bytes/frame and the most faithful agrees "
            f"{best_agreement.agreement:.3f}"
        )
    if max_bytes is not None:
        return max(fits, key=lambda p: (p.agreement, -p.mean_bytes))
    return min(fits, key=lambda p: (p.mean_bytes, -p.agreement))


def to_markdown(points: list[ThresholdPoint]) -> str:
    """The sweep as a table, so the trade-off is read rather than trusted."""
    lines = [
        "| threshold | KB/frame | agreement | escalated |",
        "|---:|---:|---:|---:|",
    ]
    for point in points:
        lines.append(
            f"| {point.threshold:.2f} | {point.mean_bytes / 1024:.3f} "
            f"| {point.agreement:.3f} | {point.escalation_rate:.0%} |"
        )
    return "\n".join(lines)


def agreement_bytes_pareto(points: list[ThresholdPoint]) -> list[ThresholdPoint]:
    """Thresholds not beaten on both axes at once."""
    return [
        point
        for point in points
        if not any(
            other is not point
            and other.mean_bytes <= point.mean_bytes
            and other.agreement >= point.agreement
            and (other.mean_bytes < point.mean_bytes or other.agreement > point.agreement)
            for other in points
        )
    ]


__all__ = [
    "FrameProbe",
    "ThresholdPoint",
    "agreement_bytes_pareto",
    "choose_threshold",
    "detection_agreement",
    "iou",
    "probe_frames",
    "sweep_thresholds",
    "to_markdown",
]
