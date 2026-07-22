from __future__ import annotations

import pytest
import torch

from axonmesh.cascade import (
    Cascade,
    CascadeResult,
    jpeg_roundtrip,
    mean_confidence,
    min_confidence,
    quantile_confidence,
)
from axonmesh.policy import AdaptivePolicy, ConfidenceEMADrift, Detection, Mode


def detections(*confidences) -> list[Detection]:
    return [Detection(0, c, (0.1, 0.1, 0.2, 0.2)) for c in confidences]


def never_drifts() -> ConfidenceEMADrift:
    return ConfidenceEMADrift(warmup=10**9)


@pytest.mark.parametrize(
    ("statistic", "expected"),
    [(min_confidence, 0.2), (mean_confidence, 0.5), (quantile_confidence(0.25), 0.2)],
)
def test_frame_confidence_statistics(statistic, expected):
    assert statistic(detections(0.9, 0.8, 0.2, 0.1)) == pytest.approx(expected, abs=0.15)


@pytest.mark.parametrize("statistic", [min_confidence, mean_confidence, quantile_confidence(0.25)])
def test_a_frame_with_no_detections_has_no_confidence(statistic):
    """None is not zero: it routes to escalation rather than to a threshold."""
    assert statistic([]) is None


def test_min_confidence_is_hostage_to_one_marginal_box():
    """Why the statistic is configurable at all.

    Nine confident objects and one doubtful one: `min` calls the frame
    doubtful, so on a crowded scene the policy escalates everything and the
    cascade degenerates into always-send.
    """
    crowded = detections(*([0.95] * 9), 0.26)
    assert min_confidence(crowded) < 0.3
    assert mean_confidence(crowded) > 0.85


def smooth_frame() -> torch.Tensor:
    """A gradient, not noise: JPEG is built for natural images and cannot
    preserve uniform random pixels at all, so noise would make any fidelity
    assertion meaningless rather than strict."""
    ramp = torch.linspace(0, 1, 64)
    return torch.stack([ramp.expand(64, 64), ramp.expand(64, 64).T, ramp.expand(64, 64) * 0.5])


def test_jpeg_roundtrip_returns_both_the_size_and_the_damage():
    frame = smooth_frame()
    nbytes, received = jpeg_roundtrip(frame, quality=20)

    assert 0 < nbytes < frame.numel()  # compressed, and to something plausible
    assert received.shape == frame.shape
    assert not torch.equal(received, frame)  # lossy: the receiver gets less
    assert (received - frame).abs().mean() < 0.1  # but still the same picture


def test_a_lower_quality_ships_fewer_bytes_and_arrives_worse():
    frame = smooth_frame()
    big, good = jpeg_roundtrip(frame, quality=90)
    small, poor = jpeg_roundtrip(frame, quality=5)

    assert small < big
    assert (poor - frame).abs().mean() > (good - frame).abs().mean()


class ConstantModel(torch.nn.Module):
    """Returns a fixed prediction, tagged so routing is observable."""

    def __init__(self, confidence: float, tag: float) -> None:
        super().__init__()
        self.confidence = confidence
        self.tag = tag
        self.seen: list[torch.Tensor] = []

    def forward(self, x):
        self.seen.append(x.clone())
        out = torch.zeros(x.shape[0], 6, 4)
        out[:, :4, :] = self.tag  # boxes
        out[:, 4, :] = self.confidence
        return out


def cascade_over(edge_conf, threshold, batch=2):
    """Build a cascade over two stub models and run one batch through it."""
    edge, cloud = ConstantModel(edge_conf, tag=1.0), ConstantModel(0.99, tag=9.0)
    cascade = Cascade(
        edge=edge,
        cloud=cloud,
        policy=AdaptivePolicy(conf_high=threshold, conf_low=0.0, drift=never_drifts()),
        imgsz=64,
        conf=0.01,
        frame_confidence=mean_confidence,
    )
    frames = torch.stack([smooth_frame() for _ in range(batch)])
    cascade(frames)
    return cascade, edge, cloud, frames


def test_a_confident_frame_never_reaches_the_cloud():
    cascade, _, cloud, _ = cascade_over(edge_conf=0.95, threshold=0.5)

    assert set(cascade.stats.modes) == {Mode.DETECTIONS}
    assert cloud.seen == []  # the point of the whole design
    assert cascade.stats.mean_bytes < 100  # detections, not a frame


def test_an_unconfident_frame_reaches_the_cloud_as_the_wire_delivered_it():
    """The escalated frame must arrive degraded — it was charged for a JPEG."""
    cascade, _, cloud, frames = cascade_over(edge_conf=0.1, threshold=0.9)

    assert set(cascade.stats.modes) == {Mode.FEATURES}
    assert len(cloud.seen) == 1
    assert not torch.equal(cloud.seen[0], frames)  # it went through the codec
    assert cascade.stats.mean_bytes > 100  # a frame costs orders more than boxes


def test_escalating_costs_far_more_than_answering_locally():
    confident, *_ = cascade_over(edge_conf=0.95, threshold=0.5)
    escalated, *_ = cascade_over(edge_conf=0.10, threshold=0.9)
    assert escalated.stats.mean_bytes > 50 * confident.stats.mean_bytes


def test_stats_summarise_the_routing():
    cascade, *_ = cascade_over(edge_conf=0.95, threshold=0.5, batch=4)
    assert cascade.stats.frames == 4
    assert cascade.stats.escalated == 0
    assert cascade.stats.escalation_rate == 0.0


def test_result_reports_both_axes_as_shares_of_the_cloud_endpoint():
    result = CascadeResult(
        edge_only_map50_95=0.385,
        cloud_only_map50_95=0.448,
        cascade_map50_95=0.440,
        edge_only_bytes=39,
        cloud_only_bytes=11431,
        cascade_bytes=5427,
        escalation_rate=0.47,
        frames=129,
    )
    assert result.map_kept == pytest.approx(0.982, abs=0.005)
    assert result.bytes_saved == pytest.approx(0.525, abs=0.005)
    assert set(result.to_dict()) >= {"map_kept", "bytes_saved", "escalation_rate"}
