from __future__ import annotations

import pytest

from axonmesh.calibrate import (
    FrameProbe,
    ThresholdPoint,
    agreement_bytes_pareto,
    choose_threshold,
    detection_agreement,
    iou,
    sweep_thresholds,
    to_markdown,
)
from axonmesh.policy import Detection


def box(cls_id=0, conf=0.9, xyxyn=(0.1, 0.1, 0.3, 0.3)) -> Detection:
    return Detection(cls_id, conf, xyxyn)


def probe(confidence, agreement, detection_bytes=30, frame_bytes=10_000) -> FrameProbe:
    return FrameProbe(confidence, agreement, detection_bytes, frame_bytes)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0, 0, 1, 1), (0, 0, 1, 1), 1.0),
        ((0, 0, 1, 1), (2, 2, 3, 3), 0.0),  # disjoint
        ((0, 0, 2, 2), (1, 1, 3, 3), 1 / 7),  # 1 overlap, 7 union
        ((0, 0, 0, 0), (0, 0, 0, 0), 0.0),  # degenerate, not a division by zero
    ],
)
def test_iou(a, b, expected):
    assert iou(a, b) == pytest.approx(expected)


def test_identical_detections_agree_completely():
    dets = [box(0), box(1, xyxyn=(0.5, 0.5, 0.7, 0.7))]
    assert detection_agreement(dets, list(dets)) == 1.0


def test_two_empty_frames_agree():
    """Both models seeing nothing is a frame the cascade handles perfectly.

    Scoring it zero would drag the threshold toward escalating precisely the
    frames that need it least.
    """
    assert detection_agreement([], []) == 1.0


def test_one_side_empty_agrees_on_nothing():
    assert detection_agreement([box()], []) == 0.0
    assert detection_agreement([], [box()]) == 0.0


def test_a_box_of_the_wrong_class_is_not_a_match():
    assert detection_agreement([box(cls_id=0)], [box(cls_id=1)]) == 0.0


def test_a_box_in_the_wrong_place_is_not_a_match():
    assert detection_agreement([box()], [box(xyxyn=(0.8, 0.8, 0.95, 0.95))]) == 0.0


def test_partial_overlap_scores_between():
    """One of two edge boxes matches: F1 over 2 and 1 boxes is 2/3."""
    edge = [box(0), box(0, xyxyn=(0.6, 0.6, 0.8, 0.8))]
    assert detection_agreement(edge, [box(0)]) == pytest.approx(2 / 3)


def test_each_cloud_box_can_only_be_claimed_once():
    """Two edge boxes over one cloud box must not both count as matched."""
    edge = [box(0), box(0, conf=0.8)]  # same place, twice
    assert detection_agreement(edge, [box(0)]) == pytest.approx(2 / 3)


def test_sweeping_trades_bytes_for_agreement_monotonically():
    probes = [probe(c, agreement=a) for c, a in [(0.9, 0.95), (0.5, 0.6), (0.2, 0.1)]]
    points = sweep_thresholds(probes, steps=11)

    assert points[0].threshold == 0.0 and points[-1].threshold == 1.0
    assert points[0].escalation_rate == 0.0  # threshold 0: nothing escalates
    assert points[-1].escalation_rate == 1.0  # threshold 1: everything does
    assert points[-1].agreement == 1.0  # always-escalate agrees with itself
    assert points[-1].mean_bytes > points[0].mean_bytes


def test_a_frame_with_no_detections_always_escalates():
    """No confidence is not zero confidence: there is nothing to be sure about."""
    points = sweep_thresholds([probe(None, agreement=0.0)], steps=3)
    assert all(point.escalation_rate == 1.0 for point in points)


def test_sweeping_nothing_is_an_error_not_an_empty_curve():
    with pytest.raises(ValueError, match="needs frames"):
        sweep_thresholds([], steps=5)


def test_a_bandwidth_ceiling_picks_the_most_faithful_that_fits():
    points = [
        ThresholdPoint(0.2, mean_bytes=1000, agreement=0.80, escalation_rate=0.1),
        ThresholdPoint(0.5, mean_bytes=4000, agreement=0.93, escalation_rate=0.4),
        ThresholdPoint(0.8, mean_bytes=9000, agreement=0.99, escalation_rate=0.9),
    ]
    assert choose_threshold(points, max_bytes=5000).threshold == 0.5


def test_an_agreement_floor_picks_the_cheapest_that_clears_it():
    points = [
        ThresholdPoint(0.2, mean_bytes=1000, agreement=0.80, escalation_rate=0.1),
        ThresholdPoint(0.5, mean_bytes=4000, agreement=0.93, escalation_rate=0.4),
        ThresholdPoint(0.8, mean_bytes=9000, agreement=0.99, escalation_rate=0.9),
    ]
    assert choose_threshold(points, min_agreement=0.9).threshold == 0.5


def test_an_impossible_constraint_is_refused_with_what_was_achievable():
    """Returning the closest miss would let a deployment believe a budget was met."""
    points = [ThresholdPoint(0.5, mean_bytes=4000, agreement=0.93, escalation_rate=0.4)]
    with pytest.raises(ValueError, match="no threshold satisfies"):
        choose_threshold(points, max_bytes=100)
    with pytest.raises(ValueError, match=r"0\.930"):
        choose_threshold(points, min_agreement=0.999)


def test_choosing_without_a_constraint_is_meaningless():
    with pytest.raises(ValueError, match="give a constraint"):
        choose_threshold([ThresholdPoint(0.5, 4000, 0.93, 0.4)])


def test_the_pareto_front_drops_dominated_thresholds():
    cheap_and_good = ThresholdPoint(0.3, mean_bytes=1000, agreement=0.9, escalation_rate=0.2)
    dominated = ThresholdPoint(0.4, mean_bytes=2000, agreement=0.8, escalation_rate=0.3)
    expensive_and_best = ThresholdPoint(0.9, mean_bytes=9000, agreement=1.0, escalation_rate=0.9)

    front = agreement_bytes_pareto([cheap_and_good, dominated, expensive_and_best])
    assert dominated not in front
    assert {cheap_and_good, expensive_and_best} <= set(front)


def test_markdown_has_a_row_per_threshold():
    points = sweep_thresholds([probe(0.7, 0.9)], steps=5)
    table = to_markdown(points)
    assert len(table.splitlines()) == 2 + len(points)
    assert table.startswith("| threshold |")
