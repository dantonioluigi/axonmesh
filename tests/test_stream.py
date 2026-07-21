from __future__ import annotations

import numpy as np
import pytest

from axonmesh.measure import to_input_tensor
from axonmesh.policy import AdaptivePolicy, ConfidenceEMADrift, Detection, Mode
from axonmesh.split import SplitRunner
from axonmesh.stream import (
    iter_image_frames,
    simulate_stream,
    summarize_stream,
    transport_feature_bytes,
)
from axonmesh.transport import Int8Transport


def det(conf: float) -> Detection:
    return Detection(0, conf, (0.1, 0.1, 0.5, 0.5))


@pytest.fixture()
def frames(bgr_image):
    return [(f"f{i}.jpg", bgr_image) for i in range(4)]


def scripted_inferer(script):
    """Return per-frame detections from a list, in order."""
    frames_seen = iter(script)

    def infer(_image):
        return next(frames_seen)

    return infer


def test_stream_prices_each_mode(frames):
    script = [[det(0.9)], [det(0.5)], [det(0.1)], [det(0.9)]]
    policy = AdaptivePolicy(drift=ConfidenceEMADrift(threshold=0.0))
    reports = simulate_stream(
        frames, scripted_inferer(script), policy, feature_bytes=lambda _img: 5000
    )
    assert [r.mode for r in reports] == [
        Mode.DETECTIONS,
        Mode.FEATURES,
        Mode.FRAME,
        Mode.DETECTIONS,
    ]
    assert reports[0].nbytes == 2 + 11  # one serialised detection
    assert reports[1].nbytes == 5000
    assert reports[2].nbytes == reports[2].jpeg_bytes
    assert reports[2].retrain and not reports[0].retrain


def test_summary_accounts_bytes_and_modes(frames):
    script = [[det(0.9)], [det(0.5)], [det(0.1)], [det(0.9)]]
    policy = AdaptivePolicy(drift=ConfidenceEMADrift(threshold=0.0))
    reports = simulate_stream(
        frames, scripted_inferer(script), policy, feature_bytes=lambda _img: 5000
    )
    summary = summarize_stream(reports)
    assert summary["frames"] == 4
    assert summary["frames_detections"] == 2
    assert summary["frames_features"] == 1
    assert summary["frames_frame"] == 1
    assert summary["retrain_frames"] == 1
    assert summary["total_bytes"] == sum(r.nbytes for r in reports)
    assert summary["baseline_jpeg_bytes"] == sum(r.jpeg_bytes for r in reports)
    assert 0.0 < summary["saved_vs_jpeg"] < 1.0

    total_when_confident = simulate_stream(
        frames,
        scripted_inferer([[det(0.9)]] * 4),
        AdaptivePolicy(drift=ConfidenceEMADrift(threshold=0.0)),
        feature_bytes=lambda _img: 5000,
    )
    assert summarize_stream(total_when_confident)["saved_vs_jpeg"] > summary["saved_vs_jpeg"]


def test_drift_sends_frames_to_retraining(frames):
    policy = AdaptivePolicy(drift=ConfidenceEMADrift(alpha=1.0, threshold=0.95, warmup=1))
    reports = simulate_stream(
        frames,
        scripted_inferer([[det(0.5)]] * 4),
        policy,
        feature_bytes=lambda _img: 5000,
    )
    assert all(r.mode is Mode.FRAME and r.retrain for r in reports)


def test_iter_image_frames(images_dir):
    frames = list(iter_image_frames(images_dir, limit=2))
    assert [name for name, _ in frames] == ["frame_0.jpg", "frame_1.jpg"]
    assert all(isinstance(img, np.ndarray) for _, img in frames)
    with pytest.raises(FileNotFoundError):
        list(iter_image_frames(images_dir / "missing"))


def test_transport_feature_bytes_uses_runner_transport(det_model, bgr_image):
    runner = SplitRunner(det_model, transport=Int8Transport())
    price = transport_feature_bytes(runner, imgsz=160)
    nbytes = price(bgr_image)
    wire = runner.edge(to_input_tensor(bgr_image, 160))
    assert nbytes == runner.transport(wire)[1]


def test_transport_feature_bytes_requires_transport(det_model):
    with pytest.raises(ValueError, match="transport"):
        transport_feature_bytes(SplitRunner(det_model))
