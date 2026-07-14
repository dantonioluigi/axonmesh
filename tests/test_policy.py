from __future__ import annotations

import pytest

from yolosplit.policy import (
    AdaptivePolicy,
    ConfidenceEMADrift,
    Detection,
    Mode,
    deserialize_detections,
    serialize_detections,
)


def det(conf: float, cls_id: int = 1) -> Detection:
    return Detection(cls_id, conf, (0.1, 0.2, 0.6, 0.8))


def no_drift() -> ConfidenceEMADrift:
    return ConfidenceEMADrift(threshold=0.0)  # can never fire


class TestSerialization:
    def test_round_trip(self):
        original = [det(0.9, 2), det(0.31, 0)]
        restored = deserialize_detections(serialize_detections(original))
        assert len(restored) == 2
        for a, b in zip(original, restored, strict=True):
            assert a.cls_id == b.cls_id
            assert a.conf == pytest.approx(b.conf, abs=1e-4)
            assert a.xyxyn == pytest.approx(b.xyxyn, abs=1e-4)

    def test_size_is_11_bytes_per_detection(self):
        assert len(serialize_detections([])) == 2
        assert len(serialize_detections([det(0.5)] * 7)) == 2 + 7 * 11

    def test_coordinates_are_clamped(self):
        wild = Detection(3, 1.0, (-0.5, 0.0, 1.7, 1.0))
        (restored,) = deserialize_detections(serialize_detections([wild]))
        assert restored.xyxyn[0] == 0.0
        assert restored.xyxyn[2] == 1.0


class TestDrift:
    def test_no_drift_on_confident_stream(self):
        drift = ConfidenceEMADrift(alpha=0.5, threshold=0.5, warmup=3)
        assert [drift.update(0.9) for _ in range(10)] == [False] * 10

    def test_drift_fires_after_warmup_on_low_confidence(self):
        drift = ConfidenceEMADrift(alpha=0.5, threshold=0.5, warmup=3)
        results = [drift.update(0.1) for _ in range(5)]
        assert results[:2] == [False, False]  # warmup shield
        assert results[2:] == [True, True, True]

    def test_missing_detections_count_as_zero_confidence(self):
        drift = ConfidenceEMADrift(alpha=1.0, threshold=0.5, warmup=1)
        assert drift.update(None) is True
        assert drift.ema == 0.0

    def test_recovery_clears_drift(self):
        drift = ConfidenceEMADrift(alpha=0.9, threshold=0.5, warmup=1)
        assert drift.update(0.1) is True
        assert drift.update(0.95) is False


class TestPolicy:
    def test_high_confidence_ships_detections(self):
        decision = AdaptivePolicy(drift=no_drift()).decide([det(0.9), det(0.8)])
        assert decision.mode is Mode.DETECTIONS
        assert not decision.retrain

    def test_frame_confidence_is_the_minimum(self):
        decision = AdaptivePolicy(drift=no_drift()).decide([det(0.9), det(0.5)])
        assert decision.mode is Mode.FEATURES
        assert decision.frame_conf == pytest.approx(0.5)

    def test_low_confidence_ships_frame_for_retraining(self):
        decision = AdaptivePolicy(drift=no_drift()).decide([det(0.2)])
        assert decision.mode is Mode.FRAME
        assert decision.retrain

    def test_no_detections_ship_features(self):
        decision = AdaptivePolicy(drift=no_drift()).decide([])
        assert decision.mode is Mode.FEATURES
        assert decision.frame_conf is None

    def test_drift_overrides_confidence(self):
        policy = AdaptivePolicy(drift=ConfidenceEMADrift(alpha=1.0, threshold=0.99, warmup=1))
        decision = policy.decide([det(0.9)])
        assert decision.mode is Mode.FRAME
        assert decision.retrain
        assert decision.reason == "drift detected"

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            AdaptivePolicy(conf_high=0.3, conf_low=0.6)
