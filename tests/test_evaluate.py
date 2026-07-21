from __future__ import annotations

import pytest
import torch

from axonmesh.evaluate import MapComparison, split_inference
from axonmesh.split import SplitRunner
from axonmesh.transport import Int8Transport


def test_patch_reroutes_forward_through_split(det_model, probe):
    baseline = det_model(probe)[0]
    with split_inference(det_model) as runner:
        patched_out = det_model(probe)[0]
    torch.testing.assert_close(patched_out, baseline, rtol=0, atol=0)
    assert runner.stats.frames == 1


def test_patch_records_transport_bytes(det_model, probe):
    with split_inference(det_model, transport=Int8Transport()) as runner:
        det_model(probe)
        det_model(probe)
    assert runner.stats.frames == 2
    raw = SplitRunner(det_model)
    raw(probe)
    assert runner.stats.mean_bytes < 0.3 * raw.stats.mean_bytes


def test_patch_is_restored_on_exit(det_model, probe):
    assert "_predict_once" not in det_model.__dict__
    with split_inference(det_model):
        assert "_predict_once" in det_model.__dict__
    assert "_predict_once" not in det_model.__dict__
    # And restored even when the body raises.
    try:
        with split_inference(det_model):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "_predict_once" not in det_model.__dict__
    torch.testing.assert_close(det_model(probe)[0], det_model(probe)[0])


def test_nested_patch_restores_previous_patch(det_model, probe):
    with split_inference(det_model) as outer:
        with split_inference(det_model, cut=4) as inner:
            det_model(probe)
        det_model(probe)
    assert inner.stats.frames == 1
    assert outer.stats.frames == 1


def test_map_comparison_deltas():
    comparison = MapComparison(
        cut=10,
        baseline_map50=0.90,
        baseline_map50_95=0.80,
        split_map50=0.88,
        split_map50_95=0.79,
        frames=100,
        wire_mean_bytes=50_000.0,
    )
    assert comparison.delta_map50 == pytest.approx(-0.02)
    assert comparison.delta_map50_95 == pytest.approx(-0.01)
    d = comparison.to_dict()
    assert d["delta_map50"] == pytest.approx(-0.02)
    assert d["cut"] == 10
