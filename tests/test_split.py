from __future__ import annotations

import pytest
import torch

from splitflow.split import SplitRunner, raw_nbytes
from splitflow.topology import backbone_cut, wire_indices
from splitflow.transport import Int8Transport


@pytest.mark.parametrize("cut", [0, 2, 4, 8, 10, 13, 16, 21])
def test_split_output_matches_full_model(det_model, probe, cut):
    """Edge(0..cut) + cloud(cut+1..) must reproduce the unsplit forward exactly."""
    expected = det_model(probe)[0]
    result = SplitRunner(det_model, cut=cut)(probe)[0]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_default_cut_is_backbone_end(det_model, graph):
    runner = SplitRunner(det_model)
    assert runner.cut == backbone_cut(graph)
    assert runner.wire == wire_indices(graph, runner.cut)


def test_edge_ships_exactly_the_wire_set(det_model, graph, probe):
    runner = SplitRunner(det_model)
    wire = runner.edge(probe)
    assert set(wire) == set(runner.wire)
    assert all(isinstance(t, torch.Tensor) for t in wire.values())


def test_edge_tensors_have_backbone_shapes(det_model, probe):
    """At the backbone cut the wire is the P3/P4/P5 pyramid (strides 8/16/32)."""
    wire = SplitRunner(det_model).edge(probe)
    strides = {i: probe.shape[-1] // wire[i].shape[-1] for i in wire}
    assert sorted(strides.values()) == [8, 16, 32]


def test_stats_record_raw_bytes_without_transport(det_model, probe):
    runner = SplitRunner(det_model)
    runner(probe)
    runner(probe)
    assert runner.stats.frames == 2
    expected = raw_nbytes(runner.edge(probe))
    assert runner.stats.bytes_per_frame == [expected, expected]
    assert runner.stats.mean_bytes == expected


def test_int8_transport_bytes_and_output_shape(det_model, probe):
    baseline = SplitRunner(det_model)
    baseline_out = baseline(probe)[0]
    raw = baseline.stats.total_bytes

    runner = SplitRunner(det_model, transport=Int8Transport())
    out = runner(probe)[0]

    assert out.shape == baseline_out.shape
    assert torch.isfinite(out).all()
    # INT8 payload must be < 30% of fp32 (header overhead stays small).
    assert runner.stats.total_bytes < 0.3 * raw


def test_wire_stats_empty_mean(det_model):
    assert SplitRunner(det_model).stats.mean_bytes == 0.0
