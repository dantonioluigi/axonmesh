from __future__ import annotations

import json

import pytest

from axonmesh.planner import enumerate_cuts
from axonmesh.replanning import (
    BandwidthEstimator,
    ReplanningController,
    simulate_trace,
)


@pytest.fixture(scope="module")
def options(det_model):
    return enumerate_cuts(det_model, imgsz=160)


class TestBandwidthEstimator:
    def test_first_sample_sets_estimate(self):
        est = BandwidthEstimator(alpha=0.3)
        assert est.mbps is None
        # 1_000_000 bytes in 1 s = 8 Mbps.
        assert est.update(1_000_000, 1.0) == pytest.approx(8.0)
        assert est.mbps == pytest.approx(8.0)

    def test_ewma_smooths(self):
        est = BandwidthEstimator(alpha=0.5, initial_mbps=10.0)
        # sample = 2 Mbps -> 0.5*10 + 0.5*2 = 6.
        assert est.update(250_000, 1.0) == pytest.approx(6.0)

    def test_rejects_bad_alpha_and_duration(self):
        with pytest.raises(ValueError):
            BandwidthEstimator(alpha=0)
        with pytest.raises(ValueError):
            BandwidthEstimator(alpha=1.5)
        with pytest.raises(ValueError):
            BandwidthEstimator().update(100, 0)


class TestController:
    def test_initial_plan_fits_budget(self, options):
        ctl = ReplanningController(options, fps=10)
        d = ctl.observe(bandwidth_mbps=1000)
        assert d.switched and d.plan is not None
        assert d.plan.wire_bytes("int8") <= d.budget_bytes
        assert d.reason == "initial plan"

    def test_stable_bandwidth_does_not_switch(self, options):
        ctl = ReplanningController(options, fps=10)
        ctl.observe(500)
        decisions = [ctl.observe(500) for _ in range(5)]
        assert all(not d.switched for d in decisions)
        assert all(d.reason == "stable" for d in decisions)

    def test_degrade_is_immediate(self, options):
        ctl = ReplanningController(options, fps=10, patience=3)
        ctl.observe(2000)  # generous: offloads a lot (small edge share)
        big = ctl.current
        d = ctl.observe(1)  # collapse the link
        # Either a lighter plan was chosen immediately, or nothing fits.
        assert d.switched
        assert d.plan is None or d.plan.wire_bytes("int8") <= d.budget_bytes
        assert d.plan is None or d.plan.edge_params_share >= big.edge_params_share

    def test_upgrade_waits_for_patience(self, options):
        # Start tight so the initial plan is a heavy-edge/light-wire cut.
        ctl = ReplanningController(options, fps=10, patience=3, margin=0.1)
        first = ctl.observe(5)
        # Jump to generous bandwidth: a lighter-edge plan becomes affordable.
        d1 = ctl.observe(5000)
        d2 = ctl.observe(5000)
        d3 = ctl.observe(5000)
        # No switch until the candidate has been stable `patience` times.
        assert not d1.switched and "pending (1/3)" in d1.reason
        assert not d2.switched and "pending (2/3)" in d2.reason
        assert d3.switched and "upgraded" in d3.reason
        assert d3.plan.edge_params_share <= first.plan.edge_params_share

    def test_transient_spike_does_not_switch(self, options):
        ctl = ReplanningController(options, fps=10, patience=3)
        ctl.observe(5)
        ctl.observe(5000)  # pending 1/3
        d = ctl.observe(5)  # spike gone -> pending resets, stay put
        assert not d.switched

    def test_hot_edge_offloads_without_waiting(self, options):
        ctl = ReplanningController(options, fps=10, patience=5, load_ceiling=0.85)
        ctl.observe(5, edge_load=0.1)
        d = ctl.observe(5000, edge_load=0.95)  # edge hot + upgrade available
        assert d.switched
        assert "overloaded" in d.reason
        assert d.edge_load == 0.95

    def test_no_plan_when_budget_tiny(self, options):
        ctl = ReplanningController(options, fps=1000)
        d = ctl.observe(0.001)
        assert d.plan is None
        assert "no plan fits" in d.reason


def test_simulate_trace_end_to_end(options):
    ctl = ReplanningController(options, fps=10, patience=2)
    trace = [(5, 0.1), (5, 0.1), (5000, 0.2), (5000, 0.2), (5000, 0.2), (1, 0.9)]
    decisions = simulate_trace(ctl, trace)
    assert len(decisions) == len(trace)
    # Ends on a degrade (bandwidth collapsed on the last sample).
    assert decisions[-1].switched


def test_replan_cli(capsys, tmp_path):
    from axonmesh.cli import main

    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps([[5, 0.1], [5000, 0.2], [5000, 0.2], [5000, 0.2], [1, 0.9]]))
    out_json = tmp_path / "timeline.json"
    code = main(
        [
            "replan",
            "--model",
            "yolo11n.yaml",
            "--imgsz",
            "160",
            "--trace",
            str(trace),
            "--fps",
            "10",
            "--patience",
            "2",
            "--json",
            str(out_json),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "switch(es) over 5 observations" in out
    timeline = json.loads(out_json.read_text())
    assert len(timeline) == 5
    assert timeline[0]["cut"] is not None
