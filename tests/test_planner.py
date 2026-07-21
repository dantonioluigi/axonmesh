from __future__ import annotations

import pytest

from splitflow.planner import (
    INT8_TENSOR_OVERHEAD,
    CutOption,
    budget_bytes_per_frame,
    enumerate_cuts,
    plan_cut,
)
from splitflow.topology import backbone_cut, wire_indices


@pytest.fixture(scope="module")
def options(det_model):
    return enumerate_cuts(det_model, imgsz=160)


def test_enumerate_covers_every_cut(det_model, graph, options):
    assert [o.cut for o in options] == list(range(len(graph) - 1))
    for o in options:
        assert o.wire == wire_indices(graph, o.cut)
        assert o.wire_elements > 0


def test_edge_params_share_is_monotonic(options):
    from itertools import pairwise

    shares = [o.edge_params_share for o in options]
    assert all(b >= a for a, b in pairwise(shares))
    assert 0 < shares[0] < shares[-1] < 1


def test_wire_bytes_per_transport():
    option = CutOption(cut=3, wire=(3,), wire_elements=1000, edge_params_share=0.1)
    assert option.wire_bytes("fp32") == 4000
    assert option.wire_bytes("fp16") == 2000
    assert option.wire_bytes("int8") == 1000 + INT8_TENSOR_OVERHEAD
    with pytest.raises(ValueError):
        option.wire_bytes("zlib")


def test_budget_math():
    # 8 Mbps at 10 fps -> 1 Mbit/frame -> 100 KB/frame.
    assert budget_bytes_per_frame(8, 10) == 100_000
    with pytest.raises(ValueError):
        budget_bytes_per_frame(0, 10)
    with pytest.raises(ValueError):
        budget_bytes_per_frame(8, -1)


def test_plan_picks_least_edge_compute_that_fits(options):
    generous = max(o.wire_bytes("int8") for o in options)
    choice = plan_cut(options, generous)
    assert choice is not None
    # With everything feasible, the earliest (lightest) cut wins.
    assert choice.edge_params_share == min(o.edge_params_share for o in options)


def test_plan_respects_budget(options):
    sizes = sorted(o.wire_bytes("int8") for o in options)
    tight = sizes[len(sizes) // 2]  # only about half the cuts fit
    choice = plan_cut(options, tight)
    assert choice is not None
    assert choice.wire_bytes("int8") <= tight


def test_plan_returns_none_when_nothing_fits(options):
    assert plan_cut(options, budget=10) is None


def test_backbone_cut_is_among_options(det_model, graph, options):
    cut = backbone_cut(graph)
    assert options[cut].wire == (4, 6, 10)


def test_plan_cli(capsys, tmp_path):
    from splitflow.cli import main

    report = tmp_path / "plan.json"
    code = main(
        [
            "plan",
            "--model",
            "yolo11n.yaml",
            "--imgsz",
            "160",
            "--bandwidth-mbps",
            "100",
            "--fps",
            "10",
            "--json",
            str(report),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "budget:" in out
    assert "<- chosen" in out
    assert report.exists()


def test_plan_cli_fails_when_budget_too_small(capsys):
    from splitflow.cli import main

    code = main(
        [
            "plan",
            "--model",
            "yolo11n.yaml",
            "--imgsz",
            "160",
            "--bandwidth-mbps",
            "0.01",
            "--fps",
            "30",
        ]
    )
    assert code == 1
    assert "no cut fits" in capsys.readouterr().out
