from __future__ import annotations

import pytest
import torch

from splitflow.model import SplitModel
from splitflow.split import SplitRunner
from splitflow.transport import Int8Transport


@pytest.fixture()
def model(det_model):
    return SplitModel(det_model, imgsz=160)


def test_defaults_to_the_backbone_cut(model):
    assert model.cut == 10
    assert model.wire == (4, 6, 10)
    assert model.transport is None
    assert len(model.graph) == 24


def test_run_matches_the_unsplit_model(model, det_model, probe):
    torch.testing.assert_close(model.run(probe)[0], det_model(probe)[0], rtol=0, atol=0)


def test_call_is_run(model, probe):
    torch.testing.assert_close(model(probe)[0], model.run(probe)[0], rtol=0, atol=0)


def test_edge_and_cloud_compose(model, probe):
    wire = model.edge(probe)
    assert set(wire) == set(model.wire)
    out = model.cloud(wire)
    torch.testing.assert_close(out[0], model.run(probe)[0], rtol=0, atol=0)


def test_split_reconfigures_and_chains(model, probe):
    returned = model.split(cut=4, transport=Int8Transport())
    assert returned is model
    assert model.cut == 4
    assert isinstance(model.transport, Int8Transport)
    # Still runs, and now the codec charges for the wire.
    model.run(probe)
    assert model.stats.total_bytes > 0


def test_split_keeps_unspecified_settings(model):
    model.split(transport=Int8Transport())
    model.split(cut=6)  # transport not passed -> preserved
    assert model.cut == 6
    assert isinstance(model.transport, Int8Transport)


def test_cut_options_price_every_cut(model):
    options = model.cut_options()
    assert [o.cut for o in options] == list(range(len(model.graph) - 1))
    assert all(o.wire_elements > 0 for o in options)


def test_plan_picks_and_applies_a_cut(model):
    choice = model.plan(bandwidth_mbps=5000, fps=10)
    assert choice is not None
    assert model.cut == choice.cut


def test_plan_leaves_the_cut_alone_when_nothing_fits(model):
    before = model.cut
    assert model.plan(bandwidth_mbps=0.001, fps=1000) is None
    assert model.cut == before


def test_stats_track_the_wire(model, probe):
    model.split(transport=Int8Transport(compress=True))
    model.run(probe)
    model.run(probe)
    assert model.stats.frames == 2
    assert model.stats.mean_bytes > 0


class TestDeploy:
    def test_pins_the_current_cut(self, model):
        cr = model.deploy(name="detector", image="img:1", model_url="https://s/m.pt")
        assert cr["apiVersion"] == "split.dev/v1alpha1"
        assert cr["kind"] == "SplitInference"
        assert cr["metadata"] == {"name": "detector"}
        assert cr["spec"]["cut"] == {"mode": "fixed", "fixed": 10}
        assert cr["spec"]["cloud"] == {"image": "img:1", "replicas": 1, "imgsz": 160}
        assert "bottleneck" not in cr["spec"]

    def test_auto_mode_hands_the_edge_a_budget(self, model):
        cr = model.deploy("d", "img:1", "https://s/m.pt", auto=(50, 10))
        assert cr["spec"]["cut"] == {
            "mode": "auto",
            "auto": {"bandwidthMbps": 50, "fps": 10},
        }

    def test_optional_fields(self, model):
        cr = model.deploy(
            "d",
            "img:1",
            "https://s/m.pt",
            bottleneck_url="https://s/b.pt",
            replicas=3,
            namespace="edge",
        )
        assert cr["metadata"]["namespace"] == "edge"
        assert cr["spec"]["bottleneck"] == {"url": "https://s/b.pt"}
        assert cr["spec"]["cloud"]["replicas"] == 3


def test_accepts_an_adapter_directly(det_model, probe):
    from splitflow.adapters import UltralyticsAdapter

    model = SplitModel(UltralyticsAdapter(det_model), imgsz=160)
    torch.testing.assert_close(
        model.run(probe)[0], SplitRunner(det_model)(probe)[0], rtol=0, atol=0
    )
