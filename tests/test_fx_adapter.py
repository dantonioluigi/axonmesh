"""The generic backend: any traceable module, no ultralytics involved.

If these pass, the abstraction is real — the planner, codecs, wire protocol and
facade work on architectures the project was never written for.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from splitflow.adapters import FxAdapter, TraceError, adapter_for
from splitflow.adapters.fx import _is_traceable_candidate
from splitflow.model import SplitModel
from splitflow.split import SplitRunner
from splitflow.topology import MODEL_INPUT
from splitflow.transport import Int8Transport


class SkipNet(nn.Module):
    """Tiny net with a skip connection — the thing a naive slice gets wrong."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.a = nn.Conv2d(8, 8, 3, padding=1)
        self.b = nn.Conv2d(8, 8, 3, padding=1)
        self.head = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        s = torch.relu(self.stem(x))
        h = torch.relu(self.a(s))
        h = torch.relu(self.b(h))
        return self.head(torch.cat([h, s], dim=1))  # consumes `s` across the graph


@pytest.fixture()
def skipnet():
    torch.manual_seed(15)
    return SkipNet().eval()


@pytest.fixture()
def x():
    torch.manual_seed(15)
    return torch.rand(1, 3, 32, 32)


class TestGraphRecovery:
    def test_traces_into_layers(self, skipnet):
        graph = FxAdapter(skipnet).graph()
        names = [layer.name for layer in graph]
        assert "Conv2d" in names and "cat" in names and names[-1] == "output"
        assert graph[0].sources == (MODEL_INPUT,)

    def test_skip_connection_shows_up_as_a_non_adjacent_source(self, skipnet):
        graph = FxAdapter(skipnet).graph()
        cat = next(layer for layer in graph if layer.name == "cat")
        # `cat` consumes two tensors, one of them produced well before it.
        assert len(cat.sources) == 2
        assert any(s < cat.index - 1 for s in cat.sources)

    def test_params_are_attributed_to_module_layers(self, skipnet):
        graph = FxAdapter(skipnet).graph()
        assert sum(layer.params for layer in graph) == sum(p.numel() for p in skipnet.parameters())

    def test_probe_shapes_reports_every_layer(self, skipnet, x):
        adapter = FxAdapter(skipnet, example_input=x)
        shapes = adapter.probe_shapes()
        assert len(shapes) == len(adapter.graph())
        assert shapes[0] == (1, 8, 32, 32)


class TestSplitCorrectness:
    def test_split_output_is_bit_identical(self, skipnet, x):
        """The whole point: cutting must not change the answer."""
        expected = skipnet(x)
        adapter = FxAdapter(skipnet, example_input=x)
        for cut in range(len(adapter.graph()) - 1):
            runner = SplitRunner(adapter, cut=cut)
            torch.testing.assert_close(runner(x), expected, rtol=0, atol=0)

    def test_wire_set_carries_the_skip_tensor(self, skipnet, x):
        adapter = FxAdapter(skipnet, example_input=x)
        graph = adapter.graph()
        cat = next(layer for layer in graph if layer.name == "cat")
        # Cut just before `cat` consumes its far source: both inputs must ship.
        cut = cat.index - 1
        runner = SplitRunner(adapter, cut=cut)
        assert set(cat.sources) <= set(runner.wire)

    def test_runs_through_a_lossy_transport(self, skipnet, x):
        adapter = FxAdapter(skipnet, example_input=x)
        runner = SplitRunner(adapter, cut=2, transport=Int8Transport(compress=True))
        out = runner(x)
        assert out.shape == skipnet(x).shape
        assert runner.stats.total_bytes > 0


class TestDefaultCut:
    def test_is_balanced_and_valid(self, skipnet, x):
        adapter = FxAdapter(skipnet, example_input=x)
        cut = adapter.default_cut()
        assert 0 <= cut < len(adapter.graph()) - 1
        # It must be usable as a split point.
        SplitRunner(adapter, cut=cut)(x)


class TestRegistry:
    def test_generic_adapter_is_the_fallback(self, skipnet):
        """A plain module resolves to fx, not to a purpose-built backend."""
        assert isinstance(adapter_for(skipnet), FxAdapter)

    def test_purpose_built_adapter_still_wins(self, det_model):
        from splitflow.adapters import UltralyticsAdapter

        assert isinstance(adapter_for(det_model), UltralyticsAdapter)

    def test_only_modules_are_claimed(self):
        assert _is_traceable_candidate(nn.Linear(2, 2)) is True
        assert _is_traceable_candidate("not a model") is False

    def test_untraceable_model_fails_clearly(self):
        class DataDependent(nn.Module):
            def forward(self, x):
                if x.sum() > 0:  # control flow fx cannot trace
                    return x * 2
                return x

        with pytest.raises(TraceError, match="cannot trace"):
            FxAdapter(DataDependent())


class TestFacadeOnAGenericModel:
    def test_split_plan_run_deploy(self, skipnet, x):
        model = SplitModel(skipnet, imgsz=32)
        assert isinstance(model.adapter, FxAdapter)

        torch.testing.assert_close(model.run(x), skipnet(x), rtol=0, atol=0)

        options = model.cut_options(imgsz=32)
        assert len(options) == len(model.graph) - 1

        model.split(cut=2, transport=Int8Transport())
        model.run(x)
        assert model.stats.total_bytes > 0

        cr = model.deploy(name="generic", image="img:1", model_url="https://s/m.pt")
        assert cr["spec"]["cut"] == {"mode": "fixed", "fixed": 2}


@pytest.mark.parametrize("depth", [18])
def test_a_real_torchvision_backbone(depth):
    """ResNet — an architecture this project was never written for."""
    torchvision = pytest.importorskip("torchvision")

    model = getattr(torchvision.models, f"resnet{depth}")(weights=None).eval()
    x = torch.rand(1, 3, 64, 64)
    adapter = FxAdapter(model, example_input=x)

    graph = adapter.graph()
    assert len(graph) > 40  # a real graph, not a toy
    assert any(layer.name == "add" for layer in graph)  # residual connections

    cut = adapter.default_cut()
    runner = SplitRunner(adapter, cut=cut, transport=Int8Transport(compress=True))
    out = runner(x)
    assert out.shape == (1, 1000)
    assert runner.stats.total_bytes > 0

    # And the split is exact without a lossy codec.
    torch.testing.assert_close(SplitRunner(adapter, cut=cut)(x), model(x), rtol=0, atol=0)


class TestAwkwardGraphs:
    """Things a second architecture surfaces that the first one never did."""

    def test_dotted_get_attr_is_resolved(self):
        """Parameters reached through a submodule ("encoder.pos_embedding")."""

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.pos = nn.Parameter(torch.ones(1, 4))

        class WithEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(x) + self.inner.pos

        net = WithEmbedding().eval()
        x = torch.rand(2, 4)
        adapter = FxAdapter(net, example_input=x)
        assert any(n.op == "get_attr" and "." in str(n.target) for n in adapter.gm.graph.nodes)
        torch.testing.assert_close(SplitRunner(adapter, cut=0)(x), net(x), rtol=0, atol=0)

    def test_accepts_a_pre_traced_graphmodule(self):
        """The escape hatch for models plain symbolic_trace refuses."""

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return torch.relu(self.fc(x))

        net = Net().eval()
        gm = torch.fx.symbolic_trace(net)
        x = torch.rand(2, 4)
        adapter = FxAdapter(gm, example_input=x)
        torch.testing.assert_close(SplitRunner(adapter, cut=0)(x), net(x), rtol=0, atol=0)


def test_a_vision_transformer():
    """ViT — a transformer, traced through torchvision's leaf-aware tracer.

    torchvision's ViT asserts on the input size, which plain symbolic_trace
    cannot evaluate; pre-tracing is the documented escape hatch.
    """
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models.feature_extraction import NodePathTracer

    net = torchvision.models.vit_b_16(weights=None).eval()
    gm = torch.fx.GraphModule(net, NodePathTracer().trace(net))
    x = torch.rand(1, 3, 224, 224)
    adapter = FxAdapter(gm, example_input=x)

    graph = adapter.graph()
    assert len(graph) > 200  # a transformer's worth of nodes
    cut = adapter.default_cut()
    torch.testing.assert_close(SplitRunner(adapter, cut=cut)(x), net(x), rtol=0, atol=0)
