from __future__ import annotations

import pytest
import torch

from axonmesh.adapters import (
    ModelAdapter,
    UltralyticsAdapter,
    UnsupportedModelError,
    adapter_for,
    cache_indices,
    register_adapter,
    registered_adapters,
)
from axonmesh.split import SplitRunner
from axonmesh.topology import LayerInfo


def test_ultralytics_is_registered():
    assert "ultralytics" in registered_adapters()


def test_adapter_for_detects_ultralytics(det_model):
    adapter = adapter_for(det_model)
    assert isinstance(adapter, UltralyticsAdapter)
    assert isinstance(adapter, ModelAdapter)
    assert adapter.module is det_model


def test_adapter_passthrough(det_model):
    adapter = UltralyticsAdapter(det_model)
    assert adapter_for(adapter) is adapter


def test_unknown_model_is_rejected():
    # The torch.fx fallback claims any nn.Module, so only a non-module is unsupported.
    with pytest.raises(UnsupportedModelError, match="no adapter"):
        adapter_for("not a model")


def test_adapter_answers_the_four_questions(det_model):
    adapter = adapter_for(det_model)
    graph = adapter.graph()
    assert len(graph) == len(det_model.model)
    assert adapter.default_cut() == 10  # backbone end for YOLO11
    shapes = adapter.probe_shapes(imgsz=160)
    assert len(shapes) == len(graph)
    # run_span replays a prefix and fills the cache it is given.
    cache: dict[int, torch.Tensor] = {}
    out = adapter.run_span(0, 4, torch.zeros(1, 3, 160, 160), cache)
    assert isinstance(out, torch.Tensor)
    assert set(cache) <= set(range(5))


def test_cache_indices_are_the_non_adjacent_sources(graph):
    needed = cache_indices(graph)
    # Every cached index really is consumed by a non-adjacent layer.
    for i in needed:
        assert any(i in layer.sources and layer.index - 1 != i for layer in graph)
    # YOLO11's neck reaches back to the P3/P4/P5 taps.
    assert {4, 6} <= needed


def test_cache_indices_ignores_the_previous_layer():
    chain = [
        LayerInfo(0, "Conv", (-1,), 0),
        LayerInfo(1, "Conv", (0,), 0),
        LayerInfo(2, "Conv", (1,), 0),
    ]
    assert cache_indices(chain) == set()


def test_split_runner_accepts_an_adapter(det_model, probe):
    from_module = SplitRunner(det_model)(probe)[0]
    from_adapter = SplitRunner(UltralyticsAdapter(det_model))(probe)[0]
    torch.testing.assert_close(from_module, from_adapter, rtol=0, atol=0)


def test_registering_a_custom_backend(det_model):
    """A new model family is a registration, not a fork.

    It must also beat the generic torch.fx fallback, which claims every module.
    """

    class Marker(torch.nn.Module):
        pass

    class MarkerAdapter(UltralyticsAdapter):
        name = "marker"

    marker = Marker()
    register_adapter("marker", lambda m: isinstance(m, Marker), lambda m: MarkerAdapter(det_model))
    try:
        assert isinstance(adapter_for(marker), MarkerAdapter)
        assert "marker" in registered_adapters()
        # Registered after the fallback, but resolved before it.
        names = registered_adapters()
        assert names.index("marker") < names.index("torch.fx")
    finally:
        from axonmesh.adapters import base

        base._REGISTRY[:] = [e for e in base._REGISTRY if e[0] != "marker"]


def test_fallbacks_always_sort_last():
    from axonmesh.adapters import base

    register_adapter("late-specific", lambda m: False, lambda m: m)
    try:
        names = registered_adapters()
        assert names[-1] == "torch.fx"
        assert names.index("late-specific") < names.index("torch.fx")
    finally:
        base._REGISTRY[:] = [e for e in base._REGISTRY if e[0] != "late-specific"]


def test_a_failing_detector_does_not_break_resolution(det_model):
    def explodes(_model):
        raise RuntimeError("bad probe")

    register_adapter("explodes", explodes, lambda m: m)
    try:
        # Resolution must skip the broken probe and still find ultralytics.
        assert isinstance(adapter_for(det_model), UltralyticsAdapter)
    finally:
        from axonmesh.adapters import base

        base._REGISTRY[:] = [e for e in base._REGISTRY if e[0] != "explodes"]
