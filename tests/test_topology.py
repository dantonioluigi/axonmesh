from __future__ import annotations

import pytest

from yolosplit.topology import (
    MODEL_INPUT,
    LayerInfo,
    UnsupportedTopologyError,
    backbone_cut,
    build_graph,
    probe_output_shapes,
    wire_indices,
)


def test_graph_covers_every_layer(det_model, graph):
    assert len(graph) == len(det_model.model)
    assert [layer.index for layer in graph] == list(range(len(graph)))


def test_sources_are_resolved_and_valid(graph):
    for layer in graph:
        for s in layer.sources:
            assert MODEL_INPUT <= s < layer.index
    # Only the stem consumes the raw input.
    assert graph[0].sources == (MODEL_INPUT,)
    assert all(MODEL_INPUT not in layer.sources for layer in graph[1:])


def test_skip_connections_are_present(graph):
    """YOLO11 concatenates backbone P3/P4/P5 into the neck — the reason slicing fails."""
    non_sequential = [layer for layer in graph if not layer.is_sequential and layer.index > 0]
    assert non_sequential, "expected Concat/Detect layers with non-adjacent sources"
    assert any(len(layer.sources) > 1 for layer in non_sequential)


def test_backbone_cut_is_before_first_upsample(graph):
    cut = backbone_cut(graph)
    assert "Upsample" in graph[cut + 1].name
    assert cut == 10  # YOLO11: backbone is layers 0..10 (C2PSA is the last)


def test_backbone_cut_requires_an_upsample():
    fake = [LayerInfo(0, "Conv", (MODEL_INPUT,), 0), LayerInfo(1, "Conv", (0,), 0)]
    with pytest.raises(UnsupportedTopologyError):
        backbone_cut(fake)


def test_wire_at_backbone_cut_is_p3_p4_p5(graph):
    assert wire_indices(graph, backbone_cut(graph)) == (4, 6, 10)


def test_wire_is_complete_for_every_cut(graph):
    """No layer after any cut may reference a pre-cut tensor missing from the wire."""
    for cut in range(len(graph) - 1):
        wire = set(wire_indices(graph, cut))
        for layer in graph[cut + 1 :]:
            for s in layer.sources:
                if s <= cut:
                    assert s in wire, f"cut={cut}: layer {layer.index} needs {s}"


def test_wire_rejects_out_of_range_cuts(graph):
    with pytest.raises(ValueError):
        wire_indices(graph, -1)
    with pytest.raises(ValueError):
        wire_indices(graph, len(graph) - 1)


def test_wire_rejects_model_input_after_cut():
    fake = [
        LayerInfo(0, "Conv", (MODEL_INPUT,), 0),
        LayerInfo(1, "Conv", (MODEL_INPUT,), 0),
    ]
    with pytest.raises(UnsupportedTopologyError):
        wire_indices(fake, 0)


def test_build_graph_rejects_forward_references(det_model):
    layer = det_model.model[3]
    original = layer.f
    layer.f = 7  # forward reference: layer 3 cannot consume layer 7
    try:
        with pytest.raises(UnsupportedTopologyError):
            build_graph(det_model)
    finally:
        layer.f = original


def test_build_graph_without_np_attribute(det_model):
    """Checkpoints restored from disk may lack ``m.np``; params come from the weights."""
    for m in det_model.model:
        if hasattr(m, "np"):
            del m.np
    graph = build_graph(det_model)
    assert len(graph) == len(det_model.model)
    for layer, m in zip(graph, det_model.model, strict=True):
        assert layer.params == sum(p.numel() for p in m.parameters())


def test_probe_shapes_match_strides(det_model, graph):
    shapes = probe_output_shapes(det_model, imgsz=160)
    assert len(shapes) == len(graph)
    assert shapes[-1] is None  # Detect returns a tuple in eval mode
    # P3/P4/P5 taps at strides 8/16/32.
    assert shapes[4][2:] == (20, 20)
    assert shapes[6][2:] == (10, 10)
    assert shapes[10][2:] == (5, 5)
    assert all(s[0] == 1 for s in shapes[:-1])
