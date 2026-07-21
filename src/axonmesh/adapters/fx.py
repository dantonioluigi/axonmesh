"""A generic adapter: any traceable ``nn.Module``, via ``torch.fx``.

The ultralytics adapter reads a framework-specific attribute (``m.f``/``m.i``)
to learn the graph. Most models do not publish one — but every traceable module
*has* a graph, and ``torch.fx`` can recover it. This adapter symbolically traces
the model, turns the resulting node DAG into the same :class:`LayerInfo` list
the rest of axonmesh consumes, and interprets a span of nodes on demand.

That makes it the fallback backend: registered last, so a purpose-built adapter
still wins, but anything else traceable (torchvision detectors, ``timm``
backbones, a hand-written module) gets the split machinery for free.

Two things a generic backend cannot borrow from YOLO:

- **Where to cut.** There is no "backbone" to end. The default is the *thinnest
  wire* among reasonably balanced cuts — the natural bottleneck of the graph —
  and :mod:`axonmesh.planner` remains the tool for choosing deliberately.
- **What the output means.** The wire carries opaque result bytes, so the task
  head stays the caller's business (see the server's ``postprocess``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node, symbolic_trace
from torch.fx.node import map_arg

from ..topology import MODEL_INPUT, LayerInfo, wire_indices
from .base import ModelAdapter, cache_indices, register_adapter

#: Keep the default cut away from the degenerate ends of the graph.
_BALANCE_BAND = (0.15, 0.85)


class TraceError(TypeError):
    """The model could not be symbolically traced."""


def _node_name(node: Node, gm: GraphModule) -> str:
    """A short, readable label for a node."""
    if node.op == "call_module":
        return type(gm.get_submodule(str(node.target))).__name__
    if node.op in ("call_function", "call_method"):
        target = node.target
        return getattr(target, "__name__", str(target)).rsplit(".", 1)[-1]
    return node.op


class FxAdapter:
    """Splits any symbolically traceable module."""

    name = "torch.fx"

    def __init__(self, model: nn.Module, example_input: torch.Tensor | None = None) -> None:
        if isinstance(model, GraphModule):
            self.gm = model
        else:
            try:
                self.gm = symbolic_trace(model)
            except Exception as err:
                raise TraceError(
                    f"torch.fx cannot trace {type(model).__name__}: {err}. "
                    "Models with data-dependent control flow need a purpose-built adapter."
                ) from err

        self._module = model if isinstance(model, nn.Module) else self.gm
        self._example = example_input

        # Placeholders are the model input; everything else is a layer.
        self._layers: list[Node] = []
        self._index: dict[Node, int] = {}
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                self._index[node] = MODEL_INPUT
            else:
                self._index[node] = len(self._layers)
                self._layers.append(node)

        self._graph = [self._layer_info(i, node) for i, node in enumerate(self._layers)]
        self._cache_set = cache_indices(self._graph)

    # -- ModelAdapter ----------------------------------------------------

    @property
    def module(self) -> nn.Module:
        return self._module

    def graph(self) -> list[LayerInfo]:
        return self._graph

    def probe_shapes(self, imgsz: int = 640) -> list[tuple[int, ...] | None]:
        """Run once and record every layer's output shape."""
        x = self._example if self._example is not None else torch.zeros(1, 3, imgsz, imgsz)
        device = next(self.gm.parameters(), torch.zeros(1)).device
        shapes: list[tuple[int, ...] | None] = []
        env: dict[int, Any] = {}
        with torch.no_grad():
            for index, node in enumerate(self._layers):
                out = self._exec(node, x.to(device), env, {})
                env[index] = out
                shapes.append(tuple(out.shape) if isinstance(out, torch.Tensor) else None)
        return shapes

    def default_cut(self) -> int:
        """The thinnest wire among reasonably balanced cuts.

        A generic graph has no backbone/neck boundary, so pick the natural
        bottleneck instead: the cut whose wire set is smallest, ignoring cuts
        that would leave almost all the compute on one side.
        """
        shapes = self.probe_shapes()
        total = sum(layer.params for layer in self._graph) or 1
        best_cut, best_elements = None, math.inf
        running = 0
        low, high = _BALANCE_BAND
        for cut in range(len(self._graph) - 1):
            running += self._graph[cut].params
            share = running / total
            try:
                wire = wire_indices(self._graph, cut)
            except Exception:  # a cut that would need the raw input is not a cut
                continue
            if not low <= share <= high:
                continue
            elements = sum(math.prod(shapes[i][1:]) for i in wire if shapes[i] is not None)
            if elements and elements < best_elements:
                best_cut, best_elements = cut, elements
        if best_cut is None:  # tiny graph: fall back to the middle
            return max(0, len(self._graph) // 2 - 1)
        return best_cut

    def run_span(self, start: int, stop: int, x: Any, cache: dict[int, torch.Tensor]) -> Any:
        env: dict[int, Any] = {}
        out: Any = x
        for index in range(start, stop + 1):
            node = self._layers[index]
            out = self._exec(node, x, env, cache)
            env[index] = out
            if index in self._cache_set:
                cache[index] = out
        return out

    # -- internals -------------------------------------------------------

    def _layer_info(self, index: int, node: Node) -> LayerInfo:
        sources = tuple(self._index[n] for n in node.all_input_nodes)
        params = 0
        if node.op == "call_module":
            params = sum(p.numel() for p in self.gm.get_submodule(str(node.target)).parameters())
        # No fallback to MODEL_INPUT: a `get_attr` node reads a parameter and
        # genuinely depends on nothing, so claiming it consumes the input would
        # forbid every cut placed before it.
        return LayerInfo(
            index=index,
            name=_node_name(node, self.gm),
            sources=sources,
            params=params,
        )

    def _exec(self, node: Node, x: Any, env: dict[int, Any], cache: dict[int, torch.Tensor]) -> Any:
        def value_of(n: Node) -> Any:
            index = self._index[n]
            if index == MODEL_INPUT:
                return x
            if index in env:
                return env[index]
            if index in cache:
                return cache[index]
            raise KeyError(f"layer {index} was not computed and is not on the wire")

        args = map_arg(node.args, value_of)
        kwargs = map_arg(node.kwargs, value_of)

        if node.op == "call_module":
            return self.gm.get_submodule(str(node.target))(*args, **kwargs)
        if node.op == "call_function":
            return node.target(*args, **kwargs)
        if node.op == "call_method":
            receiver, *rest = args
            return getattr(receiver, str(node.target))(*rest, **kwargs)
        if node.op == "get_attr":
            # Targets are dotted paths ("encoder.pos_embedding"), not plain names.
            attr: Any = self.gm
            for atom in str(node.target).split("."):
                attr = getattr(attr, atom)
            return attr
        if node.op == "output":
            return args[0]
        raise TraceError(f"unsupported fx op {node.op!r}")  # pragma: no cover


def _is_traceable_candidate(model: Any) -> bool:
    """Claim any module as a last resort; tracing decides if it really works."""
    return isinstance(model, nn.Module)


# Registered last: a purpose-built adapter always wins over the generic one.
register_adapter("torch.fx", _is_traceable_candidate, FxAdapter, fallback=True)

_: type[ModelAdapter] = FxAdapter
