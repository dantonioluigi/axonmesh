"""yolosplit: split computing experiments for YOLO11 detection models."""

from .evaluate import MapComparison, compare_map, split_inference
from .quantize import QuantizedTensor, dequantize, quantize
from .split import SplitRunner, WireStats
from .topology import (
    LayerInfo,
    UnsupportedTopologyError,
    backbone_cut,
    build_graph,
    probe_output_shapes,
    wire_indices,
)
from .transport import Int8Transport, RawTransport

__version__ = "0.1.0"

__all__ = [
    "Int8Transport",
    "LayerInfo",
    "MapComparison",
    "QuantizedTensor",
    "RawTransport",
    "SplitRunner",
    "UnsupportedTopologyError",
    "WireStats",
    "__version__",
    "backbone_cut",
    "build_graph",
    "compare_map",
    "dequantize",
    "probe_output_shapes",
    "quantize",
    "split_inference",
    "wire_indices",
]
