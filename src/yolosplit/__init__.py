"""yolosplit: split computing experiments for YOLO11 detection models."""

from .bottleneck import (
    Bottleneck,
    BottleneckTransport,
    LevelCodec,
    load_bottleneck,
    save_bottleneck,
)
from .evaluate import MapComparison, compare_map, split_inference
from .planner import CutOption, budget_bytes_per_frame, enumerate_cuts, plan_cut
from .policy import (
    AdaptivePolicy,
    ConfidenceEMADrift,
    Decision,
    Detection,
    Mode,
    deserialize_detections,
    serialize_detections,
)
from .quantize import QuantizedTensor, dequantize, quantize
from .split import SplitRunner, WireStats
from .stream import FrameReport, simulate_stream, summarize_stream
from .topology import (
    LayerInfo,
    UnsupportedTopologyError,
    backbone_cut,
    build_graph,
    probe_output_shapes,
    wire_indices,
)
from .train import TrainResult, train_bottleneck
from .transport import Int8Transport, RawTransport

__version__ = "0.3.0"

__all__ = [
    "AdaptivePolicy",
    "Bottleneck",
    "BottleneckTransport",
    "ConfidenceEMADrift",
    "CutOption",
    "Decision",
    "Detection",
    "FrameReport",
    "Int8Transport",
    "LayerInfo",
    "LevelCodec",
    "MapComparison",
    "Mode",
    "QuantizedTensor",
    "RawTransport",
    "SplitRunner",
    "TrainResult",
    "UnsupportedTopologyError",
    "WireStats",
    "__version__",
    "backbone_cut",
    "budget_bytes_per_frame",
    "build_graph",
    "compare_map",
    "dequantize",
    "deserialize_detections",
    "enumerate_cuts",
    "load_bottleneck",
    "plan_cut",
    "probe_output_shapes",
    "quantize",
    "save_bottleneck",
    "serialize_detections",
    "simulate_stream",
    "split_inference",
    "summarize_stream",
    "train_bottleneck",
    "wire_indices",
]
