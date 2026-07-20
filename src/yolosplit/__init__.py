"""yolosplit: split computing for detection models, from probe to K8s service."""

from .bottleneck import (
    Bottleneck,
    BottleneckTransport,
    LevelCodec,
    load_bottleneck,
    save_bottleneck,
)
from .edge import EdgeClient, run_edge
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
from .protocol import (
    Handshake,
    Kind,
    ProtocolError,
    module_fingerprint,
    pack_tensors,
    unpack_tensors,
)
from .quantize import QuantizedTensor, dequantize, quantize
from .server import CloudServer, Metrics, start_metrics_server
from .split import SplitRunner, WireStats
from .stream import FrameReport, simulate_stream, summarize_stream
from .sweep import SweepConfig, SweepResult, run_sweep
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

__version__ = "0.5.0"

__all__ = [
    "AdaptivePolicy",
    "Bottleneck",
    "BottleneckTransport",
    "CloudServer",
    "ConfidenceEMADrift",
    "CutOption",
    "Decision",
    "Detection",
    "EdgeClient",
    "FrameReport",
    "Handshake",
    "Int8Transport",
    "Kind",
    "LayerInfo",
    "LevelCodec",
    "MapComparison",
    "Metrics",
    "Mode",
    "ProtocolError",
    "QuantizedTensor",
    "RawTransport",
    "SplitRunner",
    "SweepConfig",
    "SweepResult",
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
    "module_fingerprint",
    "pack_tensors",
    "plan_cut",
    "probe_output_shapes",
    "quantize",
    "run_edge",
    "run_sweep",
    "save_bottleneck",
    "serialize_detections",
    "simulate_stream",
    "split_inference",
    "start_metrics_server",
    "summarize_stream",
    "train_bottleneck",
    "unpack_tensors",
    "wire_indices",
]
