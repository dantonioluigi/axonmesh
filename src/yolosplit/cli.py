"""Command-line interface: ``yolosplit inspect|measure|evaluate|train-bottleneck|stream``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import __version__


def _load_model(model: str):
    """Load a DetectionModel from a ``.pt`` checkpoint or build one from a YAML."""
    if model.endswith((".yaml", ".yml")):
        from ultralytics.nn.tasks import DetectionModel

        det_model = DetectionModel(cfg=model, verbose=False)
    else:
        from ultralytics import YOLO

        det_model = YOLO(model).model
    return det_model.float().eval()


def _load_yolo(model: str):
    """Load the full ultralytics ``YOLO`` wrapper (needed for predict/val)."""
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.model.float().eval()
    return yolo


_CUT_HELP = "cut layer (default: backbone)"


def _kb(nbytes: float) -> str:
    return f"{nbytes / 1024:.1f}"


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .topology import backbone_cut, build_graph, probe_output_shapes, wire_indices

    det_model = _load_model(args.model)
    graph = build_graph(det_model)
    shapes = probe_output_shapes(det_model, imgsz=args.imgsz)
    bb_cut = backbone_cut(graph)

    print(f"# Layers ({args.model}, input 1x3x{args.imgsz}x{args.imgsz})")
    print(f"{'idx':>4} {'module':<16} {'sources':<14} {'params':>10}  output")
    for info, shape in zip(graph, shapes, strict=True):
        out = "x".join(map(str, shape)) if shape else "-"
        srcs = ",".join(map(str, info.sources))
        print(f"{info.index:>4} {info.name:<16} {srcs:<14} {info.params:>10}  {out}")

    print(f"\n# Cut analysis (backbone ends at layer {bb_cut}, marked *)")
    print(f"{'cut':>4} {'wire tensors':<20} {'fp32 KB':>9} {'fp16 KB':>9} {'int8 KB':>9}")
    for cut in range(len(graph) - 1):
        wire = wire_indices(graph, cut)
        elems = sum(math.prod(shapes[i][1:]) for i in wire)
        mark = "*" if cut == bb_cut else " "
        wire_s = ",".join(map(str, wire))
        print(
            f"{cut:>3}{mark} {wire_s:<20} {_kb(elems * 4):>9} {_kb(elems * 2):>9} {_kb(elems):>9}"
        )
    print("\nint8 column is raw element count; run `yolosplit measure` for exact wire bytes.")
    return 0


def _build_transport(args: argparse.Namespace):
    from .transport import Int8Transport, RawTransport

    axis = 1 if getattr(args, "per_channel", False) else None
    if getattr(args, "bottleneck", None):
        from .bottleneck import BottleneckTransport, load_bottleneck

        return BottleneckTransport(
            load_bottleneck(args.bottleneck), axis=axis, compress=args.compress
        )
    if args.transport == "int8":
        return Int8Transport(axis=axis, compress=args.compress)
    import torch

    return RawTransport(dtype=torch.float16 if args.transport == "fp16" else torch.float32)


def _cmd_measure(args: argparse.Namespace) -> int:
    from .measure import measure_directory, summarize, to_markdown
    from .split import SplitRunner

    runner = SplitRunner(_load_model(args.model), cut=args.cut)
    measurements = measure_directory(
        runner, args.images, imgsz=args.imgsz, quality=args.quality, limit=args.limit
    )
    print(f"cut={runner.cut} wire tensors={list(runner.wire)} jpeg quality={args.quality}\n")
    print(to_markdown(measurements))
    if args.json:
        summary = summarize(measurements) | {"cut": runner.cut, "jpeg_quality": args.quality}
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"\nsummary written to {args.json}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:  # pragma: no cover - needs a dataset
    from .evaluate import compare_map

    comparison = compare_map(
        weights=args.model,
        data=args.data,
        cut=args.cut,
        transport=_build_transport(args),
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
    )
    report = comparison.to_dict()
    print(json.dumps(report, indent=2))
    print(
        f"\nΔmAP50={comparison.delta_map50:+.4f} ΔmAP50-95={comparison.delta_map50_95:+.4f} "
        f"mean wire={comparison.wire_mean_bytes / 1024:.1f} KB/frame"
    )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


def _cmd_train_bottleneck(args: argparse.Namespace) -> int:
    from .bottleneck import save_bottleneck
    from .train import train_bottleneck

    bottleneck, result = train_bottleneck(
        _load_model(args.model),
        args.images,
        cut=args.cut,
        latent_channels=args.latent_channels,
        stride=args.stride,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        imgsz=args.imgsz,
        limit=args.limit,
        device=args.device,
        quant_noise=not args.no_quant_noise,
    )
    for epoch, loss in enumerate(result.epoch_losses, 1):
        print(f"epoch {epoch}/{args.epochs}: normalised MSE {loss:.4f}")
    for i, err in sorted(result.relative_errors.items()):
        print(f"layer {i}: relative reconstruction error {err:.3f}")
    save_bottleneck(bottleneck, args.out)
    print(f"bottleneck saved to {args.out}")
    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    from .policy import AdaptivePolicy, ConfidenceEMADrift
    from .split import SplitRunner
    from .stream import (
        iter_image_frames,
        simulate_stream,
        summarize_stream,
        transport_feature_bytes,
        yolo_inferer,
    )

    if args.bottleneck:
        from .bottleneck import BottleneckTransport, load_bottleneck

        transport = BottleneckTransport(load_bottleneck(args.bottleneck), compress=True)
    else:
        from .transport import Int8Transport

        transport = Int8Transport(compress=True)

    yolo = _load_yolo(args.model)
    runner = SplitRunner(yolo.model, cut=args.cut, transport=transport)
    policy = AdaptivePolicy(
        conf_high=args.conf_high,
        conf_low=args.conf_low,
        drift=ConfidenceEMADrift(threshold=args.drift_threshold),
    )
    reports = simulate_stream(
        iter_image_frames(args.images, limit=args.limit),
        yolo_inferer(yolo, imgsz=args.imgsz, conf=args.conf),
        policy,
        transport_feature_bytes(runner, imgsz=args.imgsz),
        quality=args.quality,
    )

    print(f"{'frame':<28} {'mode':<11} {'conf':>5} {'KB':>9}  reason")
    for r in reports:
        conf = f"{r.frame_conf:.2f}" if r.frame_conf is not None else "-"
        print(f"{r.name:<28} {r.mode.value:<11} {conf:>5} {_kb(r.nbytes):>9}  {r.reason}")

    summary = summarize_stream(reports)
    retrain = [r.name for r in reports if r.retrain]
    print(
        f"\nmodes: {summary['frames_detections']:.0f} detections / "
        f"{summary['frames_features']:.0f} features / {summary['frames_frame']:.0f} frame"
    )
    print(
        f"adaptive total {_kb(summary['total_bytes'])} KB vs always-JPEG "
        f"{_kb(summary['baseline_jpeg_bytes'])} KB -> saved {summary['saved_vs_jpeg']:.1%}"
    )
    print(f"retraining candidates ({len(retrain)}): {', '.join(retrain) if retrain else 'none'}")
    if args.json:
        Path(args.json).write_text(json.dumps(summary | {"retrain": retrain}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolosplit",
        description="Split computing experiments for YOLO11 detection models.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True, help="weights (.pt) or model YAML")
        p.add_argument("--imgsz", type=int, default=640, help="inference image size")

    p_inspect = sub.add_parser("inspect", help="show layer graph and per-cut wire sizes")
    common(p_inspect)
    p_inspect.set_defaults(func=_cmd_inspect)

    p_measure = sub.add_parser("measure", help="JPEG vs wire tensor bytes on real images")
    common(p_measure)
    p_measure.add_argument("--images", required=True, help="directory of images")
    p_measure.add_argument("--cut", type=int, default=None, help=_CUT_HELP)
    p_measure.add_argument("--quality", type=int, default=85, help="JPEG quality")
    p_measure.add_argument("--limit", type=int, default=None, help="max images to measure")
    p_measure.add_argument("--json", default=None, help="write summary JSON here")
    p_measure.set_defaults(func=_cmd_measure)

    p_eval = sub.add_parser("evaluate", help="baseline vs split mAP on a dataset")
    common(p_eval)
    p_eval.add_argument("--data", required=True, help="ultralytics dataset YAML")
    p_eval.add_argument("--cut", type=int, default=None, help=_CUT_HELP)
    p_eval.add_argument(
        "--transport", choices=["int8", "fp16", "fp32"], default="int8", help="wire simulation"
    )
    p_eval.add_argument("--per-channel", action="store_true", help="per-channel quantisation")
    p_eval.add_argument("--compress", action="store_true", help="zlib on top of INT8")
    p_eval.add_argument("--bottleneck", default=None, help="trained bottleneck checkpoint")
    p_eval.add_argument("--device", default="cpu", help="cpu, 0, 0,1 ...")
    p_eval.add_argument("--batch", type=int, default=1, help="validation batch size")
    p_eval.add_argument("--json", default=None, help="write comparison JSON here")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_train = sub.add_parser("train-bottleneck", help="train the learned bottleneck at the cut")
    common(p_train)
    p_train.add_argument("--images", required=True, help="directory of training images")
    p_train.add_argument("--cut", type=int, default=None, help=_CUT_HELP)
    p_train.add_argument("--latent-channels", type=int, default=8, help="latent channels/level")
    p_train.add_argument("--stride", type=int, default=2, help="latent spatial downsampling")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--limit", type=int, default=None, help="max training images")
    p_train.add_argument("--device", default="cpu", help="cpu, 0, ...")
    p_train.add_argument(
        "--no-quant-noise", action="store_true", help="disable simulated INT8 noise"
    )
    p_train.add_argument("--out", default="bottleneck.pt", help="checkpoint output path")
    p_train.set_defaults(func=_cmd_train_bottleneck)

    p_stream = sub.add_parser("stream", help="simulate the adaptive edge->cloud stream")
    common(p_stream)
    p_stream.add_argument("--images", required=True, help="directory of frames")
    p_stream.add_argument("--cut", type=int, default=None, help=_CUT_HELP)
    p_stream.add_argument("--bottleneck", default=None, help="trained bottleneck checkpoint")
    p_stream.add_argument("--conf", type=float, default=0.25, help="detector confidence floor")
    p_stream.add_argument("--conf-high", type=float, default=0.75, help="detections-only above")
    p_stream.add_argument("--conf-low", type=float, default=0.4, help="full frame below")
    p_stream.add_argument("--drift-threshold", type=float, default=0.5, help="EMA drift trigger")
    p_stream.add_argument("--quality", type=int, default=85, help="JPEG quality")
    p_stream.add_argument("--limit", type=int, default=None, help="max frames")
    p_stream.add_argument("--json", default=None, help="write summary JSON here")
    p_stream.set_defaults(func=_cmd_stream)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
