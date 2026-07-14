"""Command-line interface: ``yolosplit inspect | measure | evaluate``."""

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

    if args.transport == "int8":
        return Int8Transport(axis=1 if args.per_channel else None, compress=args.compress)
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
    p_measure.add_argument("--cut", type=int, default=None, help="cut layer (default: backbone)")
    p_measure.add_argument("--quality", type=int, default=85, help="JPEG quality")
    p_measure.add_argument("--limit", type=int, default=None, help="max images to measure")
    p_measure.add_argument("--json", default=None, help="write summary JSON here")
    p_measure.set_defaults(func=_cmd_measure)

    p_eval = sub.add_parser("evaluate", help="baseline vs split mAP on a dataset")
    common(p_eval)
    p_eval.add_argument("--data", required=True, help="ultralytics dataset YAML")
    p_eval.add_argument("--cut", type=int, default=None, help="cut layer (default: backbone)")
    p_eval.add_argument(
        "--transport", choices=["int8", "fp16", "fp32"], default="int8", help="wire simulation"
    )
    p_eval.add_argument("--per-channel", action="store_true", help="per-channel quantisation")
    p_eval.add_argument("--compress", action="store_true", help="zlib on top of INT8")
    p_eval.add_argument("--device", default="cpu", help="cpu, 0, 0,1 ...")
    p_eval.add_argument("--batch", type=int, default=1, help="validation batch size")
    p_eval.add_argument("--json", default=None, help="write comparison JSON here")
    p_eval.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
