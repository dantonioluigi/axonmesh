"""Bandwidth measurements: JPEG frames vs intermediate wire tensors.

The honest comparison is against the *same pixels the model sees*: frames are
letterboxed to the inference resolution and JPEG-encoded at the production
quality, then compared with the wire set produced by the edge half at that same
resolution (raw fp32/fp16, INT8, INT8+zlib).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

from .quantize import quantize
from .split import SplitRunner

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def jpeg_nbytes(image_bgr: np.ndarray, quality: int = 85) -> int:
    """Size in bytes of the JPEG encoding of a BGR image."""
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return int(buf.nbytes)


def letterbox(image_bgr: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """Resize keeping aspect ratio and pad to ``imgsz x imgsz`` (YOLO-style)."""
    h, w = image_bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    return cv2.copyMakeBorder(
        resized,
        top,
        imgsz - nh - top,
        left,
        imgsz - nw - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def to_input_tensor(image_bgr: np.ndarray, imgsz: int = 640) -> torch.Tensor:
    """Letterbox + BGR→RGB + CHW + [0,1] float32 batch of one."""
    boxed = letterbox(image_bgr, imgsz)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0


@dataclass(frozen=True)
class FrameMeasurement:
    """Byte counts for one frame at inference resolution."""

    name: str
    file_bytes: int  # original file as stored on disk
    jpeg_bytes: int  # letterboxed frame re-encoded at the production quality
    wire_fp32: int
    wire_fp16: int
    wire_int8: int
    wire_int8_zlib: int

    @property
    def int8_vs_jpeg(self) -> float:
        """Compression ratio of INT8 wire vs JPEG (>1 means INT8 is smaller)."""
        return self.jpeg_bytes / self.wire_int8


def measure_frame(
    runner: SplitRunner,
    image_bgr: np.ndarray,
    name: str = "",
    file_bytes: int = 0,
    imgsz: int = 640,
    quality: int = 85,
    zlib_level: int = 6,
) -> FrameMeasurement:
    """Run the edge half on one frame and price every wire option."""
    x = to_input_tensor(image_bgr, imgsz)
    wire = runner.edge(x)
    tensors = list(wire.values())
    payloads = [quantize(t).to_bytes() for t in tensors]
    return FrameMeasurement(
        name=name,
        file_bytes=file_bytes,
        jpeg_bytes=jpeg_nbytes(letterbox(image_bgr, imgsz), quality),
        wire_fp32=sum(t.numel() * 4 for t in tensors),
        wire_fp16=sum(t.numel() * 2 for t in tensors),
        wire_int8=sum(len(p) for p in payloads),
        wire_int8_zlib=sum(len(zlib.compress(p, zlib_level)) for p in payloads),
    )


def measure_directory(
    runner: SplitRunner,
    images_dir: str | Path,
    imgsz: int = 640,
    quality: int = 85,
    limit: int | None = None,
) -> list[FrameMeasurement]:
    """Measure every image in a directory (sorted, optionally capped at ``limit``)."""
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[
        :limit
    ]
    if not paths:
        raise FileNotFoundError(f"no images found in {images_dir}")
    results = []
    for p in paths:
        image = cv2.imread(str(p))
        if image is None:
            raise ValueError(f"could not read image {p}")
        results.append(
            measure_frame(
                runner,
                image,
                name=p.name,
                file_bytes=p.stat().st_size,
                imgsz=imgsz,
                quality=quality,
            )
        )
    return results


_BYTE_FIELDS = ("file_bytes", "jpeg_bytes", "wire_fp32", "wire_fp16", "wire_int8", "wire_int8_zlib")


def summarize(measurements: list[FrameMeasurement]) -> dict[str, float]:
    """Mean of every byte field plus the mean INT8-vs-JPEG compression ratio."""
    summary = {f"mean_{n}": mean(getattr(m, n) for m in measurements) for n in _BYTE_FIELDS}
    summary["mean_int8_vs_jpeg"] = mean(m.int8_vs_jpeg for m in measurements)
    summary["frames"] = len(measurements)
    return summary


def to_markdown(measurements: list[FrameMeasurement]) -> str:
    """Render measurements (plus a mean row) as a GitHub-flavoured markdown table."""
    header = (
        "| frame | file KB | jpeg KB | fp32 KB | fp16 KB | int8 KB | int8+z KB | int8 vs jpeg |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    kb = 1024.0

    def row(name: str, m: FrameMeasurement | dict[str, float]) -> str:
        get = (lambda k: m[f"mean_{k}"]) if isinstance(m, dict) else (lambda k: getattr(m, k))
        ratio = m["mean_int8_vs_jpeg"] if isinstance(m, dict) else m.int8_vs_jpeg
        cells = [
            f"{get(k) / kb:.1f}"
            for k in (
                "file_bytes",
                "jpeg_bytes",
                "wire_fp32",
                "wire_fp16",
                "wire_int8",
                "wire_int8_zlib",
            )
        ]
        return f"| {name} | " + " | ".join(cells) + f" | {ratio:.2f}x |"

    lines = [header]
    lines += [row(m.name, m) for m in measurements]
    lines.append(row("**mean**", summarize(measurements)))
    return "\n".join(lines)
