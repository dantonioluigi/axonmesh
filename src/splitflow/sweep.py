"""Bottleneck configuration sweep: the bytes-vs-quality Pareto curve (Phase 1).

Trains one bottleneck per configuration (latent channels x stride) and prices
each on the same sample of frames: serialised INT8 latent bytes (plain and
zlib), the JPEG baseline, and the feature reconstruction error. Optionally —
when a dataset YAML is given — the end-to-end mAP cost per configuration.

Configurations whose stride does not divide the feature maps at the chosen
image size are skipped, not fatal: a sweep that dies mid-grid wastes every
configuration trained before the crash.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean

import cv2
import torch.nn as nn

from .bottleneck import Bottleneck
from .measure import jpeg_nbytes, letterbox, to_input_tensor
from .quantize import quantize
from .split import SplitRunner
from .train import TrainResult, _image_paths, train_bottleneck


@dataclass(frozen=True)
class SweepConfig:
    latent_channels: int
    stride: int


@dataclass
class SweepResult:
    """One trained configuration, priced on the sample frames."""

    config: SweepConfig
    int8_bytes: int  # mean serialised latent bytes/frame
    int8_zlib_bytes: int
    jpeg_bytes: int  # mean JPEG baseline on the same letterboxed frames
    relative_error: float  # mean over pyramid levels
    epoch_losses: list[float]
    pareto: bool = False

    @property
    def vs_jpeg(self) -> float:
        """Compression ratio vs JPEG (>1 means the latent is smaller)."""
        return self.jpeg_bytes / self.int8_zlib_bytes


def _price_bottleneck(
    bottleneck: Bottleneck,
    runner: SplitRunner,
    sample_paths: list[Path],
    imgsz: int,
    quality: int,
) -> tuple[int, int, int]:
    """Mean (int8, int8+zlib, jpeg) bytes/frame over the sample."""
    plain, compressed, jpeg = [], [], []
    for path in sample_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not read image {path}")
        wire = runner.edge(to_input_tensor(image, imgsz))
        payloads = [quantize(z).to_bytes() for z in bottleneck.encode(wire).values()]
        plain.append(sum(len(p) for p in payloads))
        compressed.append(sum(len(zlib.compress(p, 6)) for p in payloads))
        jpeg.append(jpeg_nbytes(letterbox(image, imgsz), quality))
    return round(mean(plain)), round(mean(compressed)), round(mean(jpeg))


def mark_pareto(results: list[SweepResult]) -> None:
    """Flag configurations not dominated on (zlib bytes, reconstruction error)."""
    for r in results:
        r.pareto = not any(
            other is not r
            and other.int8_zlib_bytes <= r.int8_zlib_bytes
            and other.relative_error <= r.relative_error
            and (
                other.int8_zlib_bytes < r.int8_zlib_bytes or other.relative_error < r.relative_error
            )
            for other in results
        )


def run_sweep(
    det_model: nn.Module,
    images_dir: str | Path,
    latents: list[int],
    strides: list[int],
    cut: int | None = None,
    epochs: int = 5,
    batch: int = 4,
    lr: float = 1e-3,
    imgsz: int = 640,
    limit: int | None = None,
    device: str = "cpu",
    quality: int = 85,
    sample: int = 4,
) -> list[SweepResult]:
    """Train and price every (latent_channels, stride) configuration."""
    results: list[SweepResult] = []
    sample_paths = _image_paths(images_dir, limit)[:sample]
    for latent_channels, stride in product(latents, strides):
        try:
            bottleneck, train_result = train_bottleneck(
                det_model,
                images_dir,
                cut=cut,
                latent_channels=latent_channels,
                stride=stride,
                epochs=epochs,
                batch=batch,
                lr=lr,
                imgsz=imgsz,
                limit=limit,
                device=device,
                progress=False,  # the sweep prints its own per-config summary
            )
        except ValueError as err:
            print(f"skip latent={latent_channels} stride={stride}: {err}")
            continue
        runner = SplitRunner(det_model, cut=cut)
        int8, int8_zlib, jpeg = _price_bottleneck(bottleneck, runner, sample_paths, imgsz, quality)
        results.append(
            SweepResult(
                config=SweepConfig(latent_channels, stride),
                int8_bytes=int8,
                int8_zlib_bytes=int8_zlib,
                jpeg_bytes=jpeg,
                relative_error=_mean_error(train_result),
                epoch_losses=train_result.epoch_losses,
            )
        )
    mark_pareto(results)
    return results


def _mean_error(train_result: TrainResult) -> float:
    return mean(train_result.relative_errors.values())


def to_markdown(results: list[SweepResult]) -> str:
    """Sweep results as a markdown table, smallest wire first."""
    header = (
        "| latent | stride | int8 KB | int8+z KB | jpeg KB | vs jpeg | rel err | pareto |\n"
        "|---:|---:|---:|---:|---:|---:|---:|:---:|"
    )
    lines = [header]
    for r in sorted(results, key=lambda r: r.int8_zlib_bytes):
        lines.append(
            f"| {r.config.latent_channels} | {r.config.stride} "
            f"| {r.int8_bytes / 1024:.1f} | {r.int8_zlib_bytes / 1024:.1f} "
            f"| {r.jpeg_bytes / 1024:.1f} | {r.vs_jpeg:.2f}x "
            f"| {r.relative_error:.3f} | {'*' if r.pareto else ''} |"
        )
    return "\n".join(lines)


def to_dicts(results: list[SweepResult]) -> list[dict]:
    return [
        {
            "latent_channels": r.config.latent_channels,
            "stride": r.config.stride,
            "int8_bytes": r.int8_bytes,
            "int8_zlib_bytes": r.int8_zlib_bytes,
            "jpeg_bytes": r.jpeg_bytes,
            "vs_jpeg": r.vs_jpeg,
            "relative_error": r.relative_error,
            "epoch_losses": r.epoch_losses,
            "pareto": r.pareto,
        }
        for r in results
    ]
