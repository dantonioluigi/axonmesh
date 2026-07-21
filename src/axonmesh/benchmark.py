"""Unified benchmark: what a split configuration actually costs.

A deployment decision needs four numbers together — accuracy, throughput,
bandwidth and latency — and they only mean something side by side: a cut that
halves the bytes is worthless if it doubles the edge latency. This module
measures them in one pass, broken down per stage of the split (preprocess,
edge half, wire, cloud half), so two configurations can be compared honestly,
including against not splitting at all (the JPEG baseline).

Power is sampled when the platform exposes it (Jetson's INA3221 rails); it is
reported as unavailable elsewhere rather than guessed.
"""

from __future__ import annotations

import glob
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn as nn

from .measure import jpeg_nbytes, letterbox, to_input_tensor
from .split import SplitRunner, raw_nbytes
from .stream import iter_image_frames
from .train import normalize_device

#: Jetson exposes board power on INA3221 rails via hwmon.
_JETSON_POWER_GLOBS = (
    "/sys/bus/i2c/drivers/ina3221x/*/hwmon/hwmon*/power1_input",
    "/sys/class/hwmon/hwmon*/power1_input",
)

PowerSampler = Callable[[], float | None]


def read_jetson_power() -> float | None:
    """Instantaneous board power in watts, or ``None`` off a supported board.

    Reads the first INA3221 rail exposed through hwmon (values are milliwatts).
    """
    for pattern in _JETSON_POWER_GLOBS:
        for path in sorted(glob.glob(pattern)):
            try:
                return int(Path(path).read_text().strip()) / 1000.0
            except (OSError, ValueError):
                continue
    return None


def _sync(device: torch.device) -> None:
    """Make GPU work observable before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class StageTimings:
    """Mean milliseconds per frame, split by pipeline stage."""

    prep_ms: float
    edge_ms: float
    wire_ms: float
    cloud_ms: float

    @property
    def total_ms(self) -> float:
        return self.prep_ms + self.edge_ms + self.wire_ms + self.cloud_ms

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms else 0.0


@dataclass
class BenchmarkResult:
    """Everything one configuration costs, in one object."""

    frames: int
    device: str
    cut: int
    timings: StageTimings
    wire_bytes: float  # mean bytes/frame actually shipped
    jpeg_bytes: float  # mean bytes/frame if we shipped the frame instead
    power_w: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    baseline_map50_95: float | None = None

    @property
    def bandwidth_mbps(self) -> float:
        """Link rate this configuration needs to sustain its own frame rate."""
        return self.wire_bytes * 8 * self.timings.fps / 1e6

    @property
    def vs_jpeg(self) -> float:
        """How much smaller the wire is than shipping the frame (>1 is better)."""
        return self.jpeg_bytes / self.wire_bytes if self.wire_bytes else 0.0

    @property
    def delta_map50_95(self) -> float | None:
        if self.map50_95 is None or self.baseline_map50_95 is None:
            return None
        return self.map50_95 - self.baseline_map50_95

    def to_dict(self) -> dict:
        return asdict(self) | {
            "total_ms": self.timings.total_ms,
            "fps": self.timings.fps,
            "bandwidth_mbps": self.bandwidth_mbps,
            "vs_jpeg": self.vs_jpeg,
            "delta_map50_95": self.delta_map50_95,
        }

    def to_markdown(self) -> str:
        t = self.timings
        rows = [
            ("frames", f"{self.frames}"),
            ("device", self.device),
            ("cut", str(self.cut)),
            ("latency total", f"{t.total_ms:.1f} ms"),
            ("  · preprocess", f"{t.prep_ms:.1f} ms"),
            ("  · edge half", f"{t.edge_ms:.1f} ms"),
            ("  · wire (encode+codec)", f"{t.wire_ms:.1f} ms"),
            ("  · cloud half", f"{t.cloud_ms:.1f} ms"),
            ("throughput", f"{t.fps:.1f} FPS"),
            ("wire", f"{self.wire_bytes / 1024:.1f} KB/frame"),
            ("JPEG baseline", f"{self.jpeg_bytes / 1024:.1f} KB/frame"),
            ("wire vs JPEG", f"{self.vs_jpeg:.2f}x"),
            ("bandwidth needed", f"{self.bandwidth_mbps:.1f} Mbps"),
            ("power", f"{self.power_w:.1f} W" if self.power_w is not None else "n/a"),
            ("mAP50-95 (split)", f"{self.map50_95:.3f}" if self.map50_95 is not None else "n/a"),
            (
                "mAP50-95 delta",
                f"{self.delta_map50_95:+.3f}" if self.delta_map50_95 is not None else "n/a",
            ),
        ]
        body = "\n".join(f"| {k} | {v} |" for k, v in rows)
        return f"| metric | value |\n|---|---|\n{body}"


def benchmark_split(
    det_model: nn.Module,
    frames: Iterable[tuple[str, np.ndarray]],
    cut: int | None = None,
    transport=None,
    imgsz: int = 640,
    device: str = "cpu",
    warmup: int = 3,
    quality: int = 85,
    power_sampler: PowerSampler | None = read_jetson_power,
) -> BenchmarkResult:
    """Time one split configuration end to end, stage by stage.

    ``frames`` yields ``(name, BGR image)``. The first ``warmup`` frames are run
    but not measured, so lazy CUDA/cuDNN initialisation does not land in the
    numbers.
    """
    dev = torch.device(normalize_device(device))
    det_model = det_model.to(dev).float().eval()
    runner = SplitRunner(det_model, cut=cut, transport=transport)

    prep, edge, wire_t, cloud = [], [], [], []
    wire_sizes, jpeg_sizes, power = [], [], []
    counted = 0

    for index, (_name, image) in enumerate(frames):
        measured = index >= warmup

        t0 = time.perf_counter()
        x = to_input_tensor(image, imgsz).to(dev)
        _sync(dev)
        t1 = time.perf_counter()

        wire = runner.edge(x)
        _sync(dev)
        t2 = time.perf_counter()

        if transport is not None:
            received, nbytes = transport(wire)
        else:
            received, nbytes = wire, raw_nbytes(wire)
        _sync(dev)
        t3 = time.perf_counter()

        runner.cloud(received)
        _sync(dev)
        t4 = time.perf_counter()

        if not measured:
            continue
        counted += 1
        prep.append((t1 - t0) * 1000)
        edge.append((t2 - t1) * 1000)
        wire_t.append((t3 - t2) * 1000)
        cloud.append((t4 - t3) * 1000)
        wire_sizes.append(nbytes)
        jpeg_sizes.append(jpeg_nbytes(letterbox(image, imgsz), quality))
        if power_sampler is not None:
            watts = power_sampler()
            if watts is not None:
                power.append(watts)

    if not counted:
        raise ValueError(f"no frames left to measure after {warmup} warmup frames")

    return BenchmarkResult(
        frames=counted,
        device=str(dev),
        cut=runner.cut,
        timings=StageTimings(mean(prep), mean(edge), mean(wire_t), mean(cloud)),
        wire_bytes=mean(wire_sizes),
        jpeg_bytes=mean(jpeg_sizes),
        power_w=mean(power) if power else None,
    )


def benchmark_directory(
    det_model: nn.Module,
    images_dir: str | Path,
    limit: int | None = None,
    **kwargs,
) -> BenchmarkResult:
    """Convenience wrapper: benchmark over every image in a directory."""
    return benchmark_split(det_model, iter_image_frames(images_dir, limit=limit), **kwargs)


def to_json(result: BenchmarkResult) -> str:
    return json.dumps(result.to_dict(), indent=2)
