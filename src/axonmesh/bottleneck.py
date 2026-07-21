"""Learned bottleneck at the cut: the piece that has to close the ~30x gap.

The first measurement showed that INT8(+zlib) on the raw P3/P4/P5 pyramid ships
~30x more bytes than the JPEG frame. A small convolutional autoencoder per wire
tensor squeezes each level to ``latent_channels`` channels at ``1/stride``
spatial resolution before quantisation; with the defaults (8 channels, stride 2)
the INT8 latent for YOLO11l @640 is ~17 KB/frame vs ~47 KB of JPEG q85.

The encoder runs on the edge, the decoder on the cloud; both are trained
offline (see :mod:`axonmesh.train`) with the detector frozen, so deploying a
bottleneck never touches the detector weights.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .quantize import QuantizedTensor, dequantize, quantize
from .split import SplitRunner
from .topology import probe_output_shapes


class LevelCodec(nn.Module):
    """Encoder/decoder pair for one wire tensor (one pyramid level)."""

    def __init__(self, channels: int, latent_channels: int, stride: int = 2) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.stride = stride
        hidden = max(2 * latent_channels, 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, latent_channels, 3, stride, 1),
        )
        decoder: list[nn.Module] = [nn.Conv2d(latent_channels, hidden, 3, 1, 1), nn.SiLU()]
        if stride > 1:
            decoder.append(nn.Upsample(scale_factor=stride, mode="nearest"))
        decoder.append(nn.Conv2d(hidden, channels, 3, 1, 1))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, t: torch.Tensor) -> torch.Tensor:
        h, w = t.shape[-2:]
        if h % self.stride or w % self.stride:
            raise ValueError(
                f"spatial size {h}x{w} not divisible by stride {self.stride}; "
                "the decoder could not restore the original shape"
            )
        return self.encoder(t)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


Latents = int | dict[int, int]


def resolve_latents(channels: dict[int, int], latent_channels: Latents) -> dict[int, int]:
    """Per-level latent widths, from either one number or an explicit mapping.

    One number for every level spends the budget where the pixels are, which is
    not where the accuracy is. Measured on YOLO11n at the backbone cut, the
    shallowest wire level takes 72% of the bytes and accounts for 22% of the
    error the codec induces, while the deepest takes 6% and accounts for 47%:
    the levels that matter most are the cheapest to widen, because they are
    spatially tiny. Pass a mapping to spend accordingly.
    """
    if isinstance(latent_channels, dict):
        missing = set(channels) - set(latent_channels)
        if missing:
            raise ValueError(
                f"no latent width given for wire level(s) {sorted(missing)}; "
                f"the wire set is {sorted(channels)}"
            )
        return {i: int(latent_channels[i]) for i in channels}
    return dict.fromkeys(channels, int(latent_channels))


class Bottleneck(nn.Module):
    """One :class:`LevelCodec` per wire tensor, keyed by layer index."""

    def __init__(
        self, channels: dict[int, int], latent_channels: Latents = 8, stride: int = 2
    ) -> None:
        super().__init__()
        latents = resolve_latents(channels, latent_channels)
        self.codecs = nn.ModuleDict(
            {str(i): LevelCodec(c, latents[i], stride) for i, c in channels.items()}
        )
        self.config: dict[str, Any] = {
            "channels": dict(channels),
            "latent_channels": latents,
            "stride": stride,
        }

    @classmethod
    def for_runner(
        cls,
        runner: SplitRunner,
        latent_channels: Latents = 8,
        stride: int = 2,
        imgsz: int = 640,
    ) -> Bottleneck:
        """Build codecs matching the channel counts of the runner's wire set."""
        shapes = probe_output_shapes(runner.det_model, imgsz=imgsz)
        channels = {i: shapes[i][1] for i in runner.wire}
        return cls(channels, latent_channels=latent_channels, stride=stride)

    def encode(self, wire: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {i: self.codecs[str(i)].encode(t) for i, t in wire.items()}

    def decode(self, latents: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {i: self.codecs[str(i)].decode(z) for i, z in latents.items()}

    def forward(
        self, wire: dict[int, torch.Tensor], quant_noise: bool = False
    ) -> dict[int, torch.Tensor]:
        """Full reconstruction pass, optionally simulating INT8 error while training.

        ``quant_noise`` adds uniform noise with the amplitude of the INT8
        quantisation step to the latents, so the decoder learns to be robust to
        the lossy wire it will actually see at inference time.
        """
        latents = self.encode(wire)
        if quant_noise:
            latents = {i: _add_quant_noise(z) for i, z in latents.items()}
        return self.decode(latents)


def _add_quant_noise(z: torch.Tensor) -> torch.Tensor:
    span = (z.detach().amax() - z.detach().amin()).clamp(min=1e-8)
    return z + (torch.rand_like(z) - 0.5) * span / 255.0


def save_bottleneck(bottleneck: Bottleneck, path: str | Path) -> None:
    torch.save({"config": bottleneck.config, "state_dict": bottleneck.state_dict()}, str(path))


def load_bottleneck(path: str | Path, map_location: str = "cpu") -> Bottleneck:
    checkpoint = torch.load(str(path), map_location=map_location, weights_only=True)
    config = checkpoint["config"]
    bottleneck = Bottleneck(
        channels=config["channels"],
        latent_channels=config["latent_channels"],
        stride=config["stride"],
    )
    bottleneck.load_state_dict(checkpoint["state_dict"])
    return bottleneck.eval()


@dataclass(frozen=True)
class BottleneckTransport:
    """Encode → INT8 → wire → decode. Byte counts are the serialised latents."""

    bottleneck: Bottleneck
    axis: int | None = None
    compress: bool = False
    level: int = 6

    def __call__(self, wire: dict[int, torch.Tensor]) -> tuple[dict[int, torch.Tensor], int]:
        device = next(iter(wire.values())).device
        self.bottleneck.to(device)  # match the wire's device (no-op if already there)
        with torch.no_grad():
            latents = self.bottleneck.encode(wire)
            received = {}
            nbytes = 0
            for i, z in latents.items():
                payload = quantize(z, axis=self.axis).to_bytes()
                nbytes += len(zlib.compress(payload, self.level)) if self.compress else len(payload)
                # from_bytes rebuilds on CPU; move back before the decoder runs.
                received[i] = dequantize(QuantizedTensor.from_bytes(payload)).to(device)
            return self.bottleneck.decode(received), nbytes
