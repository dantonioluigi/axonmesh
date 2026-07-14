"""Offline training of the bottleneck: feature distillation with a frozen detector.

The detector never changes: for every image the edge half produces the wire
tensors, and the bottleneck is trained to reconstruct them through its latent
(with simulated INT8 noise). The loss is per-level MSE normalised by the
target's mean energy, so P3/P4/P5 contribute equally regardless of scale.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck import Bottleneck
from .measure import IMAGE_SUFFIXES, to_input_tensor
from .split import SplitRunner


@dataclass
class TrainResult:
    """Per-epoch mean loss and final per-level relative reconstruction error."""

    epoch_losses: list[float] = field(default_factory=list)
    relative_errors: dict[int, float] = field(default_factory=dict)


def _image_paths(images_dir: str | Path, limit: int | None) -> list[Path]:
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[
        :limit
    ]
    if not paths:
        raise FileNotFoundError(f"no images found in {images_dir}")
    return paths


def _load_batch(paths: list[Path], imgsz: int, device: torch.device) -> torch.Tensor:
    tensors = []
    for p in paths:
        image = cv2.imread(str(p))
        if image is None:
            raise ValueError(f"could not read image {p}")
        tensors.append(to_input_tensor(image, imgsz))
    return torch.cat(tensors).to(device)


def _distillation_loss(
    recon: dict[int, torch.Tensor], target: dict[int, torch.Tensor]
) -> torch.Tensor:
    # Normalise by the target's mean energy (not variance: near-constant levels
    # would explode the scale and destabilise the first optimiser steps).
    losses = [
        F.mse_loss(recon[i], target[i]) / (target[i].pow(2).mean() + 1e-6) for i in sorted(target)
    ]
    return torch.stack(losses).mean()


def train_bottleneck(
    det_model: nn.Module,
    images_dir: str | Path,
    cut: int | None = None,
    latent_channels: int = 8,
    stride: int = 2,
    epochs: int = 5,
    batch: int = 4,
    lr: float = 1e-3,
    imgsz: int = 640,
    limit: int | None = None,
    device: str = "cpu",
    seed: int = 15,
    quant_noise: bool = True,
) -> tuple[Bottleneck, TrainResult]:
    """Train a bottleneck for ``det_model`` on the images in ``images_dir``.

    Returns the trained bottleneck (in eval mode) and the loss history. The
    detector is frozen and left in eval mode; only bottleneck weights change.
    """
    dev = torch.device(device)
    det_model = det_model.to(dev).float().eval()
    for p in det_model.parameters():
        p.requires_grad_(False)

    runner = SplitRunner(det_model, cut=cut)
    bottleneck = Bottleneck.for_runner(
        runner, latent_channels=latent_channels, stride=stride, imgsz=imgsz
    ).to(dev)
    optimizer = torch.optim.Adam(bottleneck.parameters(), lr=lr, weight_decay=0.0)

    paths = _image_paths(images_dir, limit)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    result = TrainResult()

    bottleneck.train()
    for _ in range(epochs):
        rng.shuffle(paths)
        losses = []
        for start in range(0, len(paths), batch):
            x = _load_batch(paths[start : start + batch], imgsz, dev)
            wire = runner.edge(x)  # no_grad inside: targets are detached
            recon = bottleneck(wire, quant_noise=quant_noise)
            loss = _distillation_loss(recon, wire)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        result.epoch_losses.append(sum(losses) / len(losses))

    bottleneck.eval()
    with torch.no_grad():
        x = _load_batch(paths[: min(batch, len(paths))], imgsz, dev)
        wire = runner.edge(x)
        recon = bottleneck(wire)
        result.relative_errors = {
            i: (recon[i] - wire[i]).norm().item() / (wire[i].norm().item() + 1e-8) for i in wire
        }
    return bottleneck, result
