"""Offline training of the bottleneck: distillation with a frozen detector.

The detector never changes: for every image the edge half produces the wire
tensors, and the bottleneck is trained to carry them through its latent (with
simulated INT8 noise). Two objectives are mixed by ``task_weight``:

* **feature** — per-level MSE normalised by the target's mean energy, so every
  wire level contributes equally regardless of scale;
* **task** — the same normalised MSE on the *head output*, with the gradient
  taken back through the frozen tail.

Feature error alone treats every activation as equally worth keeping, and it
plateaus: at the default configuration a codec trained on it stalls near 0.6
relative error however long it runs, because most of the capacity goes to
activations no prediction depends on. The task term prices them correctly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck import Bottleneck, BottleneckTransport, Latents
from .measure import IMAGE_SUFFIXES, to_input_tensor
from .split import SplitRunner, Transport, primary_output

try:  # tqdm ships with ultralytics; degrade gracefully if it is ever absent.
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def normalize_device(device: str) -> str:
    """Accept ultralytics-style device strings (``"0"``, ``"0,1"``) for torch.

    ``torch.device`` rejects a bare GPU index like ``"0"``, but that is the
    convention the rest of the tooling (and ``--device``) uses. Map it to
    ``"cuda:0"``; ``cpu``/``mps``/``cuda*`` pass through unchanged. Only the
    first index is used — training here is single-GPU.
    """
    d = str(device).strip().lower()
    if d in ("", "cpu"):
        return "cpu"
    if d == "mps" or d.startswith("cuda"):
        return d
    first = d.split(",")[0]
    if first.isdigit():
        return f"cuda:{first}"
    return d


@dataclass
class TrainResult:
    """What a training run is worth.

    ``output_error`` is the headline: the relative error the codec induces on
    the model's output, over the deployed encode → INT8 → decode path, on
    frames held out of training. ``relative_errors`` (per-level feature
    reconstruction, measured on training frames) is kept because it is what
    the loss minimises, but it is a diagnostic, not a verdict — it can improve
    while accuracy gets worse.
    """

    epoch_losses: list[float] = field(default_factory=list)
    relative_errors: dict[int, float] = field(default_factory=dict)
    output_error: float | None = None
    train_images: int = 0
    val_images: int = 0


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


def _normalised_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Normalise by the target's mean energy (not variance: near-constant tensors
    # would explode the scale and destabilise the first optimiser steps).
    return F.mse_loss(recon, target) / (target.pow(2).mean() + 1e-6)


def _distillation_loss(
    recon: dict[int, torch.Tensor], target: dict[int, torch.Tensor]
) -> torch.Tensor:
    losses = [_normalised_mse(recon[i], target[i]) for i in sorted(target)]
    return torch.stack(losses).mean()


def _tensors(out: object) -> list[torch.Tensor]:
    """Every tensor in a model output, whatever container it arrives in.

    Heads return different shapes — a tensor, a tuple of decoded predictions
    plus raw maps, a list per level. Walking the structure keeps the task loss
    architecture-agnostic, which is the whole point of the adapter layer.
    """
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, dict):
        out = list(out.values())
    if isinstance(out, (list, tuple)):
        return [t for item in out for t in _tensors(item)]
    return []


def _task_loss(
    runner: SplitRunner, recon: dict[int, torch.Tensor], target_out: list[torch.Tensor]
) -> torch.Tensor:
    """How far the *head output* moves when the tail runs on reconstructions.

    Reconstruction error treats every activation as equally worth keeping. The
    detector does not: much of the feature energy never reaches a prediction.
    Backpropagating through the frozen tail lets the codec spend its bits where
    the output actually moves.
    """
    through = _tensors(runner.cloud(recon, grad=True))
    if len(through) != len(target_out):  # pragma: no cover - defensive
        raise RuntimeError("cloud half returned a different output structure for the same input")
    losses = [_normalised_mse(a, b) for a, b in zip(through, target_out, strict=True)]
    return torch.stack(losses).mean()


@torch.no_grad()
def output_error(
    runner: SplitRunner,
    bottleneck: Bottleneck,
    paths: list[Path],
    imgsz: int = 640,
    batch: int = 4,
    device: str = "cpu",
    transport: Transport | None = None,
) -> float:
    """Relative error the codec induces on the model's output, averaged.

    The honest quality axis for a codec, and cheap: one extra pass through the
    cloud half per batch, against mAP's full validation run. Reconstruction
    error measures something else — a codec can improve here while getting
    worse there, and does.

    Measured on :func:`~axonmesh.split.primary_output`, i.e. the same tensor
    the server's postprocess consumes. ``transport`` prices the codec as it is
    actually deployed (encode → INT8 → decode); leaving it out measures the
    autoencoder alone, which is a different and more flattering question.
    """
    dev = torch.device(normalize_device(device))
    errors = []
    for start in range(0, len(paths), batch):
        x = _load_batch(paths[start : start + batch], imgsz, dev)
        wire = runner.edge(x)
        received = bottleneck(wire) if transport is None else transport(wire)[0]
        baseline = _tensors(primary_output(runner.cloud(wire)))
        through = _tensors(primary_output(runner.cloud(received)))
        errors.append(
            mean(
                ((a - b).norm() / (b.norm() + 1e-8)).item()
                for a, b in zip(through, baseline, strict=True)
            )
        )
    return mean(errors)


def train_bottleneck(
    det_model: nn.Module,
    images_dir: str | Path,
    cut: int | None = None,
    latent_channels: Latents = 8,
    stride: int = 2,
    epochs: int = 5,
    batch: int = 4,
    lr: float = 1e-3,
    imgsz: int = 640,
    limit: int | None = None,
    device: str = "cpu",
    seed: int = 15,
    quant_noise: bool = True,
    progress: bool = True,
    task_weight: float = 0.5,
    val_fraction: float = 0.1,
) -> tuple[Bottleneck, TrainResult]:
    """Train a bottleneck for ``det_model`` on the images in ``images_dir``.

    Returns the trained bottleneck (in eval mode) and the loss history. The
    detector is frozen and left in eval mode; only bottleneck weights change.
    ``progress`` shows a per-epoch tqdm bar with the running loss (set it False
    for quiet runs, e.g. the sweep or tests).

    ``task_weight`` mixes the two objectives::

        loss = (1 - task_weight) * feature_mse + task_weight * head_output_mse

    At 0 this is pure feature distillation (every activation equally worth
    keeping). Above 0 the gradient also comes back through the frozen tail, so
    the codec learns which activations the predictions actually depend on —
    which is what the accuracy number is measured on. Costs one extra forward
    and backward through the cloud half per step.
    """
    dev = torch.device(normalize_device(device))
    det_model = det_model.to(dev).float().eval()
    for p in det_model.parameters():
        p.requires_grad_(False)

    runner = SplitRunner(det_model, cut=cut)
    bottleneck = Bottleneck.for_runner(
        runner, latent_channels=latent_channels, stride=stride, imgsz=imgsz
    ).to(dev)
    optimizer = torch.optim.Adam(bottleneck.parameters(), lr=lr, weight_decay=0.0)

    every_path = _image_paths(images_dir, limit)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    # Hold frames out before the first step. Reporting a codec's quality on
    # frames it trained on is how a codec that has memorised 96 images looks
    # finished; the split is deterministic in `seed` so a rerun is comparable.
    rng.shuffle(every_path)
    n_val = min(int(len(every_path) * val_fraction), 256)
    val_paths, paths = every_path[:n_val], every_path[n_val:]
    result = TrainResult(train_images=len(paths), val_images=len(val_paths))

    show = progress and tqdm is not None
    bottleneck.train()
    for epoch in range(epochs):
        rng.shuffle(paths)
        losses: list[float] = []
        starts = range(0, len(paths), batch)
        bar = (
            tqdm(starts, desc=f"epoch {epoch + 1}/{epochs}", unit="batch", leave=False)
            if show
            else starts
        )
        for start in bar:
            x = _load_batch(paths[start : start + batch], imgsz, dev)
            wire = runner.edge(x)  # no_grad inside: targets are detached
            recon = bottleneck(wire, quant_noise=quant_noise)
            loss = _distillation_loss(recon, wire)
            if task_weight > 0:
                target_out = [t.detach() for t in _tensors(runner.cloud(wire))]
                task = _task_loss(runner, recon, target_out)
                loss = (1 - task_weight) * loss + task_weight * task
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if show:
                bar.set_postfix(loss=f"{losses[-1]:.4f}")
        mean_loss = sum(losses) / len(losses)
        result.epoch_losses.append(mean_loss)
        if progress:
            print(f"epoch {epoch + 1}/{epochs}: normalised MSE {mean_loss:.4f}")

    bottleneck.eval()
    with torch.no_grad():
        x = _load_batch(paths[: min(batch, len(paths))], imgsz, dev)
        wire = runner.edge(x)
        recon = bottleneck(wire)
        result.relative_errors = {
            i: (recon[i] - wire[i]).norm().item() / (wire[i].norm().item() + 1e-8) for i in wire
        }
    if val_paths:
        result.output_error = output_error(
            runner,
            bottleneck,
            val_paths,
            imgsz=imgsz,
            batch=batch,
            device=device,
            transport=BottleneckTransport(bottleneck, compress=True),
        )
        if progress:
            print(
                f"held-out output error {result.output_error:.4f} "
                f"({len(val_paths)} frames the codec never trained on)"
            )
    return bottleneck, result
