"""Which wire level deserves the bits — measured, before any codec is trained.

A single ``latent_channels`` spends the budget where the pixels are. That is
only correct if every wire level is equally sensitive to distortion, and on a
feature pyramid it is not: the shallow level is spatially large (most of the
bytes) and carries redundant, locally smooth activations, while the deep level
is tiny (almost no bytes) and every channel of it reaches the head.

Measuring that needs no training. Distort one level at a time — coarse
quantisation is a stand-in for any lossy codec — and watch how far the model's
output moves. The result is a sensitivity per level, and from it an allocation
that equalises the value of the last byte spent on each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .quantize import dequantize, quantize
from .split import SplitRunner, primary_output


@dataclass(frozen=True)
class LevelSensitivity:
    """One wire level: what distorting it costs, and what carrying it costs."""

    index: int
    output_error: float  # induced by distorting this level alone
    spatial: int  # H*W of this level at the probed image size

    @property
    def value_per_channel(self) -> float:
        """Error avoided per unit of byte cost — what an allocation equalises.

        One latent channel at this level costs bytes in proportion to its
        spatial size, so a level that is both sensitive and small is where the
        budget belongs.
        """
        return self.output_error / self.spatial


def _coarsen(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Round a tensor to ``bits`` of precision, as any lossy codec would."""
    q = quantize(t)
    levels = 1 << bits
    step = 256 / levels
    coarse = torch.round((q.values.float() + 128) / step) * step - 128
    return dequantize(
        type(q)(
            values=coarse.clamp(-128, 127).to(torch.int8),
            scale=q.scale,
            zero_point=q.zero_point,
            axis=q.axis,
            orig_dtype=q.orig_dtype,
        )
    )


@torch.no_grad()
def level_sensitivity(
    runner: SplitRunner,
    frames: torch.Tensor,
    bits: int = 3,
) -> list[LevelSensitivity]:
    """Output error induced by coarsening each wire level on its own.

    ``bits`` sets how hard each level is squeezed. The absolute numbers move
    with it; the *ranking* between levels — which is all an allocation needs —
    is stable, because it reflects how much the tail depends on each level
    rather than how any particular codec behaves.
    """
    wire = runner.edge(frames)
    baseline = _flat(primary_output(runner.cloud(wire)))
    out = []
    for index, tensor in sorted(wire.items()):
        distorted = dict(wire)
        distorted[index] = _coarsen(tensor, bits)
        through = _flat(primary_output(runner.cloud(distorted)))
        error = ((through - baseline).norm() / (baseline.norm() + 1e-8)).item()
        height, width = tensor.shape[-2:]
        out.append(LevelSensitivity(index=index, output_error=error, spatial=height * width))
    return out


def _flat(out: object) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out.flatten()
    if isinstance(out, (list, tuple)):
        return torch.cat([_flat(item) for item in out])
    raise TypeError(f"cannot flatten a {type(out).__name__} model output")


def propose_allocation(
    sensitivity: list[LevelSensitivity],
    budget_channels: int,
    minimum: int = 2,
) -> dict[int, int]:
    """Latent widths that spend ``budget_channels`` where they buy the most.

    A channel at level *i* costs bytes in proportion to that level's spatial
    size, and buys error in proportion to its sensitivity. Equalising the ratio
    across levels means widths proportional to ``output_error / spatial`` —
    the deep levels win, which is why the uniform default underserves them.

    ``budget_channels`` is expressed as the uniform width it replaces, so
    ``propose_allocation(s, 8)`` costs roughly what ``--latent-channels 8``
    costs and merely spends it differently. ``minimum`` keeps every level
    representable: a level squeezed to zero channels is a level deleted.
    """
    if budget_channels < minimum:
        raise ValueError(f"budget of {budget_channels} channels is below the {minimum} floor")
    # width_i proportional to error_i / spatial_i, scaled so the total latent
    # cost equals what the uniform width would have shipped:
    #   sum_i width_i * spatial_i  ==  budget * sum_i spatial_i
    total_error = sum(s.output_error for s in sensitivity)
    total_spatial = sum(s.spatial for s in sensitivity)
    if total_error <= 0:  # pragma: no cover - output independent of its own features
        return {s.index: budget_channels for s in sensitivity}

    k = budget_channels * total_spatial / total_error
    return {s.index: max(minimum, math.floor(k * s.value_per_channel)) for s in sensitivity}


def allocation_cost(
    allocation: dict[int, int], sensitivity: list[LevelSensitivity], stride: int = 2
) -> int:
    """Latent elements per frame — the axis an allocation is budgeted against."""
    spatial = {s.index: s.spatial for s in sensitivity}
    return sum(width * spatial[i] // (stride * stride) for i, width in allocation.items())
