"""Affine INT8 quantisation and wire serialisation for intermediate tensors.

Feature maps after SiLU activations are asymmetric, so affine (scale +
zero-point) quantisation is used rather than symmetric. Per-tensor and
per-channel (``axis=1``) modes are supported. ``to_bytes``/``from_bytes``
produce the actual wire payload, so measured sizes include the scale/zero-point
overhead, not just the raw INT8 buffer.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import torch

_MAGIC = b"YSQ0"
_QMIN, _QMAX = -128, 127
_DTYPE_CODES = {torch.float32: 0, torch.float16: 1}
_CODE_DTYPES = {v: k for k, v in _DTYPE_CODES.items()}


@dataclass(frozen=True)
class QuantizedTensor:
    """An INT8 tensor plus everything needed to reconstruct the original."""

    values: torch.Tensor  # int8, same shape as the original
    scale: torch.Tensor  # float32; scalar (per-tensor) or [C] (per-channel)
    zero_point: torch.Tensor  # int32; same shape as scale
    axis: int | None  # channel axis for per-channel mode, None for per-tensor
    orig_dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        """Exact wire payload size in bytes."""
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialise to the wire format (header + scales + zero-points + INT8 data)."""
        shape = tuple(self.values.shape)
        scales = self.scale.reshape(-1).numpy().astype("<f4")
        zps = self.zero_point.reshape(-1).numpy().astype("<i4")
        header = struct.pack(
            f"<4sBbB{len(shape)}II",
            _MAGIC,
            _DTYPE_CODES[self.orig_dtype],
            -1 if self.axis is None else self.axis,
            len(shape),
            *shape,
            len(scales),
        )
        return header + scales.tobytes() + zps.tobytes() + self.values.numpy().tobytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> QuantizedTensor:
        """Deserialise a payload produced by :meth:`to_bytes`."""
        magic, dtype_code, axis, ndim = struct.unpack_from("<4sBbB", payload, 0)
        if magic != _MAGIC:
            raise ValueError("not a QuantizedTensor payload")
        offset = struct.calcsize("<4sBbB")
        shape = struct.unpack_from(f"<{ndim}I", payload, offset)
        offset += 4 * ndim
        (n_scales,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        scales = np.frombuffer(payload, dtype="<f4", count=n_scales, offset=offset)
        offset += 4 * n_scales
        zps = np.frombuffer(payload, dtype="<i4", count=n_scales, offset=offset)
        offset += 4 * n_scales
        values = np.frombuffer(payload, dtype=np.int8, offset=offset).reshape(shape)
        return cls(
            values=torch.from_numpy(values.copy()),
            scale=torch.from_numpy(scales.copy()),
            zero_point=torch.from_numpy(zps.copy()),
            axis=None if axis == -1 else axis,
            orig_dtype=_CODE_DTYPES[dtype_code],
        )


def _broadcast_shape(t: torch.Tensor, axis: int) -> list[int]:
    return [t.shape[axis] if d == axis else 1 for d in range(t.ndim)]


def quantize(t: torch.Tensor, axis: int | None = None) -> QuantizedTensor:
    """Affine-quantise a float tensor to INT8.

    Args:
        t: float32/float16 tensor.
        axis: channel axis for per-channel quantisation; ``None`` for per-tensor.
    """
    if t.dtype not in _DTYPE_CODES:
        raise TypeError(f"expected a float16/float32 tensor, got {t.dtype}")
    work = t.detach().float()
    if axis is None:
        t_min, t_max = work.min(), work.max()
    else:
        dims = [d for d in range(work.ndim) if d != axis]
        t_min, t_max = work.amin(dim=dims), work.amax(dim=dims)
    scale = (t_max - t_min) / (_QMAX - _QMIN)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zero_point = torch.round(_QMIN - t_min / scale).to(torch.int32)
    if axis is not None:
        scale_b = scale.reshape(_broadcast_shape(work, axis))
        zp_b = zero_point.reshape(_broadcast_shape(work, axis))
    else:
        scale_b, zp_b = scale, zero_point
    values = torch.clamp(torch.round(work / scale_b) + zp_b, _QMIN, _QMAX).to(torch.int8)
    return QuantizedTensor(
        values=values,
        scale=scale.reshape(-1),
        zero_point=zero_point.reshape(-1),
        axis=axis,
        orig_dtype=t.dtype,
    )


def dequantize(q: QuantizedTensor) -> torch.Tensor:
    """Reconstruct the float tensor from its INT8 form."""
    if q.axis is not None:
        scale = q.scale.reshape(_broadcast_shape(q.values, q.axis))
        zp = q.zero_point.reshape(_broadcast_shape(q.values, q.axis))
    else:
        scale, zp = q.scale, q.zero_point
    return ((q.values.to(torch.float32) - zp) * scale).to(q.orig_dtype)
