"""Wire simulations for the edge→cloud link.

Each transport takes the wire set (layer index → tensor), returns the tensors
as the cloud would reconstruct them, and the number of bytes that crossed the
wire. Lossy transports (INT8) round-trip through quantisation so downstream
accuracy is measured end-to-end on what the cloud actually receives.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch

from .quantize import QuantizedTensor, dequantize, quantize


@dataclass(frozen=True)
class RawTransport:
    """Ship tensors as-is at the given dtype. The bandwidth baseline.

    ``dtype=torch.float16`` halves the payload with (usually) negligible loss;
    ``torch.float32`` is lossless.
    """

    dtype: torch.dtype = torch.float32

    def __call__(self, wire: dict[int, torch.Tensor]) -> tuple[dict[int, torch.Tensor], int]:
        received = {}
        nbytes = 0
        for i, t in wire.items():
            shipped = t.to(self.dtype)
            nbytes += shipped.numel() * shipped.element_size()
            received[i] = shipped.to(t.dtype)
        return received, nbytes


@dataclass(frozen=True)
class Int8Transport:
    """Affine INT8 quantisation, optionally zlib-compressed.

    Args:
        axis: channel axis for per-channel quantisation; ``None`` for per-tensor.
        compress: also zlib-compress the serialised payload (lossless on top of
            INT8); the byte count then reflects the compressed size.
        level: zlib compression level (only used when ``compress=True``).
    """

    axis: int | None = None
    compress: bool = False
    level: int = 6

    def __call__(self, wire: dict[int, torch.Tensor]) -> tuple[dict[int, torch.Tensor], int]:
        received = {}
        nbytes = 0
        for i, t in wire.items():
            payload = quantize(t, axis=self.axis).to_bytes()
            nbytes += len(zlib.compress(payload, self.level)) if self.compress else len(payload)
            received[i] = dequantize(QuantizedTensor.from_bytes(payload))
        return received, nbytes
