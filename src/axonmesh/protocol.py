"""Wire protocol v1 for the edge→cloud link (task-agnostic framing).

Framing, little-endian: magic ``YSP``, version u8, kind u8, frame id u64,
payload length u32, payload. ``FEATURES`` carries quantised tensors keyed by
layer index; ``FRAME`` a JPEG; ``DETECTIONS``/``RESULT`` carry *opaque* result
bytes — the built-in detection codec is just the default, the protocol itself
does not know what a detection is. That is the seam that keeps the wire usable
for other tasks and architectures.

The HELLO/ACK handshake exchanges model and bottleneck fingerprints plus the
cut point. The silent nightmare of split computing is two halves running
different weights and producing confidently wrong results: fingerprints turn
that into a loud connect-time failure.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from enum import IntEnum

import torch
import torch.nn as nn

from .quantize import QuantizedTensor, dequantize, quantize

MAGIC = b"YSP"
PROTOCOL_VERSION = 1
#: Sanity cap on a single payload (a fp32 backbone pyramid is ~17 MB at 640).
MAX_PAYLOAD = 1 << 29

_HEADER = struct.Struct("<3sBBQI")
_TENSOR_HEADER = struct.Struct("<HBI")  # layer index, zlib flag, byte length
_COUNT = struct.Struct("<H")


class ProtocolError(RuntimeError):
    """Malformed traffic or a failed handshake."""


class ConnectionClosed(ProtocolError):
    """The peer closed the connection."""


class Kind(IntEnum):
    HELLO = 1
    ACK = 2
    ERROR = 3
    DETECTIONS = 4
    FEATURES = 5
    FRAME = 6
    RESULT = 7


def send_message(sock, kind: Kind, payload: bytes = b"", frame_id: int = 0) -> int:
    """Send one framed message; returns the bytes put on the wire."""
    header = _HEADER.pack(MAGIC, PROTOCOL_VERSION, kind, frame_id, len(payload))
    sock.sendall(header + payload)
    return len(header) + len(payload)


def _recv_exact(sock, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionClosed("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock) -> tuple[Kind, int, bytes]:
    """Receive one framed message as ``(kind, frame_id, payload)``."""
    magic, version, kind, frame_id, length = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {version}, expected {PROTOCOL_VERSION}")
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"payload of {length} bytes exceeds the {MAX_PAYLOAD} cap")
    return Kind(kind), frame_id, _recv_exact(sock, length)


def pack_tensors(
    tensors: dict[int, torch.Tensor], axis: int | None = None, compress: bool = True
) -> bytes:
    """Quantise tensors to INT8 and pack them into a FEATURES payload."""
    parts = [_COUNT.pack(len(tensors))]
    for index in sorted(tensors):
        data = quantize(tensors[index], axis=axis).to_bytes()
        if compress:
            data = zlib.compress(data, 6)
        parts.append(_TENSOR_HEADER.pack(index, int(compress), len(data)))
        parts.append(data)
    return b"".join(parts)


def unpack_tensors(payload: bytes) -> dict[int, torch.Tensor]:
    """Reverse of :func:`pack_tensors`: dequantised tensors keyed by layer index."""
    (count,) = _COUNT.unpack_from(payload, 0)
    offset = _COUNT.size
    tensors: dict[int, torch.Tensor] = {}
    for _ in range(count):
        index, compressed, length = _TENSOR_HEADER.unpack_from(payload, offset)
        offset += _TENSOR_HEADER.size
        data = payload[offset : offset + length]
        offset += length
        if compressed:
            data = zlib.decompress(data)
        tensors[index] = dequantize(QuantizedTensor.from_bytes(data))
    return tensors


def module_fingerprint(module: nn.Module) -> str:
    """Deterministic 16-hex-digit digest of a module's weights."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Handshake:
    """What both halves must agree on before any frame flows."""

    model: str
    bottleneck: str | None
    cut: int
    imgsz: int
    protocol: int = PROTOCOL_VERSION

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_bytes(cls, payload: bytes) -> Handshake:
        return cls(**json.loads(payload.decode()))

    def mismatches(self, other: Handshake) -> list[str]:
        """Field names on which the two sides disagree (empty = compatible)."""
        return [
            field
            for field in ("protocol", "model", "bottleneck", "cut", "imgsz")
            if getattr(self, field) != getattr(other, field)
        ]
