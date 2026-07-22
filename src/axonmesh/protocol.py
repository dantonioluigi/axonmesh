"""Wire protocol v2 for the edge→cloud link (task-agnostic framing).

Framing, little-endian: magic ``YSP``, version u8, kind u8, frame id u64,
payload length u32, payload. ``FEATURES`` carries quantised tensors keyed by
layer index; ``FRAME`` a JPEG; ``DETECTIONS``/``RESULT`` carry *opaque* result
bytes — the built-in detection codec is just the default, the protocol itself
does not know what a detection is. That is the seam that keeps the wire usable
for other tasks and architectures.

The HELLO/ACK handshake exchanges model and bottleneck fingerprints, the cut
point, and the *role* the two ends are in. The silent nightmare of split
computing is two halves running different weights and producing confidently
wrong results: under ``Role.SPLIT`` the fingerprints turn that into a loud
connect-time failure. Under ``Role.CASCADE`` the two models are meant to
differ, and only the role, protocol and image size have to agree.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum

import torch
import torch.nn as nn

from .quantize import QuantizedTensor, dequantize, quantize

MAGIC = b"YSP"
#: Bumped to 2 when the handshake gained ``role``: a v1 peer sends no role and
#: a v1 server would ignore one, so the two would agree to run a cascade as if
#: it were a split and reject each other's weights. Cheaper to fail at connect.
PROTOCOL_VERSION = 2
#: Sanity cap on a single payload (a fp32 backbone pyramid is ~17 MB at 640).
MAX_PAYLOAD = 1 << 29
#: Cap on the *decompressed* tensor bytes in one FEATURES frame, summed over
#: every tensor in it. MAX_PAYLOAD bounds what arrives on the wire, which says
#: nothing about what it inflates to: zlib will happily turn 400 KB into 400 MB.
#: 64 MB leaves room for an fp32 pyramid several times over and still refuses a
#: bomb long before the process is killed.
MAX_TENSOR_BYTES = 1 << 26

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
    """Receive one framed message as ``(kind, frame_id, payload)``.

    Everything past the socket is untrusted input, so every failure here is a
    :class:`ProtocolError`: the server's connection loop distinguishes "this
    peer is talking nonsense" (drop it, count it) from a bug on our side, and
    it can only do that if malformed traffic never surfaces as struct.error.
    """
    magic, version, kind, frame_id, length = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {version}, expected {PROTOCOL_VERSION}")
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"payload of {length} bytes exceeds the {MAX_PAYLOAD} cap")
    try:
        message_kind = Kind(kind)
    except ValueError as err:
        raise ProtocolError(f"unknown message kind {kind}") from err
    return message_kind, frame_id, _recv_exact(sock, length)


def _unpack_from(spec: struct.Struct, payload: bytes, offset: int, what: str) -> tuple:
    if len(payload) - offset < spec.size:
        raise ProtocolError(
            f"payload truncated: {what} needs {spec.size} bytes at offset {offset}, "
            f"{max(len(payload) - offset, 0)} available"
        )
    return spec.unpack_from(payload, offset)


def _decompress(data: bytes, budget: int) -> bytes:
    """Inflate a tensor payload, refusing anything that expands past ``budget``.

    ``zlib.decompress`` has no output bound, so a few hundred KB on the wire
    can ask for hundreds of MB of memory — and the framing cap only limits the
    *compressed* size. Inflating incrementally against a ceiling turns an
    out-of-memory kill into a rejected frame.
    """
    inflator = zlib.decompressobj()
    try:
        out = inflator.decompress(data, budget)
    except zlib.error as err:
        raise ProtocolError(f"corrupt compressed tensor: {err}") from err
    if inflator.unconsumed_tail:
        raise ProtocolError(
            f"compressed tensors expand past the {MAX_TENSOR_BYTES} byte cap for one frame"
        )
    return out


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
    """Reverse of :func:`pack_tensors`: dequantised tensors keyed by layer index.

    Every length in the payload is the peer's claim, not a fact: slicing past
    the end silently returns short bytes, so each one is checked before it is
    trusted.
    """
    (count,) = _unpack_from(_COUNT, payload, 0, "tensor count")
    offset = _COUNT.size
    budget = MAX_TENSOR_BYTES  # shared across the frame: 65535 tensors are declarable
    tensors: dict[int, torch.Tensor] = {}
    for n in range(count):
        index, compressed, length = _unpack_from(
            _TENSOR_HEADER, payload, offset, f"header of tensor {n}"
        )
        offset += _TENSOR_HEADER.size
        if len(payload) - offset < length:
            raise ProtocolError(f"tensor {n} claims {length} bytes, {len(payload) - offset} remain")
        data = payload[offset : offset + length]
        offset += length
        if compressed:
            data = _decompress(data, budget)
        budget -= len(data)
        if budget < 0:
            raise ProtocolError(
                f"tensors in this frame exceed the {MAX_TENSOR_BYTES} byte decompressed cap"
            )
        try:
            tensors[index] = dequantize(QuantizedTensor.from_bytes(data))
        except (ValueError, KeyError, struct.error) as err:
            raise ProtocolError(f"tensor {n} is not a valid quantised payload: {err}") from err
    return tensors


def module_fingerprint(module: nn.Module) -> str:
    """Deterministic 16-hex-digit digest of a module's weights."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


class Role(str, Enum):
    """What relationship the two ends are in — which decides what must match.

    ``SPLIT`` is two halves of one network: identical weights are mandatory,
    because halves on different weights produce confidently wrong output and
    nothing else would catch it. ``CASCADE`` is two *independent* models, a
    small one on the edge and a larger one consulted when it is unsure —
    there, differing weights are the entire point, and enforcing a match would
    reject the working configuration.
    """

    SPLIT = "split"
    CASCADE = "cascade"


@dataclass(frozen=True)
class Handshake:
    """What both ends must agree on before any frame flows."""

    model: str
    bottleneck: str | None
    cut: int
    imgsz: int
    protocol: int = PROTOCOL_VERSION
    role: str = Role.SPLIT.value

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_bytes(cls, payload: bytes) -> Handshake:
        """Parse a peer's HELLO. Anything unparseable is a protocol error.

        This runs before the fingerprint check, i.e. on traffic from a peer
        that has not been vetted at all — the one point where an unhandled
        exception type is least acceptable.
        """
        try:
            fields = json.loads(payload.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise ProtocolError(f"handshake is not valid JSON: {err}") from err
        if not isinstance(fields, dict):
            raise ProtocolError(f"handshake must be an object, got {type(fields).__name__}")
        try:
            return cls(**fields)
        except TypeError as err:
            raise ProtocolError(f"handshake fields do not match the protocol: {err}") from err

    def mismatches(self, other: Handshake) -> list[str]:
        """Field names on which the two sides disagree (empty = compatible).

        Which fields are checked depends on the role. Under ``CASCADE`` the
        weights, the cut and the bottleneck are all expected to differ — the
        cloud runs a different and larger model, and there is no split — so
        only the protocol, the role itself and the image size are compared.
        Comparing weights there would reject every working deployment.
        """
        if self.role != other.role:
            return ["role"]
        checked = ("protocol", "role", "imgsz")
        if self.role == Role.SPLIT.value:
            checked += ("model", "bottleneck", "cut")
        return [field for field in checked if getattr(self, field) != getattr(other, field)]
