from __future__ import annotations

import socket
import struct
import zlib

import pytest
import torch

from axonmesh.protocol import (
    MAGIC,
    MAX_PAYLOAD,
    MAX_TENSOR_BYTES,
    PROTOCOL_VERSION,
    Handshake,
    Kind,
    ProtocolError,
    module_fingerprint,
    pack_tensors,
    recv_message,
    send_message,
    unpack_tensors,
)


@pytest.fixture()
def pair():
    a, b = socket.socketpair()
    yield a, b
    a.close()
    b.close()


def test_message_round_trip(pair):
    a, b = pair
    sent = send_message(a, Kind.RESULT, b"payload", frame_id=42)
    kind, frame_id, payload = recv_message(b)
    assert (kind, frame_id, payload) == (Kind.RESULT, 42, b"payload")
    assert sent == 17 + len(b"payload")  # header is 17 bytes


def test_bad_magic_is_rejected(pair):
    a, b = pair
    a.sendall(struct.pack("<3sBBQI", b"XXX", 1, 1, 0, 0))
    with pytest.raises(ProtocolError, match="magic"):
        recv_message(b)


def test_wrong_version_is_rejected(pair):
    a, b = pair
    a.sendall(struct.pack("<3sBBQI", MAGIC, 99, 1, 0, 0))
    with pytest.raises(ProtocolError, match="version"):
        recv_message(b)


def test_peer_close_raises_connection_closed(pair):
    from axonmesh.protocol import ConnectionClosed

    a, b = pair
    a.close()
    with pytest.raises(ConnectionClosed):
        recv_message(b)


@pytest.mark.parametrize("compress", [False, True])
def test_tensor_pack_round_trip(compress):
    torch.manual_seed(15)
    tensors = {
        4: torch.nn.functional.silu(torch.randn(1, 8, 16, 16)),
        10: torch.nn.functional.silu(torch.randn(1, 16, 4, 4)),
    }
    restored = unpack_tensors(pack_tensors(tensors, compress=compress))
    assert set(restored) == {4, 10}
    for i in tensors:
        span = tensors[i].max() - tensors[i].min()
        assert (restored[i] - tensors[i]).abs().max() <= span / 255 + 1e-6


def test_fingerprint_distinguishes_weights():
    torch.manual_seed(15)
    a = torch.nn.Conv2d(3, 4, 3)
    torch.manual_seed(15)
    b = torch.nn.Conv2d(3, 4, 3)
    torch.manual_seed(16)
    c = torch.nn.Conv2d(3, 4, 3)
    assert module_fingerprint(a) == module_fingerprint(b)
    assert module_fingerprint(a) != module_fingerprint(c)
    assert len(module_fingerprint(a)) == 16


def test_handshake_round_trip_and_mismatches():
    ours = Handshake(model="abc", bottleneck=None, cut=10, imgsz=640)
    same = Handshake.from_bytes(ours.to_bytes())
    assert ours.mismatches(same) == []

    theirs = Handshake(model="xyz", bottleneck="b1", cut=8, imgsz=640)
    assert ours.mismatches(theirs) == ["model", "bottleneck", "cut"]


class _Replay:
    """A socket that hands back a fixed byte string, then EOF."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def recv(self, n: int) -> bytes:
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


def _features(*, index=4, compressed=0, length=None, body=b"") -> bytes:
    """A FEATURES payload declaring one tensor, with the lengths as given."""
    length = len(body) if length is None else length
    return struct.pack("<H", 1) + struct.pack("<HBI", index, compressed, length) + body


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("empty", b""),
        ("truncated count", b"\x01"),
        ("header cut short", struct.pack("<H", 1) + b"\x00\x00"),
        ("length longer than the payload", _features(length=9999, body=b"abc")),
        ("zlib flag over non-zlib bytes", _features(compressed=1, body=b"\x00\x00\x00\x00")),
        ("valid zlib, not a tensor", _features(compressed=1, body=zlib.compress(b"junk"))),
    ],
)
def test_hostile_features_payloads_are_protocol_errors(name, payload):
    """Malformed input must never reach the caller as struct/zlib/ValueError.

    The server's connection loop separates "this peer is talking nonsense"
    from "we have a bug" purely by exception type: anything that leaks out as
    struct.error is counted as neither, and the error metric under-reports
    exactly when it matters.
    """
    with pytest.raises(ProtocolError):
        unpack_tensors(payload)


def test_a_compression_bomb_is_refused_not_allocated():
    """400 KB on the wire must not become 400 MB of resident memory."""
    bomb = zlib.compress(b"\x00" * (8 * MAX_TENSOR_BYTES), 9)
    assert len(bomb) < MAX_PAYLOAD  # the framing cap alone lets this through
    with pytest.raises(ProtocolError, match="expand past"):
        unpack_tensors(_features(compressed=1, body=bomb))


def test_unknown_message_kind_is_a_protocol_error():
    frame = struct.pack("<3sBBQI", MAGIC, PROTOCOL_VERSION, 99, 0, 0)
    with pytest.raises(ProtocolError, match="unknown message kind 99"):
        recv_message(_Replay(frame))


@pytest.mark.parametrize(
    "payload", [b"{{{", b"not json at all", b'{"unexpected": 1}', b"[1, 2, 3]", b"\xff\xfe"]
)
def test_malformed_handshakes_are_protocol_errors(payload):
    """This parses before the fingerprint check — i.e. on wholly unvetted input."""
    with pytest.raises(ProtocolError):
        Handshake.from_bytes(payload)
