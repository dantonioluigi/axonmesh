from __future__ import annotations

import socket
import struct

import pytest
import torch

from yolosplit.protocol import (
    MAGIC,
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
    from yolosplit.protocol import ConnectionClosed

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
