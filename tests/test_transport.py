from __future__ import annotations

import pytest
import torch

from yolosplit.transport import Int8Transport, RawTransport


@pytest.fixture()
def wire():
    torch.manual_seed(15)
    return {
        4: torch.nn.functional.silu(torch.randn(1, 8, 20, 20)),
        6: torch.nn.functional.silu(torch.randn(1, 16, 10, 10)),
        10: torch.nn.functional.silu(torch.randn(1, 32, 5, 5)),
    }


def _numel(wire: dict[int, torch.Tensor]) -> int:
    return sum(t.numel() for t in wire.values())


def test_raw_fp32_is_lossless_and_counts_4_bytes(wire):
    received, nbytes = RawTransport(torch.float32)(wire)
    assert nbytes == 4 * _numel(wire)
    for i in wire:
        torch.testing.assert_close(received[i], wire[i])


def test_raw_fp16_halves_bytes_with_small_error(wire):
    received, nbytes = RawTransport(torch.float16)(wire)
    assert nbytes == 2 * _numel(wire)
    for i in wire:
        assert received[i].dtype == wire[i].dtype
        torch.testing.assert_close(received[i], wire[i], rtol=1e-2, atol=1e-3)


def test_int8_quarters_bytes_with_bounded_error(wire):
    received, nbytes = Int8Transport()(wire)
    assert nbytes < 0.3 * 4 * _numel(wire)
    assert set(received) == set(wire)
    for i in wire:
        span = wire[i].max() - wire[i].min()
        err = (received[i] - wire[i]).abs().max()
        assert err <= span / 255 + 1e-6


def test_int8_per_channel_not_worse_than_per_tensor(wire):
    per_tensor, _ = Int8Transport(axis=None)(wire)
    per_channel, _ = Int8Transport(axis=1)(wire)
    for i in wire:
        err_pt = (per_tensor[i] - wire[i]).abs().mean()
        err_pc = (per_channel[i] - wire[i]).abs().mean()
        assert err_pc <= err_pt * 1.05


def test_zlib_compression_never_explodes(wire):
    plain_received, plain = Int8Transport()(wire)
    received, compressed = Int8Transport(compress=True)(wire)
    assert compressed <= plain * 1.01 + 128
    for i in wire:  # compression is lossless on top of INT8
        torch.testing.assert_close(received[i], plain_received[i])
