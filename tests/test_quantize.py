from __future__ import annotations

import pytest
import torch

from axonmesh.quantize import QuantizedTensor, dequantize, quantize


@pytest.fixture()
def activation():
    torch.manual_seed(15)
    # SiLU-like asymmetric range, per-channel spread like real feature maps.
    t = torch.randn(1, 8, 16, 16) * torch.linspace(0.1, 4.0, 8).reshape(1, 8, 1, 1)
    return torch.nn.functional.silu(t)


def test_round_trip_error_is_bounded_per_tensor(activation):
    q = quantize(activation)
    err = (dequantize(q) - activation).abs().max()
    assert err <= q.scale.max() * 0.5 + 1e-6


def test_round_trip_error_is_bounded_per_channel(activation):
    q = quantize(activation, axis=1)
    scale = q.scale.reshape(1, -1, 1, 1)
    err = (dequantize(q) - activation).abs()
    assert (err <= scale * 0.5 + 1e-6).all()


def test_per_channel_beats_per_tensor(activation):
    """With unevenly scaled channels, per-channel error must not be worse."""
    err_pt = (dequantize(quantize(activation)) - activation).abs().mean()
    err_pc = (dequantize(quantize(activation, axis=1)) - activation).abs().mean()
    assert err_pc <= err_pt


def test_zero_tensor_is_exact():
    t = torch.zeros(2, 3, 4, 4)
    torch.testing.assert_close(dequantize(quantize(t)), t)


def test_constant_tensor_stays_within_fallback_scale():
    t = torch.full((2, 3, 4, 4), 2.5)
    err = (dequantize(quantize(t)) - t).abs().max()
    assert err <= 0.5  # zero range falls back to scale=1


def test_values_are_int8_with_original_shape(activation):
    q = quantize(activation)
    assert q.values.dtype == torch.int8
    assert q.values.shape == activation.shape
    assert q.axis is None
    assert q.scale.shape == q.zero_point.shape == (1,)


def test_per_channel_has_one_scale_per_channel(activation):
    q = quantize(activation, axis=1)
    assert q.scale.shape == (activation.shape[1],)
    assert q.axis == 1


def test_float16_round_trips_to_float16(activation):
    q = quantize(activation.half())
    restored = dequantize(q)
    assert q.orig_dtype == torch.float16
    assert restored.dtype == torch.float16


def test_rejects_non_float_tensors():
    with pytest.raises(TypeError):
        quantize(torch.ones(4, dtype=torch.int32))


@pytest.mark.parametrize("axis", [None, 1])
def test_serialization_round_trip(activation, axis):
    q = quantize(activation, axis=axis)
    restored = QuantizedTensor.from_bytes(q.to_bytes())
    assert torch.equal(restored.values, q.values)
    torch.testing.assert_close(restored.scale, q.scale)
    assert torch.equal(restored.zero_point, q.zero_point)
    assert restored.axis == q.axis
    assert restored.orig_dtype == q.orig_dtype
    torch.testing.assert_close(dequantize(restored), dequantize(q))


def test_nbytes_matches_payload_and_is_near_one_byte_per_element(activation):
    q = quantize(activation)
    assert q.nbytes == len(q.to_bytes())
    overhead = q.nbytes - activation.numel()
    assert 0 < overhead < 64


def test_from_bytes_rejects_garbage():
    with pytest.raises(ValueError):
        QuantizedTensor.from_bytes(b"XXXX" + b"\x00" * 32)
