from __future__ import annotations

import pytest
import torch

from yolosplit.bottleneck import (
    Bottleneck,
    BottleneckTransport,
    LevelCodec,
    load_bottleneck,
    save_bottleneck,
)
from yolosplit.split import SplitRunner, raw_nbytes
from yolosplit.train import train_bottleneck


@pytest.fixture()
def wire():
    torch.manual_seed(15)
    return {
        4: torch.nn.functional.silu(torch.randn(1, 16, 20, 20)),
        6: torch.nn.functional.silu(torch.randn(1, 32, 10, 10)),
        10: torch.nn.functional.silu(torch.randn(1, 32, 6, 6)),
    }


@pytest.fixture()
def bottleneck(wire):
    torch.manual_seed(15)
    return Bottleneck({i: t.shape[1] for i, t in wire.items()}, latent_channels=4, stride=2)


@pytest.mark.parametrize("stride", [1, 2])
def test_level_codec_round_trips_shape(stride):
    torch.manual_seed(15)
    codec = LevelCodec(channels=16, latent_channels=4, stride=stride)
    t = torch.randn(2, 16, 8, 8)
    z = codec.encode(t)
    assert z.shape == (2, 4, 8 // stride, 8 // stride)
    assert codec.decode(z).shape == t.shape


def test_level_codec_rejects_indivisible_input():
    codec = LevelCodec(channels=8, latent_channels=4, stride=2)
    with pytest.raises(ValueError, match="not divisible"):
        codec.encode(torch.randn(1, 8, 5, 6))


def test_level_codec_rejects_bad_stride():
    with pytest.raises(ValueError):
        LevelCodec(channels=8, latent_channels=4, stride=0)


def test_bottleneck_reconstructs_shapes(bottleneck, wire):
    recon = bottleneck(wire)
    assert set(recon) == set(wire)
    for i in wire:
        assert recon[i].shape == wire[i].shape


def test_quant_noise_only_perturbs(bottleneck, wire):
    torch.manual_seed(15)
    clean = bottleneck(wire)
    noisy = bottleneck(wire, quant_noise=True)
    for i in wire:
        assert not torch.equal(clean[i], noisy[i])
        # Noise amplitude is one INT8 step: reconstructions stay close.
        assert (clean[i] - noisy[i]).abs().mean() < 0.1


def test_for_runner_matches_wire_channels(det_model):
    runner = SplitRunner(det_model)
    bottleneck = Bottleneck.for_runner(runner, latent_channels=4, stride=2, imgsz=160)
    assert set(bottleneck.codecs.keys()) == {str(i) for i in runner.wire}


def test_transport_shrinks_bytes_well_below_int8(bottleneck, wire):
    received, nbytes = BottleneckTransport(bottleneck)(wire)
    raw = raw_nbytes(wire)
    assert set(received) == set(wire)
    for i in wire:
        assert received[i].shape == wire[i].shape
    # latent has 4 channels vs 16/32 and 1/4 the spatial area -> far below raw INT8.
    assert nbytes < raw / 16


def test_transport_in_split_runner(det_model, probe):
    runner = SplitRunner(det_model)
    bottleneck = Bottleneck.for_runner(runner, latent_channels=4, stride=1, imgsz=160)
    baseline = runner(probe)[0]

    bn_runner = SplitRunner(det_model, transport=BottleneckTransport(bottleneck))
    out = bn_runner(probe)[0]
    assert out.shape == baseline.shape
    assert torch.isfinite(out).all()
    assert bn_runner.stats.total_bytes < runner.stats.total_bytes / 16


def test_save_load_round_trip(tmp_path, bottleneck, wire):
    path = tmp_path / "bn.pt"
    save_bottleneck(bottleneck, path)
    restored = load_bottleneck(path)
    assert restored.config == bottleneck.config
    with torch.no_grad():
        a = bottleneck.eval()(wire)
        b = restored(wire)
    for i in wire:
        torch.testing.assert_close(a[i], b[i])


def test_training_reduces_loss(det_model, images_dir):
    _, result = train_bottleneck(
        det_model,
        images_dir,
        latent_channels=4,
        stride=1,
        epochs=4,
        batch=3,
        lr=1e-3,
        imgsz=160,
    )
    assert len(result.epoch_losses) == 4
    assert result.epoch_losses[-1] < result.epoch_losses[0]
    assert set(result.relative_errors) == {4, 6, 10}
    assert all(err > 0 for err in result.relative_errors.values())


def test_training_rejects_empty_dir(det_model, tmp_path):
    with pytest.raises(FileNotFoundError):
        train_bottleneck(det_model, tmp_path, epochs=1, imgsz=160)
