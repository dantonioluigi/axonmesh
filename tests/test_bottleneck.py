from __future__ import annotations

import pytest
import torch

from axonmesh.bottleneck import (
    Bottleneck,
    BottleneckTransport,
    LevelCodec,
    load_bottleneck,
    resolve_latents,
    save_bottleneck,
)
from axonmesh.split import SplitRunner, raw_nbytes
from axonmesh.train import _tensors, normalize_device, output_error, train_bottleneck


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
        # Decoded tensors keep the wire's device (GPU-safe contract).
        assert received[i].device == wire[i].device
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


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("cpu", "cpu"),
        ("", "cpu"),
        ("0", "cuda:0"),
        ("1", "cuda:1"),
        ("0,1", "cuda:0"),  # single-GPU training: first index wins
        ("cuda", "cuda"),
        ("cuda:0", "cuda:0"),
        ("mps", "mps"),
    ],
)
def test_normalize_device(given, expected):
    assert normalize_device(given) == expected


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
        progress=False,
        task_weight=0.0,
    )
    assert len(result.epoch_losses) == 4
    assert result.epoch_losses[-1] < result.epoch_losses[0]
    assert set(result.relative_errors) == {4, 6, 10}
    assert all(err > 0 for err in result.relative_errors.values())


def test_training_progress_prints_per_epoch(det_model, images_dir, capsys):
    train_bottleneck(
        det_model,
        images_dir,
        latent_channels=4,
        stride=1,
        epochs=2,
        batch=3,
        imgsz=160,
        progress=True,
        task_weight=0.0,
    )
    out = capsys.readouterr().out
    assert "epoch 1/2: normalised MSE" in out
    assert "epoch 2/2: normalised MSE" in out


def test_training_progress_false_is_silent(det_model, images_dir, capsys):
    train_bottleneck(
        det_model,
        images_dir,
        latent_channels=4,
        stride=1,
        epochs=1,
        batch=3,
        imgsz=160,
        progress=False,
        task_weight=0.0,
    )
    assert capsys.readouterr().out == ""


def test_training_rejects_empty_dir(det_model, tmp_path):
    with pytest.raises(FileNotFoundError):
        train_bottleneck(det_model, tmp_path, epochs=1, imgsz=160)


def test_task_loss_steers_the_codec_elsewhere(det_model, images_dir):
    """The task objective must reach the codec, not just ride along.

    Same seed, same data, same budget — only the objective differs. If the
    gradient through the frozen tail did not arrive, the two runs would land on
    identical weights.
    """
    kwargs = dict(
        latent_channels=4, stride=1, epochs=2, batch=3, lr=1e-3, imgsz=160, progress=False
    )
    features_only, _ = train_bottleneck(det_model, images_dir, task_weight=0.0, **kwargs)
    task_aware, _ = train_bottleneck(det_model, images_dir, task_weight=1.0, **kwargs)

    a = features_only.state_dict()
    b = task_aware.state_dict()
    assert not any(torch.allclose(a[k], b[k]) for k in a)


def test_cloud_grad_reaches_the_wire_tensors(det_model, probe):
    """Without this the task loss is unexpressible: no graph, no gradient."""
    runner = SplitRunner(det_model)
    wire = {i: t.clone().requires_grad_(True) for i, t in runner.edge(probe).items()}

    assert _tensors(runner.cloud(wire))[0].grad_fn is None  # default stays no_grad
    out = _tensors(runner.cloud(wire, grad=True))[0]
    out.sum().backward()
    assert all(t.grad is not None for t in wire.values())


def test_tensors_walks_whatever_the_head_returns():
    a, b, c = torch.zeros(1), torch.ones(2), torch.full((3,), 2.0)
    assert _tensors(a) == [a]
    assert _tensors((a, [b, (c,)])) == [a, b, c]  # nested containers, in order
    assert _tensors({"pred": a, "aux": [b]}) == [a, b]
    assert _tensors(("not a tensor", 7, None)) == []


def test_output_error_ranks_codecs_by_what_the_model_sees(det_model, images_dir):
    runner = SplitRunner(det_model)
    paths = sorted(images_dir.iterdir())
    torch.manual_seed(15)
    channels = {i: t.shape[1] for i, t in runner.edge(torch.rand(1, 3, 160, 160)).items()}
    untrained = Bottleneck(channels, latent_channels=4, stride=1)
    trained, _ = train_bottleneck(
        det_model,
        images_dir,
        latent_channels=4,
        stride=1,
        epochs=3,
        batch=3,
        imgsz=160,
        progress=False,
    )
    before = output_error(runner, untrained, paths, imgsz=160, batch=3)
    after = output_error(runner, trained, paths, imgsz=160, batch=3)
    assert 0 < after < before


def test_quality_is_scored_on_frames_held_out_of_training(det_model, images_dir):
    """The reported number must not come from frames the codec fitted.

    A codec that memorised its training set scores perfectly on it, which is
    exactly the reading that makes an unfinished codec look finished.
    """
    _, result = train_bottleneck(
        det_model,
        images_dir,
        latent_channels=4,
        stride=1,
        epochs=1,
        batch=2,
        imgsz=160,
        progress=False,
        task_weight=0.0,
        val_fraction=0.34,  # 3 frames in the fixture -> 1 held out
    )
    assert result.val_images == 1
    assert result.train_images == 2
    assert result.output_error is not None and result.output_error > 0


def test_no_held_out_frames_means_no_quality_claim(det_model, images_dir):
    _, result = train_bottleneck(
        det_model,
        images_dir,
        latent_channels=4,
        stride=1,
        epochs=1,
        batch=2,
        imgsz=160,
        progress=False,
        task_weight=0.0,
        val_fraction=0.0,
    )
    assert result.val_images == 0
    assert result.output_error is None  # silence beats a number measured on training data


def test_uniform_latents_expand_to_every_level():
    assert resolve_latents({4: 64, 6: 128, 10: 256}, 8) == {4: 8, 6: 8, 10: 8}


def test_per_level_latents_are_taken_as_given():
    given = {4: 4, 6: 16, 10: 48}
    assert resolve_latents({4: 64, 6: 128, 10: 256}, given) == given


def test_a_level_left_out_of_the_allocation_is_an_error():
    """Silently defaulting the missing level would ship a codec nobody sized."""
    with pytest.raises(ValueError, match=r"\[10\]"):
        resolve_latents({4: 64, 6: 128, 10: 256}, {4: 4, 6: 16})


def test_per_level_allocation_survives_a_save_load_round_trip(tmp_path):
    torch.manual_seed(15)
    original = Bottleneck({4: 16, 6: 32}, latent_channels={4: 2, 6: 8}, stride=1)
    path = tmp_path / "alloc.pt"
    save_bottleneck(original, path)
    restored = load_bottleneck(path)

    assert restored.config["latent_channels"] == {4: 2, 6: 8}
    wire = {4: torch.randn(1, 16, 8, 8), 6: torch.randn(1, 32, 4, 4)}
    latents = restored.encode(wire)
    assert latents[4].shape[1] == 2
    assert latents[6].shape[1] == 8
