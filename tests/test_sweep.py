from __future__ import annotations

import json

import pytest

from yolosplit.sweep import (
    SweepConfig,
    SweepResult,
    mark_pareto,
    run_sweep,
    to_dicts,
    to_markdown,
)


@pytest.fixture()
def results(det_model, images_dir):
    return run_sweep(
        det_model,
        images_dir,
        latents=[2, 4],
        strides=[1],
        epochs=1,
        batch=3,
        imgsz=160,
    )


def result(latent, stride, zlib_bytes, err) -> SweepResult:
    return SweepResult(
        config=SweepConfig(latent, stride),
        int8_bytes=zlib_bytes * 2,
        int8_zlib_bytes=zlib_bytes,
        jpeg_bytes=10_000,
        relative_error=err,
        epoch_losses=[1.0],
    )


def test_sweep_trains_every_config(results):
    assert [r.config for r in results] == [SweepConfig(2, 1), SweepConfig(4, 1)]
    for r in results:
        assert r.int8_bytes > r.int8_zlib_bytes / 2  # zlib can't beat entropy that hard
        assert r.jpeg_bytes > 0
        assert r.relative_error > 0
        assert len(r.epoch_losses) == 1


def test_smaller_latent_ships_fewer_bytes(results):
    by_latent = {r.config.latent_channels: r.int8_bytes for r in results}
    assert by_latent[2] < by_latent[4]


def test_indivisible_stride_is_skipped_not_fatal(det_model, images_dir, capsys):
    # At imgsz 160 the P5 map is 5x5: stride 3 divides nothing, stride 1 works.
    results = run_sweep(
        det_model, images_dir, latents=[2], strides=[3, 1], epochs=1, batch=3, imgsz=160
    )
    assert [r.config.stride for r in results] == [1]
    assert "skip latent=2 stride=3" in capsys.readouterr().out


def test_pareto_marking():
    a = result(2, 1, zlib_bytes=1000, err=0.5)  # smallest wire
    b = result(4, 1, zlib_bytes=2000, err=0.2)  # best quality
    c = result(8, 1, zlib_bytes=2500, err=0.3)  # dominated by b
    mark_pareto([a, b, c])
    assert a.pareto and b.pareto and not c.pareto


def test_markdown_and_dicts(results):
    table = to_markdown(results)
    assert table.splitlines()[0].startswith("| latent |")
    assert len(table.splitlines()) == 2 + len(results)

    dicts = to_dicts(results)
    assert dicts[0]["latent_channels"] in (2, 4)
    assert all(set(d) >= {"int8_zlib_bytes", "vs_jpeg", "pareto"} for d in dicts)


def test_sweep_cli(capsys, images_dir, tmp_path):
    from yolosplit.cli import main

    report = tmp_path / "sweep.json"
    code = main(
        [
            "sweep",
            "--model",
            "yolo11n.yaml",
            "--images",
            str(images_dir),
            "--imgsz",
            "160",
            "--latents",
            "2",
            "--strides",
            "1",
            "--epochs",
            "1",
            "--batch",
            "3",
            "--json",
            str(report),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "smallest pareto config" in out
    assert "evaluate --bottleneck" in out
    payload = json.loads(report.read_text())
    assert len(payload) == 1
    assert payload[0]["pareto"] is True


def test_sweep_cli_fails_when_everything_skipped(capsys, images_dir):
    from yolosplit.cli import main

    code = main(
        [
            "sweep",
            "--model",
            "yolo11n.yaml",
            "--images",
            str(images_dir),
            "--imgsz",
            "160",
            "--latents",
            "2",
            "--strides",
            "3",
            "--epochs",
            "1",
        ]
    )
    assert code == 1
    assert "no configuration completed" in capsys.readouterr().out
