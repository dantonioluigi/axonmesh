from __future__ import annotations

import json

import pytest

from splitflow.cli import build_parser, main

MODEL = "yolo11n.yaml"


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "splitflow" in capsys.readouterr().out


def test_requires_a_command():
    with pytest.raises(SystemExit):
        main([])


def test_inspect_prints_layers_and_cuts(capsys):
    assert main(["inspect", "--model", MODEL, "--imgsz", "160"]) == 0
    out = capsys.readouterr().out
    assert "C2PSA" in out
    assert " 10* " in out  # backbone cut marked
    assert "Detect" in out


def test_measure_prints_table_and_writes_json(capsys, images_dir, tmp_path):
    report = tmp_path / "summary.json"
    code = main(
        [
            "measure",
            "--model",
            MODEL,
            "--images",
            str(images_dir),
            "--imgsz",
            "160",
            "--quality",
            "80",
            "--json",
            str(report),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| frame |" in out
    assert "**mean**" in out

    summary = json.loads(report.read_text())
    assert summary["frames"] == 3
    assert summary["jpeg_quality"] == 80
    assert summary["cut"] == 10
    assert summary["mean_wire_int8"] > 0


def test_measure_accepts_explicit_cut(capsys, images_dir):
    assert (
        main(
            [
                "measure",
                "--model",
                MODEL,
                "--images",
                str(images_dir),
                "--imgsz",
                "160",
                "--cut",
                "4",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert "cut=4" in capsys.readouterr().out


def test_evaluate_parser_wires_transport_options():
    args = build_parser().parse_args(
        [
            "evaluate",
            "--model",
            "w.pt",
            "--data",
            "d.yaml",
            "--transport",
            "int8",
            "--per-channel",
            "--compress",
        ]
    )
    assert args.transport == "int8"
    assert args.per_channel and args.compress
    assert args.func is not None
    # rect defaults off so a stride>1 bottleneck (square inputs) evaluates cleanly.
    assert args.rect is False
    assert (
        build_parser()
        .parse_args(["evaluate", "--model", "w.pt", "--data", "d.yaml", "--rect"])
        .rect
        is True
    )


@pytest.mark.parametrize("transport", ["int8", "fp16", "fp32"])
def test_build_transport_variants(transport):
    from splitflow.cli import _build_transport
    from splitflow.transport import Int8Transport, RawTransport

    args = build_parser().parse_args(
        ["evaluate", "--model", "w.pt", "--data", "d.yaml", "--transport", transport]
    )
    built = _build_transport(args)
    expected = Int8Transport if transport == "int8" else RawTransport
    assert isinstance(built, expected)


@pytest.fixture()
def bottleneck_ckpt(tmp_path, images_dir):
    path = tmp_path / "bn.pt"
    code = main(
        [
            "train-bottleneck",
            "--model",
            MODEL,
            "--images",
            str(images_dir),
            "--imgsz",
            "160",
            "--latent-channels",
            "4",
            "--stride",
            "1",
            "--epochs",
            "1",
            "--batch",
            "3",
            "--out",
            str(path),
        ]
    )
    assert code == 0
    return path


def test_train_bottleneck_writes_checkpoint(bottleneck_ckpt):
    from splitflow.bottleneck import load_bottleneck

    bottleneck = load_bottleneck(bottleneck_ckpt)
    assert bottleneck.config["latent_channels"] == 4
    assert bottleneck.config["stride"] == 1


def test_build_transport_prefers_bottleneck(bottleneck_ckpt):
    from splitflow.bottleneck import BottleneckTransport
    from splitflow.cli import _build_transport

    args = build_parser().parse_args(
        ["evaluate", "--model", "w.pt", "--data", "d.yaml", "--bottleneck", str(bottleneck_ckpt)]
    )
    assert isinstance(_build_transport(args), BottleneckTransport)


def test_stream_reports_modes_and_savings(capsys, images_dir, tmp_path):
    report = tmp_path / "stream.json"
    code = main(
        [
            "stream",
            "--model",
            MODEL,
            "--images",
            str(images_dir),
            "--imgsz",
            "160",
            "--json",
            str(report),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "modes:" in out
    assert "always-JPEG" in out
    assert "retraining candidates" in out

    summary = json.loads(report.read_text())
    assert summary["frames"] == 3
    assert summary["frames_detections"] + summary["frames_features"] + summary["frames_frame"] == 3


def test_stream_with_bottleneck_checkpoint(capsys, images_dir, bottleneck_ckpt):
    code = main(
        [
            "stream",
            "--model",
            MODEL,
            "--images",
            str(images_dir),
            "--imgsz",
            "160",
            "--bottleneck",
            str(bottleneck_ckpt),
            "--limit",
            "2",
        ]
    )
    assert code == 0
    assert "adaptive total" in capsys.readouterr().out
