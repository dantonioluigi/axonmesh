from __future__ import annotations

import json

import pytest

from yolosplit.cli import build_parser, main

MODEL = "yolo11n.yaml"


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "yolosplit" in capsys.readouterr().out


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


@pytest.mark.parametrize("transport", ["int8", "fp16", "fp32"])
def test_build_transport_variants(transport):
    from yolosplit.cli import _build_transport
    from yolosplit.transport import Int8Transport, RawTransport

    args = build_parser().parse_args(
        ["evaluate", "--model", "w.pt", "--data", "d.yaml", "--transport", transport]
    )
    built = _build_transport(args)
    expected = Int8Transport if transport == "int8" else RawTransport
    assert isinstance(built, expected)
