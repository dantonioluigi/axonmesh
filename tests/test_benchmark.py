from __future__ import annotations

import json

import pytest

from splitflow.benchmark import (
    BenchmarkResult,
    StageTimings,
    benchmark_directory,
    benchmark_split,
    read_jetson_power,
    to_json,
)
from splitflow.stream import iter_image_frames
from splitflow.transport import Int8Transport


class TestStageTimings:
    def test_total_and_fps(self):
        t = StageTimings(prep_ms=1.0, edge_ms=4.0, wire_ms=3.0, cloud_ms=2.0)
        assert t.total_ms == 10.0
        assert t.fps == pytest.approx(100.0)

    def test_zero_total_gives_zero_fps(self):
        assert StageTimings(0.0, 0.0, 0.0, 0.0).fps == 0.0


class TestResultDerivedMetrics:
    def _result(self, **kw):
        base = dict(
            frames=5,
            device="cpu",
            cut=10,
            timings=StageTimings(1.0, 4.0, 3.0, 2.0),  # 10 ms -> 100 FPS
            wire_bytes=10_000.0,
            jpeg_bytes=50_000.0,
        )
        return BenchmarkResult(**(base | kw))

    def test_vs_jpeg_and_bandwidth(self):
        r = self._result()
        assert r.vs_jpeg == pytest.approx(5.0)
        # 10 kB * 8 bits * 100 fps = 8 Mbps
        assert r.bandwidth_mbps == pytest.approx(8.0)

    def test_delta_map_needs_both_numbers(self):
        assert self._result().delta_map50_95 is None
        r = self._result(map50_95=0.48, baseline_map50_95=0.50)
        assert r.delta_map50_95 == pytest.approx(-0.02)

    def test_markdown_and_json_shape(self):
        r = self._result(map50_95=0.48, baseline_map50_95=0.50, power_w=12.5)
        md = r.to_markdown()
        assert md.startswith("| metric | value |")
        for label in ("latency total", "throughput", "wire vs JPEG", "power", "mAP50-95 delta"):
            assert label in md
        d = json.loads(to_json(r))
        assert d["fps"] == pytest.approx(100.0)
        assert d["delta_map50_95"] == pytest.approx(-0.02)

    def test_missing_optional_metrics_render_as_na(self):
        md = self._result().to_markdown()
        assert "n/a" in md


def test_benchmark_measures_every_stage(det_model, images_dir):
    result = benchmark_directory(det_model, images_dir, imgsz=160, warmup=1)
    # 3 images, 1 warmup -> 2 measured.
    assert result.frames == 2
    assert result.cut == 10
    t = result.timings
    for stage in (t.prep_ms, t.edge_ms, t.cloud_ms):
        assert stage > 0
    assert t.total_ms > 0 and t.fps > 0
    assert result.wire_bytes > 0
    assert result.jpeg_bytes > 0
    # No accuracy was requested.
    assert result.map50_95 is None and result.delta_map50_95 is None


def test_transport_shrinks_the_wire(det_model, images_dir):
    raw = benchmark_directory(det_model, images_dir, imgsz=160, warmup=1)
    int8 = benchmark_directory(
        det_model, images_dir, imgsz=160, warmup=1, transport=Int8Transport(compress=True)
    )
    assert int8.wire_bytes < raw.wire_bytes
    assert int8.timings.wire_ms > 0  # the codec costs time


def test_warmup_consuming_every_frame_is_an_error(det_model, images_dir):
    with pytest.raises(ValueError, match="warmup"):
        benchmark_split(det_model, iter_image_frames(images_dir), imgsz=160, warmup=99)


def test_power_sampler_is_used_when_it_returns_a_value(det_model, images_dir):
    result = benchmark_directory(
        det_model, images_dir, imgsz=160, warmup=1, power_sampler=lambda: 7.5
    )
    assert result.power_w == pytest.approx(7.5)


def test_power_is_none_when_unsupported(det_model, images_dir):
    result = benchmark_directory(
        det_model, images_dir, imgsz=160, warmup=1, power_sampler=lambda: None
    )
    assert result.power_w is None


def test_read_jetson_power_returns_none_or_float():
    # No Jetson rails in CI; must degrade to None rather than raise.
    value = read_jetson_power()
    assert value is None or isinstance(value, float)


def test_benchmark_cli(capsys, images_dir, tmp_path):
    from splitflow.cli import main

    out_json = tmp_path / "bench.json"
    code = main(
        [
            "benchmark",
            "--model",
            "yolo11n.yaml",
            "--images",
            str(images_dir),
            "--imgsz",
            "160",
            "--warmup",
            "1",
            "--json",
            str(out_json),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| metric | value |" in out
    assert "throughput" in out
    payload = json.loads(out_json.read_text())
    assert payload["frames"] == 2
    assert payload["fps"] > 0


def test_benchmark_cli_transport_none(capsys, images_dir):
    from splitflow.cli import main

    assert (
        main(
            [
                "benchmark",
                "--model",
                "yolo11n.yaml",
                "--images",
                str(images_dir),
                "--imgsz",
                "160",
                "--warmup",
                "1",
                "--transport",
                "none",
            ]
        )
        == 0
    )
    assert "wire" in capsys.readouterr().out
