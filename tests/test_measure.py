from __future__ import annotations

import pytest
import torch

from splitflow.measure import (
    jpeg_nbytes,
    letterbox,
    measure_directory,
    measure_frame,
    summarize,
    to_input_tensor,
    to_markdown,
)
from splitflow.split import SplitRunner


def test_jpeg_nbytes_grows_with_quality(bgr_image):
    sizes = [jpeg_nbytes(bgr_image, q) for q in (20, 60, 95)]
    assert all(s > 0 for s in sizes)
    assert sizes[0] < sizes[1] < sizes[2]


def test_letterbox_pads_to_square(bgr_image):
    boxed = letterbox(bgr_image, 160)
    assert boxed.shape == (160, 160, 3)
    # 240x320 → 120x160 content, grey bars top and bottom.
    assert (boxed[0] == 114).all()
    assert (boxed[-1] == 114).all()


def test_to_input_tensor_shape_and_range(bgr_image):
    x = to_input_tensor(bgr_image, 160)
    assert x.shape == (1, 3, 160, 160)
    assert x.dtype == torch.float32
    assert x.min() >= 0.0
    assert x.max() <= 1.0


def test_measure_frame_prices_every_option(det_model, bgr_image):
    runner = SplitRunner(det_model)
    m = measure_frame(runner, bgr_image, name="f0", file_bytes=123, imgsz=160)
    assert m.wire_fp32 == 2 * m.wire_fp16
    assert 0 < m.wire_int8 < m.wire_fp16 < m.wire_fp32
    assert 0 < m.wire_int8_zlib <= m.wire_int8 * 1.01 + 128
    assert m.jpeg_bytes > 0
    assert m.int8_vs_jpeg == pytest.approx(m.jpeg_bytes / m.wire_int8)


def test_measure_directory_and_summary(det_model, images_dir):
    runner = SplitRunner(det_model)
    measurements = measure_directory(runner, images_dir, imgsz=160, quality=85)
    assert [m.name for m in measurements] == ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"]
    assert all(m.file_bytes > 0 for m in measurements)

    summary = summarize(measurements)
    assert summary["frames"] == 3
    assert summary["mean_wire_int8"] == pytest.approx(sum(m.wire_int8 for m in measurements) / 3)


def test_measure_directory_respects_limit(det_model, images_dir):
    runner = SplitRunner(det_model)
    assert len(measure_directory(runner, images_dir, imgsz=160, limit=2)) == 2


def test_measure_directory_rejects_empty(det_model, tmp_path):
    with pytest.raises(FileNotFoundError):
        measure_directory(SplitRunner(det_model), tmp_path)


def test_markdown_table_has_all_rows(det_model, images_dir):
    runner = SplitRunner(det_model)
    measurements = measure_directory(runner, images_dir, imgsz=160)
    table = to_markdown(measurements)
    lines = table.splitlines()
    assert lines[0].startswith("| frame ")
    assert len(lines) == 2 + len(measurements) + 1  # header + separator + rows + mean
    assert "**mean**" in lines[-1]
