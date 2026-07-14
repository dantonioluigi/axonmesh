"""Shared fixtures.

Tests build YOLO11n from its bundled YAML (random weights, fixed seed): the
split/quantisation machinery is weight-agnostic, so nothing needs downloading.
YOLO11n has the exact same topology as the YOLO11l used in production — only
depth/width differ.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def det_model():
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(15)
    model = DetectionModel(cfg="yolo11n.yaml", ch=3, nc=4, verbose=False)
    return model.float().eval()


@pytest.fixture(scope="session")
def graph(det_model):
    from yolosplit.topology import build_graph

    return build_graph(det_model)


@pytest.fixture(scope="session")
def probe():
    torch.manual_seed(15)
    return torch.rand(1, 3, 160, 160)


@pytest.fixture()
def bgr_image():
    rng = np.random.default_rng(15)
    # Smooth gradient + mild noise: compresses like a real frame, not like static.
    gradient = np.linspace(0, 255, 320, dtype=np.uint8)
    image = np.stack([np.tile(gradient, (240, 1))] * 3, axis=-1).astype(np.int16)
    noise = rng.integers(0, 24, size=image.shape, dtype=np.int16)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


@pytest.fixture()
def images_dir(tmp_path, bgr_image):
    import cv2

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"frame_{i}.jpg"), np.roll(bgr_image, i * 7, axis=1))
    return tmp_path
