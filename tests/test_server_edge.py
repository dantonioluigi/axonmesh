from __future__ import annotations

import threading

import pytest
import torch

from splitflow.bottleneck import Bottleneck
from splitflow.edge import EdgeClient, run_edge
from splitflow.policy import AdaptivePolicy, ConfidenceEMADrift, Detection, Mode
from splitflow.protocol import ProtocolError
from splitflow.server import CloudServer, Metrics, start_metrics_server
from splitflow.split import SplitRunner
from splitflow.stream import iter_image_frames


def start(server: CloudServer) -> None:
    threading.Thread(target=server.serve_forever, daemon=True).start()


@pytest.fixture()
def server(det_model):
    srv = CloudServer(det_model, imgsz=160, host="127.0.0.1", port=0)
    start(srv)
    yield srv
    srv.shutdown()


@pytest.fixture()
def client(server, det_model):
    with EdgeClient("127.0.0.1", server.port, det_model, imgsz=160) as c:
        yield c


def test_handshake_rejects_different_weights(server):
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(99)  # different weights on purpose
    other = DetectionModel(cfg="yolo11n.yaml", ch=3, nc=4, verbose=False).float().eval()
    with pytest.raises(ProtocolError, match="model"):
        EdgeClient("127.0.0.1", server.port, other, imgsz=160)


def test_handshake_rejects_wrong_imgsz(server, det_model):
    with pytest.raises(ProtocolError, match="imgsz"):
        EdgeClient("127.0.0.1", server.port, det_model, imgsz=320)


def test_features_round_trip_matches_local(server, client, det_model, bgr_image):
    remote, nbytes = client.infer_features(bgr_image)
    assert nbytes > 0

    # Replicate the exact pipeline locally: same modules, same INT8 round trip.
    from splitflow.measure import to_input_tensor
    from splitflow.policy import deserialize_detections
    from splitflow.protocol import pack_tensors, unpack_tensors

    runner = SplitRunner(det_model)
    wire = unpack_tensors(pack_tensors(runner.edge(to_input_tensor(bgr_image, 160))))
    local = deserialize_detections(server.postprocess(runner.cloud(wire)))
    assert remote == local


def test_frame_round_trip_and_retrain_queue(det_model, bgr_image, tmp_path):
    srv = CloudServer(det_model, imgsz=160, host="127.0.0.1", port=0, retrain_dir=tmp_path)
    start(srv)
    try:
        with EdgeClient("127.0.0.1", srv.port, det_model, imgsz=160) as client:
            detections, nbytes = client.infer_frame(bgr_image, frame_id=7)
        assert isinstance(detections, list)
        assert nbytes > 1000  # a JPEG, not a header
        jpgs = list(tmp_path.glob("*_7.jpg"))
        metas = list(tmp_path.glob("*_7.json"))
        assert len(jpgs) == 1 and len(metas) == 1
    finally:
        srv.shutdown()


def test_detections_mode_is_ack_only(server, client):
    dets = [Detection(1, 0.9, (0.1, 0.1, 0.5, 0.5))]
    nbytes = client.send_detections(dets, frame_id=3)
    assert nbytes == 17 + 2 + 11  # header + count + one detection


def test_bottleneck_wire_is_smaller(det_model, bgr_image):
    torch.manual_seed(15)
    runner = SplitRunner(det_model)
    bottleneck = Bottleneck.for_runner(runner, latent_channels=4, stride=1, imgsz=160)

    srv_plain = CloudServer(det_model, imgsz=160, host="127.0.0.1", port=0)
    srv_bn = CloudServer(det_model, imgsz=160, host="127.0.0.1", port=0, bottleneck=bottleneck)
    start(srv_plain)
    start(srv_bn)
    try:
        with EdgeClient("127.0.0.1", srv_plain.port, det_model, imgsz=160) as c:
            _, plain_bytes = c.infer_features(bgr_image)
        with EdgeClient("127.0.0.1", srv_bn.port, det_model, imgsz=160, bottleneck=bottleneck) as c:
            _, bn_bytes = c.infer_features(bgr_image)
        assert bn_bytes < plain_bytes / 4
    finally:
        srv_plain.shutdown()
        srv_bn.shutdown()


def test_run_edge_over_directory(server, client, images_dir):
    script = iter(
        [[Detection(0, 0.9, (0.1, 0.1, 0.5, 0.5))], [], [Detection(0, 0.1, (0, 0, 1, 1))]]
    )
    policy = AdaptivePolicy(drift=ConfidenceEMADrift(threshold=0.0))
    reports = run_edge(
        iter_image_frames(images_dir),
        lambda _img: next(script),
        policy,
        client,
    )
    assert [r.mode for r in reports] == [Mode.DETECTIONS, Mode.FEATURES, Mode.FRAME]
    assert all(r.nbytes > 0 for r in reports)
    metrics = server.metrics.render()
    assert 'frames_total{mode="detections"} 1' in metrics
    assert 'frames_total{mode="features"} 1' in metrics
    assert 'frames_total{mode="frame"} 1' in metrics


def test_metrics_http_endpoints():
    import urllib.error

    metrics = Metrics()
    metrics.inc("frames_total", mode="features")
    httpd = start_metrics_server(metrics, port=0, host="127.0.0.1")
    port = httpd.server_address[1]
    try:
        health = urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz")
        assert health.status == 200
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics").read().decode()
        assert 'splitflow_frames_total{mode="features"} 1.0' in body
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope")
        assert err.value.code == 404
    finally:
        httpd.shutdown()
