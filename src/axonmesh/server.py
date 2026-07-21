"""The cloud half as a long-running service — the thing Kubernetes deploys.

``CloudServer`` speaks the wire protocol on one TCP port and exposes
``/healthz`` + ``/metrics`` (Prometheus text format, dependency-free) on a
second one, so liveness probes and scraping work out of the box.

The server is generic over the task: the raw output of the cloud half is
turned into result bytes by the ``postprocess`` callable. The default is YOLO
NMS → the compact detection codec, but swapping that one function serves a
different head (segmentation, classification, another detector) over the same
protocol and the same split machinery.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import traceback
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch.nn as nn

from .bottleneck import Bottleneck
from .measure import to_input_tensor
from .policy import Detection, serialize_detections
from .protocol import (
    ConnectionClosed,
    Handshake,
    Kind,
    ProtocolError,
    module_fingerprint,
    recv_message,
    send_message,
    unpack_tensors,
)
from .split import SplitRunner, primary_output

Postprocess = Callable[[Any], bytes]


class Metrics:
    """Thread-safe counters rendered in Prometheus text format."""

    def __init__(self, prefix: str = "axonmesh") -> None:
        self.prefix = prefix
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def render(self) -> str:
        lines = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                label_s = ",".join(f'{k}="{v}"' for k, v in labels)
                series = (
                    f"{self.prefix}_{name}{{{label_s}}}" if label_s else f"{self.prefix}_{name}"
                )
                lines.append(f"{series} {value}")
        return "\n".join(lines) + "\n"


def start_metrics_server(metrics: Metrics, port: int, host: str = "0.0.0.0") -> HTTPServer:
    """Serve ``/healthz`` and ``/metrics`` on a daemon thread; returns the server."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/healthz":
                body, status, ctype = b"ok\n", 200, "text/plain"
            elif self.path == "/metrics":
                body, status, ctype = metrics.render().encode(), 200, "text/plain; version=0.0.4"
            else:
                body, status, ctype = b"not found\n", 404, "text/plain"
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep probe traffic out of stdout
            pass

    server = HTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _import_nms():
    """Locate ultralytics' NMS across versions (moved ops→nms around 8.4)."""
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:  # pragma: no cover - depends on the installed version
        from ultralytics.utils.ops import non_max_suppression
    return non_max_suppression


def make_yolo_postprocess(imgsz: int, conf: float = 0.25, iou: float = 0.45) -> Postprocess:
    """Default result codec: YOLO NMS → serialised detections.

    Coordinates are normalised to the letterboxed input (protocol v1
    convention); the edge knows the original frame size and can unletterbox.
    """

    def postprocess(raw: Any) -> bytes:
        non_max_suppression = _import_nms()

        pred = primary_output(raw)
        boxes = non_max_suppression(pred, conf_thres=conf, iou_thres=iou)[0]
        detections = [
            Detection(
                int(b[5]), float(b[4]), (b[0] / imgsz, b[1] / imgsz, b[2] / imgsz, b[3] / imgsz)
            )
            for b in boxes.tolist()
        ]
        return serialize_detections(detections)

    return postprocess


class CloudServer:
    """Receives wire tensors (or frames), finishes inference, replies with results."""

    def __init__(
        self,
        det_model: nn.Module,
        cut: int | None = None,
        bottleneck: Bottleneck | None = None,
        imgsz: int = 640,
        postprocess: Postprocess | None = None,
        retrain_dir: str | Path | None = None,
        host: str = "0.0.0.0",
        port: int = 9095,
    ) -> None:
        self.runner = SplitRunner(det_model, cut=cut)
        self.bottleneck = bottleneck
        self.imgsz = imgsz
        self.postprocess = postprocess or make_yolo_postprocess(imgsz)
        self.retrain_dir = Path(retrain_dir) if retrain_dir else None
        if self.retrain_dir:
            self.retrain_dir.mkdir(parents=True, exist_ok=True)
        self.handshake = Handshake(
            model=module_fingerprint(det_model),
            bottleneck=module_fingerprint(bottleneck) if bottleneck else None,
            cut=self.runner.cut,
            imgsz=imgsz,
        )
        self.metrics = Metrics()
        self._stop = threading.Event()
        self._listener = socket.create_server((host, port))
        self._listener.settimeout(0.5)
        self.port = self._listener.getsockname()[1]

    def serve_forever(self) -> None:
        """Accept connections until :meth:`shutdown`; one thread per client."""
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            self.metrics.inc("connections_total")
            threading.Thread(target=self._serve_client, args=(conn,), daemon=True).start()
        self._listener.close()

    def shutdown(self) -> None:
        self._stop.set()

    def _serve_client(self, conn: socket.socket) -> None:
        with conn:
            try:
                if not self._handshake_ok(conn):
                    return
                while True:
                    kind, frame_id, payload = recv_message(conn)
                    self._handle_frame(conn, kind, frame_id, payload)
            except ConnectionClosed:
                return
            except ProtocolError:
                self.metrics.inc("errors_total")
                return
            except Exception:
                # A client must not be able to take a serving thread down
                # quietly: without this the thread dies, the connection drops
                # and errors_total stays at zero, so /metrics reports a healthy
                # server while it is being fed garbage. Log it and count it.
                self.metrics.inc("errors_total")
                traceback.print_exc()
                return

    def _handshake_ok(self, conn: socket.socket) -> bool:
        kind, _, payload = recv_message(conn)
        if kind is not Kind.HELLO:
            raise ProtocolError(f"expected HELLO, got {kind.name}")
        mismatches = self.handshake.mismatches(Handshake.from_bytes(payload))
        if mismatches:
            detail = json.dumps({"mismatches": mismatches})
            send_message(conn, Kind.ERROR, detail.encode())
            self.metrics.inc("handshake_failures_total")
            return False
        send_message(conn, Kind.ACK, self.handshake.to_bytes())
        return True

    def _handle_frame(self, conn, kind: Kind, frame_id: int, payload: bytes) -> None:
        started = time.perf_counter()
        if kind is Kind.FEATURES:
            wire = unpack_tensors(payload)
            if self.bottleneck is not None:
                wire = self.bottleneck.decode(wire)
            result = self.postprocess(self.runner.cloud(wire))
            send_message(conn, Kind.RESULT, result, frame_id)
        elif kind is Kind.FRAME:
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ProtocolError("FRAME payload is not a decodable image")
            x = to_input_tensor(image, self.imgsz)
            result = self.postprocess(self.runner.cloud(self.runner.edge(x)))
            self._enqueue_retrain(frame_id, payload)
            send_message(conn, Kind.RESULT, result, frame_id)
        elif kind is Kind.DETECTIONS:
            send_message(conn, Kind.ACK, b"", frame_id)
        else:
            raise ProtocolError(f"unexpected message kind {kind.name}")
        mode = kind.name.lower()
        self.metrics.inc("frames_total", mode=mode)
        self.metrics.inc("wire_bytes_total", float(len(payload)), mode=mode)
        self.metrics.inc("inference_seconds_total", time.perf_counter() - started, mode=mode)

    def _enqueue_retrain(self, frame_id: int, jpeg: bytes) -> None:
        if self.retrain_dir is None:
            return
        stem = f"{int(time.time() * 1000)}_{frame_id}"
        (self.retrain_dir / f"{stem}.jpg").write_bytes(jpeg)
        meta = {"frame_id": frame_id, "received_at": time.time(), "bytes": len(jpeg)}
        (self.retrain_dir / f"{stem}.json").write_text(json.dumps(meta))
        self.metrics.inc("retrain_frames_total")
