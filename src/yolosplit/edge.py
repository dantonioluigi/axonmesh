"""The edge half as a network client.

``EdgeClient`` opens one connection, performs the fingerprint handshake and
then ships whatever the policy decides per frame: serialised detections,
quantised (bottlenecked) features, or the full JPEG. ``run_edge`` drives a
frame source through a policy against a live server — the networked twin of
the offline ``yolosplit stream`` simulator, reporting the same per-frame
accounting so the two are directly comparable.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterable

import cv2
import numpy as np
import torch.nn as nn

from .bottleneck import Bottleneck
from .measure import jpeg_nbytes, to_input_tensor
from .policy import AdaptivePolicy, Detection, Mode, deserialize_detections, serialize_detections
from .protocol import (
    Handshake,
    Kind,
    ProtocolError,
    module_fingerprint,
    pack_tensors,
    recv_message,
    send_message,
)
from .split import SplitRunner
from .stream import FrameReport, Inferer


class EdgeClient:
    """One authenticated connection to a :class:`~yolosplit.server.CloudServer`."""

    def __init__(
        self,
        host: str,
        port: int,
        det_model: nn.Module,
        cut: int | None = None,
        bottleneck: Bottleneck | None = None,
        imgsz: int = 640,
        axis: int | None = None,
        compress: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self.runner = SplitRunner(det_model, cut=cut)
        self.bottleneck = bottleneck
        self.imgsz = imgsz
        self.axis = axis
        self.compress = compress
        self._sock = socket.create_connection((host, port), timeout=timeout)
        hello = Handshake(
            model=module_fingerprint(det_model),
            bottleneck=module_fingerprint(bottleneck) if bottleneck else None,
            cut=self.runner.cut,
            imgsz=imgsz,
        )
        send_message(self._sock, Kind.HELLO, hello.to_bytes())
        kind, _, payload = recv_message(self._sock)
        if kind is Kind.ERROR:
            detail = json.loads(payload.decode()).get("mismatches", [])
            raise ProtocolError(f"handshake rejected, mismatched fields: {detail}")
        if kind is not Kind.ACK:
            raise ProtocolError(f"expected ACK, got {kind.name}")

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> EdgeClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _roundtrip(
        self, kind: Kind, payload: bytes, frame_id: int, expect: Kind
    ) -> tuple[bytes, int]:
        nbytes = send_message(self._sock, kind, payload, frame_id)
        reply, reply_id, body = recv_message(self._sock)
        if reply is not expect or reply_id != frame_id:
            raise ProtocolError(f"expected {expect.name} for frame {frame_id}, got {reply.name}")
        return body, nbytes

    def send_detections(self, detections: list[Detection], frame_id: int = 0) -> int:
        """Ship final detections (cloud only logs them); returns wire bytes."""
        _, nbytes = self._roundtrip(
            Kind.DETECTIONS, serialize_detections(detections), frame_id, Kind.ACK
        )
        return nbytes

    def infer_features(self, image_bgr: np.ndarray, frame_id: int = 0) -> tuple[list[Detection], int]:
        """Edge half + (optional bottleneck) + INT8 → cloud completes inference."""
        wire = self.runner.edge(to_input_tensor(image_bgr, self.imgsz))
        if self.bottleneck is not None:
            wire = self.bottleneck.encode(wire)
        payload = pack_tensors(wire, axis=self.axis, compress=self.compress)
        body, nbytes = self._roundtrip(Kind.FEATURES, payload, frame_id, Kind.RESULT)
        return deserialize_detections(body), nbytes

    def infer_frame(
        self, image_bgr: np.ndarray, frame_id: int = 0, quality: int = 85
    ) -> tuple[list[Detection], int]:
        """Ship the full JPEG; the cloud runs everything and enqueues it for retraining."""
        ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("JPEG encoding failed")
        body, nbytes = self._roundtrip(Kind.FRAME, buf.tobytes(), frame_id, Kind.RESULT)
        return deserialize_detections(body), nbytes


def run_edge(
    frames: Iterable[tuple[str, np.ndarray]],
    infer: Inferer,
    policy: AdaptivePolicy,
    client: EdgeClient,
    quality: int = 85,
) -> list[FrameReport]:
    """Drive frames through the policy against a live server; report per frame."""
    reports = []
    for frame_id, (name, image) in enumerate(frames):
        detections = infer(image)
        decision = policy.decide(detections)
        if decision.mode is Mode.DETECTIONS:
            nbytes = client.send_detections(detections, frame_id)
        elif decision.mode is Mode.FEATURES:
            _, nbytes = client.infer_features(image, frame_id)
        else:
            _, nbytes = client.infer_frame(image, frame_id, quality)
        reports.append(
            FrameReport(
                name=name,
                mode=decision.mode,
                nbytes=nbytes,
                jpeg_bytes=jpeg_nbytes(image, quality),
                frame_conf=decision.frame_conf,
                retrain=decision.retrain,
                reason=decision.reason,
            )
        )
    return reports
