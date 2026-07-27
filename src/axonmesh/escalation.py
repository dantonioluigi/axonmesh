"""Escalate to the serving stack you already run — and audit it while you do.

The TCP cascade assumes the cloud side is `axonmesh serve`. A team running
KServe, Triton or any HTTP predictor should not have to adopt a bespoke TCP
server to put a cascade in front of it: :class:`HttpEscalation` makes the
escalation target an HTTP endpoint, either the Open Inference Protocol
(KServe V2 / Triton) or a plain JPEG-in/JSON-out contract. axonmesh then does
the one thing the serving stack does not — decide which frames ever reach it.

The second class closes the loop the calibration opens.
:class:`AgreementAuditor` escalates a small fraction of *confident* frames
anyway and compares the cloud's answer with the edge's, exactly the label-free
agreement `calibrate` used to pick the threshold. Calibration is a snapshot of
one afternoon's footage; scenes drift with season, weather and time of day, and
a threshold that was measured then is a guess again six months later. The
rolling agreement is the live version of the calibrated number, and its fall
below the floor is the signal to recalibrate — no annotations at any point.
"""

from __future__ import annotations

import base64
import json
import random
import urllib.request
from collections import deque
from typing import Any

import cv2
import numpy as np

from .calibrate import detection_agreement
from .policy import Detection, serialize_detections
from .protocol import ProtocolError, Role


def _encode_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def _rows_to_detections(rows: list[list[float]]) -> list[Detection]:
    return [Detection(int(row[5]), float(row[4]), (row[0], row[1], row[2], row[3])) for row in rows]


class HttpEscalation:
    """Escalation target behind plain HTTP, drop-in where ``EdgeClient`` goes.

    Two wire formats:

    - ``"oip"`` — the Open Inference Protocol REST API (KServe V2, Triton).
      The frame travels as a base64 ``BYTES`` tensor named ``image``; the
      response's first output tensor is read as ``[N, 6]`` rows of
      ``x1, y1, x2, y2, conf, cls`` with coordinates normalised to ``[0, 1]``.
      That tensor contract is this project's, not the protocol's: the predictor
      or transformer behind the endpoint has to decode the JPEG and emit it
      (``docs/deployment.md`` spells it out). Verified against a conformance
      stub of the protocol in the test suite — not against a live KServe.
      Base64 inflates the JPEG by a third, and the byte accounting charges for
      it: the REST form of the protocol really does cost 4/3 of the frame.
    - ``"json"`` — the frame as a raw ``image/jpeg`` body, the response as
      ``{"detections": [[x1, y1, x2, y2, conf, cls], ...]}`` normalised. The
      smallest possible contract for a custom predictor.

    A confident frame is answered on the device and never reaches the
    endpoint; ``send_detections`` only prices what shipping the result to a
    backend would cost (11 bytes each), so the accounting stays comparable
    with the TCP cascade. Subtract it if detections are consumed on-device.
    """

    role = Role.CASCADE

    def __init__(
        self,
        url: str,
        fmt: str = "json",
        model: str = "detector",
        timeout: float = 30.0,
    ) -> None:
        if fmt not in ("json", "oip"):
            raise ValueError(f"format must be 'json' or 'oip', got {fmt!r}")
        self.url = url.rstrip("/")
        self.fmt = fmt
        self.model = model
        self.timeout = timeout

    def send_detections(self, detections: list[Detection], frame_id: int = 0) -> int:
        return len(serialize_detections(detections))

    def infer_features(self, image_bgr: np.ndarray, frame_id: int = 0) -> tuple[list, int]:
        raise ProtocolError(
            "an HTTP serving endpoint cannot consume another model's activations; "
            "a cascade escalates the frame"
        )

    def infer_frame(
        self, image_bgr: np.ndarray, frame_id: int = 0, quality: int = 85
    ) -> tuple[list[Detection], int]:
        jpeg = _encode_jpeg(image_bgr, quality)
        if self.fmt == "oip":
            body = json.dumps(
                {
                    "inputs": [
                        {
                            "name": "image",
                            "shape": [1],
                            "datatype": "BYTES",
                            "data": [base64.b64encode(jpeg).decode()],
                        }
                    ]
                }
            ).encode()
            request = urllib.request.Request(
                f"{self.url}/v2/models/{self.model}/infer",
                data=body,
                headers={"Content-Type": "application/json"},
            )
        else:
            body = jpeg
            request = urllib.request.Request(
                self.url, data=body, headers={"Content-Type": "image/jpeg"}
            )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode())
        except (OSError, ValueError) as err:
            raise ProtocolError(f"escalation to {self.url} failed: {err}") from err
        return self._parse(payload), len(body)

    def _parse(self, payload: dict[str, Any]) -> list[Detection]:
        try:
            if self.fmt == "oip":
                output = payload["outputs"][0]
                flat = output["data"]
                rows = [flat[i : i + 6] for i in range(0, len(flat), 6)]
            else:
                rows = payload["detections"]
            return _rows_to_detections(rows)
        except (KeyError, IndexError, TypeError, ValueError) as err:
            raise ProtocolError(f"unparseable response from {self.url}: {err}") from err

    def close(self) -> None:  # symmetry with EdgeClient; nothing is held open
        return None


class AgreementAuditor:
    """Continuously re-measure, on live traffic, what ``calibrate`` measured once.

    With probability ``rate``, a frame the policy would answer locally is
    escalated *as well*, and the cloud's answer is compared with the edge's —
    the same label-free agreement the calibration sweep is built on. The
    rolling mean is directly comparable with the agreement ``calibrate``
    reported for the chosen threshold; when it sinks below ``floor``, the
    scene has moved and the threshold is due for recalibration.

    ``stale`` withholds judgement until ``min_samples`` audits have landed:
    a single disagreeing frame is weather, a windowful is climate.
    """

    def __init__(
        self,
        rate: float,
        floor: float | None = None,
        window: int = 50,
        min_samples: int = 5,
        seed: int = 15,
    ) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"audit rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.floor = floor
        self.min_samples = min_samples
        self._samples: deque[float] = deque(maxlen=window)
        self._rng = random.Random(seed)

    def should_audit(self) -> bool:
        return self._rng.random() < self.rate

    def record(self, edge: list[Detection], cloud: list[Detection]) -> float:
        agreement = detection_agreement(edge, cloud)
        self._samples.append(agreement)
        return agreement

    @property
    def samples(self) -> int:
        return len(self._samples)

    @property
    def agreement(self) -> float | None:
        return sum(self._samples) / len(self._samples) if self._samples else None

    @property
    def stale(self) -> bool:
        """True once enough audited frames agree less than the calibrated floor."""
        return (
            self.floor is not None
            and self.samples >= self.min_samples
            and self.agreement is not None
            and self.agreement < self.floor
        )
