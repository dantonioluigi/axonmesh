from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from axonmesh.escalation import AgreementAuditor, HttpEscalation
from axonmesh.policy import AdaptivePolicy, ConfidenceEMADrift, Detection, Mode
from axonmesh.protocol import ProtocolError, Role

ROWS = [[0.1, 0.1, 0.3, 0.3, 0.9, 2.0], [0.5, 0.5, 0.7, 0.7, 0.6, 0.0]]


class StubPredictor:
    """An HTTP predictor speaking both wire formats, recording what it saw."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                record = {"path": self.path, "content_type": self.headers["Content-Type"]}
                if self.path.startswith("/v2/"):
                    record["body"] = json.loads(body)
                    reply = {
                        "outputs": [
                            {
                                "name": "detections",
                                "shape": [len(ROWS), 6],
                                "datatype": "FP32",
                                "data": [v for row in ROWS for v in row],
                            }
                        ]
                    }
                else:
                    record["body"] = body
                    reply = {"detections": ROWS}
                stub.requests.append(record)
                payload = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture()
def predictor():
    stub = StubPredictor()
    yield stub
    stub.close()


def frame() -> np.ndarray:
    rng = np.random.default_rng(15)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def expected() -> list[Detection]:
    return [Detection(int(r[5]), r[4], (r[0], r[1], r[2], r[3])) for r in ROWS]


def test_json_format_ships_the_raw_jpeg_and_parses_detections(predictor):
    client = HttpEscalation(predictor.url, fmt="json")
    detections, nbytes = client.infer_frame(frame(), quality=50)

    assert detections == expected()
    seen = predictor.requests[-1]
    assert seen["content_type"] == "image/jpeg"
    assert seen["body"][:2] == b"\xff\xd8"  # a real JPEG, not a re-encoding of one
    assert nbytes == len(seen["body"])  # billed exactly what crossed the wire


def test_oip_format_speaks_the_v2_protocol(predictor):
    """The request must be one a KServe/Triton predictor would recognise."""
    client = HttpEscalation(predictor.url, fmt="oip", model="det")
    detections, _ = client.infer_frame(frame(), quality=50)

    assert detections == expected()
    seen = predictor.requests[-1]
    assert seen["path"] == "/v2/models/det/infer"
    tensor = seen["body"]["inputs"][0]
    assert (tensor["name"], tensor["datatype"], tensor["shape"]) == ("image", "BYTES", [1])
    assert base64.b64decode(tensor["data"][0])[:2] == b"\xff\xd8"


def test_oip_billing_includes_the_base64_inflation(predictor):
    """The REST form of the protocol costs ~4/3 of the frame; hiding that would
    make the KServe path look cheaper than it is."""
    _, json_bytes = HttpEscalation(predictor.url, fmt="json").infer_frame(frame(), quality=50)
    _, oip_bytes = HttpEscalation(predictor.url, fmt="oip").infer_frame(frame(), quality=50)
    assert oip_bytes > json_bytes * 4 / 3  # base64 plus the JSON envelope


def test_features_are_refused_a_cascade_escalates_frames(predictor):
    client = HttpEscalation(predictor.url, fmt="json")
    assert client.role is Role.CASCADE
    with pytest.raises(ProtocolError, match="cannot consume another model's activations"):
        client.infer_features(frame())


def test_a_dead_endpoint_is_a_protocol_error_not_a_traceback():
    client = HttpEscalation("http://127.0.0.1:1", fmt="json", timeout=0.2)
    with pytest.raises(ProtocolError, match=r"escalation .* failed"):
        client.infer_frame(frame())


def test_an_unparseable_response_names_the_endpoint(predictor):
    client = HttpEscalation(predictor.url, fmt="oip")
    with pytest.raises(ProtocolError, match="unparseable response"):
        client._parse({"wrong": "shape"})


def test_send_detections_prices_the_result_shipping_only(predictor):
    client = HttpEscalation(predictor.url, fmt="json")
    nbytes = client.send_detections(expected())
    assert nbytes == 2 + 11 * len(expected())  # count header + 11 bytes each
    assert predictor.requests == []  # and nothing crossed the wire


def test_unknown_format_is_rejected_at_construction():
    with pytest.raises(ValueError, match="'json' or 'oip'"):
        HttpEscalation("http://x", fmt="grpc")


def test_auditor_rate_bounds_are_enforced():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        AgreementAuditor(rate=1.5)


def test_auditor_at_rate_one_audits_everything_and_at_zero_nothing():
    always = AgreementAuditor(rate=1.0)
    never = AgreementAuditor(rate=0.0)
    assert all(always.should_audit() for _ in range(20))
    assert not any(never.should_audit() for _ in range(20))


def test_auditor_needs_a_windowful_before_calling_the_calibration_stale():
    """One disagreeing frame is weather; a windowful is climate."""
    auditor = AgreementAuditor(rate=1.0, floor=0.9, min_samples=5)
    for _ in range(4):
        auditor.record(expected(), [])  # total disagreement, four times
    assert auditor.agreement == 0.0 and not auditor.stale
    auditor.record(expected(), [])
    assert auditor.stale


def test_auditor_agreement_recovers_when_the_cloud_agrees():
    auditor = AgreementAuditor(rate=1.0, floor=0.9, min_samples=2, window=4)
    auditor.record(expected(), [])
    for _ in range(4):  # window pushes the bad sample out
        auditor.record(expected(), expected())
    assert auditor.agreement == 1.0 and not auditor.stale


def test_run_edge_audits_confident_frames_through_the_http_escalation(predictor, images_dir):
    """The full loop: confident frames answer locally, the auditor escalates a
    fraction anyway, the report carries the agreement and the bytes."""
    from axonmesh.edge import run_edge
    from axonmesh.stream import iter_image_frames, summarize_stream

    confident = expected()  # min conf 0.6 >= conf_high below
    policy = AdaptivePolicy(conf_high=0.5, conf_low=0.1, drift=ConfidenceEMADrift(warmup=10**9))
    auditor = AgreementAuditor(rate=1.0, floor=0.9)
    client = HttpEscalation(predictor.url, fmt="json")

    reports = run_edge(
        list(iter_image_frames(images_dir)),
        lambda _: confident,
        policy,
        client,
        auditor=auditor,
    )

    assert all(r.mode is Mode.DETECTIONS for r in reports)  # routing unchanged
    assert all(r.audit_agreement == 1.0 for r in reports)  # stub agrees with itself
    assert all(r.nbytes > 1000 for r in reports)  # the audit's frame is billed
    summary = summarize_stream(reports)
    assert summary["audited_frames"] == len(reports)
    assert summary["audit_agreement"] == 1.0
    assert not auditor.stale
