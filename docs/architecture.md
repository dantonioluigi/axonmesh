# Architecture: why the split is graph-aware, and what is a seam

## Why it is not a `model[:k]` slice

A neck consumes several backbone taps through skip connections — in YOLO11,
layers 4, 6 and 10 — so a naive sequential slice silently drops tensors the
cloud half still needs. axonmesh resolves the model graph, computes the exact
*wire set* for any cut point, and runs the two halves so the split output is
**bit-identical** to the unsplit model (verified in the test suite).

Which layers exist and how to run them comes from a small **adapter contract**
(`graph` / `default_cut` / `probe_shapes` / `run_span`), so the planner, codecs,
wire protocol, policy and operator are architecture-agnostic. Ultralytics is the
first adapter, `torch.fx` is the catch-all: ResNet-18, MobileNetV3 and ViT-B/16
all split bit-identically without a line written for them.

```python
from axonmesh import SplitModel, Int8Transport

model = SplitModel(YOLO("yolo11l.pt").model)   # any adapted model
model.plan(bandwidth_mbps=50, fps=10)          # pick the cut for the link
model.split(transport=Int8Transport(compress=True))
detections = model.run(frame)                  # edge → wire → cloud
cr = model.deploy(name="detector", image="ghcr.io/you/cloud:0.8.0",
                  model_url="https://store/model.pt")   # kubectl apply this
```


## Not just YOLO / not just Jetson

The specifics are seams, not assumptions:

- **Model** — a `ModelAdapter` answers four questions (`graph`, `default_cut`,
  `probe_shapes`, `run_span`) and registers a detector; `SplitModel(model)` then
  resolves one automatically. `UltralyticsAdapter` reads YOLO's wiring;
  **`FxAdapter` handles anything else traceable** by recovering the graph with
  `torch.fx`, so it is the fallback for arbitrary models. Verified end to end
  (bit-identical split, INT8 wire) on:

  | model | backend | layers | split exact |
  |---|---|---:|---|
  | YOLO11 | ultralytics | 24 | ✅ |
  | ResNet-18 | torch.fx | 70 | ✅ |
  | MobileNetV3-Small | torch.fx | 158 | ✅ |
  | ViT-B/16 | torch.fx | 235 | ✅ |

  A purpose-built adapter always beats the fallback: registrations sort ahead of
  it, so adding a family is a `register_adapter(...)` call, not a fork. Models
  that assert on input shape (torchvision's ViT) can be traced by the caller and
  handed over as a `GraphModule`.
- **Task** — the wire carries *opaque* result bytes, so a different head plugs
  in by replacing the server's `postprocess` (YOLO NMS is only the default).
- **Edge device** — "edge" is any host that runs the first half and speaks the
  protocol. Jetson is the reference target, but the edge image is plain Python
  and builds for amd64/arm64 alike.
- **Transport** — raw / INT8 / learned bottleneck are pluggable `Transport`
  objects; a new codec is one class implementing the wire round-trip.
