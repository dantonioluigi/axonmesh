# yolo-split-computing

[![CI](https://github.com/dantonioluigi/yolo-split-computing/actions/workflows/ci.yml/badge.svg)](https://github.com/dantonioluigi/yolo-split-computing/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

Split computing experiments for YOLO11 detection models: cut the network after the
backbone, run the first half on the edge device (e.g. Jetson AGX Orin), and ship
**quantised intermediate tensors** to the cloud half instead of JPEG frames.

The question this repo answers with numbers: *is transmitting the backbone output
(INT8-quantised) actually cheaper than transmitting the JPEG frame, and how much
mAP does it cost?*

```mermaid
flowchart LR
    subgraph edge [Edge - Jetson]
        A[camera frame] --> B[backbone<br/>layers 0-10]
        B --> Q[INT8 quantise<br/>P3 / P4 / P5]
    end
    Q -- "wire: bytes measured here" --> D
    subgraph cloud [Cloud - K8s node]
        D[dequantise] --> E[neck + head<br/>layers 11-23]
        E --> F[detections]
    end
```

## Why it is not just `model.model[:10]`

YOLO11's neck consumes the backbone taps **P3, P4 and P5** (layers 4, 6 and 10)
through skip connections, so a naive sequential slice silently drops two of the
three tensors the cloud half needs. This repo resolves the model graph
(`m.f`/`m.i` wiring), computes the exact *wire set* for any cut point, and runs
the two halves so that the split output is **bit-identical** to the unsplit model
(verified in the test suite).

## Install

```bash
git clone https://github.com/dantonioluigi/yolo-split-computing
cd yolo-split-computing
python -m venv .venv && source .venv/bin/activate
# CPU-only torch keeps the venv small; skip this line on machines with CUDA/Jetson
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Usage

**1. Inspect the architecture and price every cut point** (works with a `.pt`
checkpoint or a bare model YAML):

```bash
yolosplit inspect --model yolo11l.pt --imgsz 640
```

Prints the layer graph (with resolved skip connections) and, for every candidate
cut, which tensors must cross the wire and their fp32/fp16/int8 sizes.

**2. Measure bandwidth on real frames** — JPEG at production quality vs the
wire set produced by the edge half on the *same letterboxed pixels*:

```bash
yolosplit measure --model yolo11l.pt --images path/to/images/val \
    --quality 85 --json results/measure.json
```

**3. Measure the accuracy cost** — full validation twice on the same dataset,
unsplit vs split+INT8:

```bash
yolosplit evaluate --model yolo11l.pt --data data.yaml \
    --transport int8 --per-channel --json results/eval.json
```

**4. Train the learned bottleneck** — the piece that closes the ~30x gap. A
small per-level autoencoder is trained by feature distillation (detector
frozen, simulated INT8 noise on the latents); with the defaults
(`--latent-channels 8 --stride 2`) the INT8 latent is ~17 KB/frame vs ~47 KB
of JPEG q85:

```bash
yolosplit train-bottleneck --model yolo11l.pt \
    --images path/to/images/train --device 0 --out bottleneck.pt
yolosplit evaluate --model yolo11l.pt --data data.yaml \
    --bottleneck bottleneck.pt --json results/eval_bottleneck.json
```

**5. Simulate the adaptive stream** — the edge runs the detector locally and
per frame ships only what the frame deserves: serialised boxes (11 bytes per
detection) when confident, (bottlenecked) features when uncertain, the full
JPEG on drift or low confidence — which is also enqueued as a retraining
candidate:

```bash
yolosplit stream --model yolo11l.pt --images path/to/images/val \
    --bottleneck bottleneck.pt --json results/stream.json
```

Everything is also available as a library:

```python
from ultralytics import YOLO
from yolosplit import SplitRunner, Int8Transport, split_inference

yolo = YOLO("yolo11l.pt")
runner = SplitRunner(yolo.model, transport=Int8Transport(axis=1))
detections = runner(x)                  # edge -> quantise -> wire -> cloud
print(runner.stats.mean_bytes)          # bytes/frame that crossed the wire

with split_inference(yolo.model, transport=Int8Transport()) as runner:
    yolo.val(data="data.yaml")       # standard ultralytics val, split underneath
```

## Results

First measurement — a YOLO11l fine-tuned on a private industrial dataset
(4 classes), 12 frames, 640×640,
backbone cut (layer 10, wire set = P3/P4/P5, i.e. layers 4/6/10):

| configuration | wire KB/frame (mean) | vs JPEG q85 |
|---|---:|---:|
| JPEG q85, letterboxed @640 (baseline) | 47.1 | 1.0x |
| fp32 tensors | 16 800 | 357x **larger** |
| fp16 tensors | 8 400 | 178x **larger** |
| INT8 per-tensor | 4 200 | 89x **larger** |
| INT8 per-tensor + zlib | 1 406 | 30x **larger** |

**Finding:** at the backbone cut, naive quantisation does not come close: even
INT8+zlib ships ~30x more bytes than the JPEG the model would otherwise consume.
This confirms the known risk rather than killing the idea — it quantifies the
gap a **learned bottleneck at the cut** has to close (~30x on top of INT8+zlib)
for feature shipping to beat frame shipping. mAP cost of INT8 (via
`yolosplit evaluate`) is only worth measuring once a bottleneck
makes the size competitive.

Latency numbers measured off-device are not representative; re-measure on the
Jetson before drawing conclusions about end-to-end delay.

## Roadmap

- [x] Graph-aware splitter, bit-exact split inference (0.1.0)
- [x] INT8 wire + bandwidth/accuracy measurement (0.1.0) → finding: raw INT8
      loses to JPEG by ~30x at the backbone cut
- [x] Learned bottleneck at the cut, trained by feature distillation (0.2.0)
- [x] Adaptive transmission policy + stream simulator with retraining queue (0.2.0)
- [ ] Validate: bottleneck mAP cost (`evaluate --bottleneck`) trained on GPU
- [x] Cut planner: pick the split point from a bandwidth/FPS budget (0.3.0)
- [ ] Live re-planning: feed measured bandwidth/GPU metrics into the planner
- [ ] Kubernetes operator with GitOps rollout (gated on the numbers above)

See [docs/experiment-protocol.md](docs/experiment-protocol.md) for the protocol
and [docs/maintenance.md](docs/maintenance.md) for how the repo is maintained.

## Development

```bash
pytest                 # runs with coverage (fails under 85%)
ruff check . && ruff format --check .
pre-commit install     # optional: run the same checks on every commit
```

Tests build YOLO11n from its bundled YAML with random weights — no downloads, no
GPU needed. YOLO11n shares its topology with the larger YOLO11 variants.

## License

[MIT](LICENSE)
