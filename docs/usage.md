# Usage: from a checkpoint to a measured decision

Every command here prints a number and says which command produced it. Byte
counts are hardware-independent; **latency is not** — measure it on the device
you will deploy on.

## Install

```bash
git clone https://github.com/dantonioluigi/axonmesh
cd axonmesh
python -m venv .venv && source .venv/bin/activate
# CPU-only torch keeps the venv small; skip this line on machines with CUDA/Jetson
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```


**1. Inspect the architecture and price every cut point** (works with a `.pt`
checkpoint or a bare model YAML):

```bash
axonmesh inspect --model yolo11l.pt --imgsz 640
```

Prints the layer graph (with resolved skip connections) and, for every candidate
cut, which tensors must cross the wire and their fp32/fp16/int8 sizes.

**2. Measure bandwidth on real frames** — JPEG at production quality vs the
wire set produced by the edge half on the *same letterboxed pixels*:

```bash
axonmesh measure --model yolo11l.pt --images path/to/images/val \
    --quality 85 --json results/measure.json
```

**3. Measure the accuracy cost** — full validation twice on the same dataset,
unsplit vs split+INT8:

```bash
axonmesh evaluate --model yolo11l.pt --data data.yaml \
    --transport int8 --per-channel --json results/eval.json
```

**4. Train the learned bottleneck** — the piece that closes the ~30x gap. A
small per-level autoencoder is trained with the detector frozen and simulated
INT8 noise on the latents; with the defaults (`--latent-channels 8 --stride 2`)
the INT8 latent is ~17 KB/frame vs ~47 KB of JPEG q85. Half the loss is taken
on the *head output* rather than the reconstructed features
(`--task-weight`), which is worth ~39% relative mAP at identical wire cost —
see [docs/validation.md](docs/validation.md), including what it does not yet
buy:

```bash
# where should the latent budget go? measured, no training needed
axonmesh allocate --model yolo11l.pt --images path/to/images/train
#   proposed --latent-channels 4:3,6:13,10:65  (same bytes, redistributed)

axonmesh train-bottleneck --model yolo11l.pt \
    --images path/to/images/train --device 0 \
    --latent-channels 4:3,6:13,10:65 --out bottleneck.pt
axonmesh evaluate --model yolo11l.pt --data data.yaml \
    --bottleneck bottleneck.pt --json results/eval_bottleneck.json
```

`allocate` exists because one latent width for every wire level spends the
budget where the pixels are, not where the accuracy is: on YOLO11n the
shallowest level takes 72% of the bytes and causes 22% of the damage, while the
deepest takes 6% and causes 47%. It coarsens each level in turn and reports what
the model's output does, so the split is measured rather than guessed.

`train-bottleneck` holds frames out before the first step and closes with the
error the codec induces on the model's output over those frames — the number
that tracks mAP. Per-level reconstruction error still prints, marked as the
diagnostic it is; it can improve while accuracy gets worse, and did.

No GPU locally? [notebooks/colab_validation.ipynb](notebooks/colab_validation.ipynb)
runs this on COCO (train2017 → val2017) on a free Colab GPU.

**5. Price edge-first inference** — the configuration that wins on bandwidth.
A small model answers on the device; the cloud is consulted only for frames it
is unsure about. Reports mAP *and* bytes so the trade is visible, not asserted:

```bash
axonmesh cascade --edge yolo11n.pt --cloud yolo11m.pt \
    --data coco128.yaml --imgsz 320 --conf-high 0.6 --statistic mean
```

`--statistic` chooses how a frame's detections become the one confidence the
threshold is applied to. The default `min` escalates if *any* object is
doubtful, which fits a station holding a few known objects and is close to a
constant on a crowded scene — on coco128 it escalates 78% of frames for the
same mAP `mean` gets from 68%.

**Pick the threshold by measuring it, not by guessing.** A detector's score is
not a probability: 0.6 does not mean "right six times in ten", and the mapping
shifts between models and scenes. `calibrate` runs both models over frames from
the deployment and asks *would the cloud have disagreed?* — which needs **no
labels**, only footage from the camera that will be running:

```bash
axonmesh calibrate --edge yolo11n.pt --cloud yolo11m.pt \
    --images ./footage --imgsz 320 --statistic mean --max-kb 5
#   chosen --conf-high 0.60  (4.677 KB/frame, agreement 0.951, escalates 41%)
```

Give it a bandwidth ceiling and it returns the most faithful threshold that
fits; give it an agreement floor and it returns the cheapest that clears it. On
public data the label-free sweep picks the same threshold the labelled mAP
measurement did.

**5b. Simulate the adaptive stream** — the same routing offline, with the
feature path available and hard frames enqueued for later use:

```bash
axonmesh stream --model yolo11l.pt --images path/to/images/val \
    --bottleneck bottleneck.pt --json results/stream.json
```

**6. Benchmark a configuration** — accuracy, throughput, bandwidth and latency
only mean something together: a cut that halves the bytes is worthless if it
doubles the edge latency. One command reports them per stage:

```bash
axonmesh benchmark --model yolo11l.pt --images path/to/frames \
    --transport int8 --compress --device 0 --json results/bench.json
# add --data data.yaml to also measure the mAP cost (slower)
```

```
| metric                  | value          |
| latency total           | 295.0 ms       |
|   · edge half           | 94.6 ms        |
|   · wire (encode+codec) | 104.2 ms       |
|   · cloud half          | 93.6 ms        |
| throughput              | 3.4 FPS        |
| wire                    | 577.7 KB/frame |
| wire vs JPEG            | 0.17x          |
| bandwidth needed        | 16.0 Mbps      |
```

Power is included on boards that expose it (Jetson INA3221 rails).

Everything is also available as a library:

```python
from ultralytics import YOLO
from axonmesh import SplitRunner, Int8Transport, split_inference

yolo = YOLO("yolo11l.pt")
runner = SplitRunner(yolo.model, transport=Int8Transport(axis=1))
detections = runner(x)                  # edge -> quantise -> wire -> cloud
print(runner.stats.mean_bytes)          # bytes/frame that crossed the wire

with split_inference(yolo.model, transport=Int8Transport()) as runner:
    yolo.val(data="data.yaml")       # standard ultralytics val, split underneath
```
