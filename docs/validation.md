# Validation: the `evaluate --bottleneck` path on public data

A reproducible end-to-end check of the accuracy path on **public** weights and
data (no private model or dataset), so anyone can run it. It exercises
`splitflow evaluate --bottleneck` — baseline vs split+bottleneck mAP on the same
validation set — and produces a first real data point.

## Setup

- Model: `yolo11n.pt` (public COCO-pretrained nano weights, ~5.6 MB).
- Data: `coco128` (128 COCO images, auto-downloaded by ultralytics).
- Hardware: **CPU only** (no GPU was available for this run).
- Bottleneck: 8 latent channels, stride 2, trained 8 epochs, batch 8.

```bash
splitflow train-bottleneck --model yolo11n.pt \
    --images datasets/coco128/images/train2017 \
    --imgsz 640 --latent-channels 8 --stride 2 --epochs 8 --batch 8 --out bn.pt
# stride-2 bottleneck needs square inputs, so evaluate with rect=False:
python -c "from splitflow.evaluate import compare_map; \
from splitflow.bottleneck import load_bottleneck, BottleneckTransport; \
print(compare_map('yolo11n.pt','coco128.yaml', \
transport=BottleneckTransport(load_bottleneck('bn.pt'), compress=True), \
imgsz=640, device='cpu', rect=False).to_dict())"
```

## Result

| configuration | mAP50 | mAP50-95 |
|---|---:|---:|
| baseline (unsplit yolo11n) | 0.674 | 0.505 |
| split + bottleneck (8 epochs, CPU) | 0.031 | 0.014 |

The final bottleneck reconstruction error was still ~0.6–0.7 (relative) per
pyramid level after 8 CPU epochs on 128 images.

## What this tells us

- **The pipeline works end to end.** The split model runs through the standard
  ultralytics `val()` and produces mAP, so `evaluate --bottleneck` is exercised
  on real weights and data — not just unit tests.
- **Bottleneck quality is the whole game.** An under-trained bottleneck
  (reconstruction error ~0.6) collapses accuracy: mAP50-95 drops from 0.505 to
  0.014. This is expected — 8 epochs on 128 images on a CPU is nowhere near
  enough. It quantifies *why* the proper validation must GPU-train the
  bottleneck on a real dataset for many more epochs, driving reconstruction
  error down toward ~0.1 before accuracy is retained.

This is a negative-but-useful data point: it validates the machinery and sets
the bar the real (GPU) training has to clear, not a claim that the method
retains accuracy — that number is still to be measured on a trained bottleneck.

## Two issues this run surfaced (both handled)

1. **Params from a loaded checkpoint.** `build_graph` read `m.np` (set by
   ultralytics' `parse_model`), which is absent on modules restored from some
   `.pt` files, raising `AttributeError`. Fixed by computing the parameter
   count directly from the module's weights.
2. **Stride-2 bottleneck vs rectangular inference.** A stride-`s` bottleneck
   needs feature maps whose spatial size is divisible by `s`; ultralytics
   `val()` defaults to rectangular batches (e.g. 8×21 feature maps), which are
   not. Use `rect=False` (square inputs) with a stride>1 bottleneck, or a
   `stride=1` bottleneck, which has no divisibility constraint and works at any
   input size.
