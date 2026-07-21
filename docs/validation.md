# Validation: the `evaluate --bottleneck` path on public data

A reproducible end-to-end check of the accuracy path on **public** weights and
data (no private model or dataset), so anyone can run it. It exercises
`axonmesh evaluate --bottleneck` — baseline vs split+bottleneck mAP on the same
validation set — and prices the two training objectives against each other.

## Setup

- Model: `yolo11n.pt` (public COCO-pretrained nano weights, ~5.6 MB).
- Data: `coco128` (128 COCO images, auto-downloaded by ultralytics). The codec
  trains on 96 of them and the remaining 32 are held out for the cheap
  diagnostics below; mAP is measured over all 128, so it is **not** a clean
  generalisation number — see the caveats.
- Hardware: **CPU only** (no GPU was available for this run), which is why
  `imgsz` is 320 rather than 640.
- Bottleneck: 8 latent channels, stride 2, 150 epochs, batch 8.

```bash
axonmesh train-bottleneck --model yolo11n.pt --images <coco128 train images> \
    --imgsz 320 --latent-channels 8 --stride 2 --epochs 150 --batch 8 \
    --task-weight 0.5 --out bn.pt
# a stride-2 bottleneck needs square inputs, so evaluate with rect off (default)
axonmesh evaluate --model yolo11n.pt --data coco128.yaml --imgsz 320 \
    --bottleneck bn.pt --compress
```

## Result

| configuration | mAP50 | mAP50-95 | wire bytes/frame |
|---|---:|---:|---:|
| baseline (unsplit yolo11n) | 0.528 | 0.385 | — |
| split, raw INT8 wire | 0.529 | 0.385 | 273 KB |
| split + bottleneck, feature loss only | 0.249 | 0.170 | 3.8 KB |
| split + bottleneck, `--task-weight 0.5` | 0.311 | 0.236 | 3.8 KB |

## What this tells us

- **The split itself is free.** Raw INT8 across the cut reproduces the
  baseline to four decimals. Every point of accuracy lost below that line is
  paid to the codec, not to splitting — which is worth stating, because it
  means the bandwidth work and the accuracy work are separable.
- **Training against the task is worth ~39% relative mAP.** Same architecture,
  same budget, same 3.8 KB on the wire: 0.170 → 0.236 mAP50-95. The only
  difference is that half the loss is taken on the head output instead of on
  the reconstructed features.
- **Reconstruction error is the wrong quality axis.** The task-aware codec
  reconstructs *worse* (0.557/0.607/0.601 vs 0.538/0.583/0.563 per level) while
  scoring better on everything that is measured downstream. A codec that
  discards activations no prediction depends on looks like a worse
  autoencoder, and is a better codec. `sweep` therefore draws its Pareto front
  against `output_error` — the relative error induced on the model's own
  output, one extra pass through the cloud half — with reconstruction error
  kept only as a diagnostic of the training loss.
- **Compute was never the binding constraint.** Over 30 epochs the feature
  loss goes 0.880 → 0.442 with the last ten epochs worth 0.006; at 150 epochs
  reconstruction error has only reached ~0.56. The curve is asymptotic, so the
  earlier plan — "train it longer on a GPU until reconstruction error drops
  under 0.2" — would have spent GPU hours to arrive at the same place.

## What it does not tell us

**The method is not yet usable at this operating point.** Losing 0.149
mAP50-95 (−39%) to save bandwidth is not a trade a deployment would take. Two
things are conflated in that number and this run cannot separate them:

1. **Capacity** — ruled out. Widening the latent buys almost nothing:

   | latent channels | wire bytes (zlib) | training loss | held-out output error |
   |---:|---:|---:|---:|
   | 8 | 3.9 KB | 0.291 | 0.1006 |
   | 16 | 7.1 KB | 0.281 | 0.1001 |
   | 32 | 13.9 KB | 0.247 | 0.0968 |

   Four times the bytes moves the error 4%. Read the middle column with the
   right one, though: the *training* loss improves substantially with width
   while held-out error does not. That is a generalisation gap, not a rate
   constraint — the codec has enough channels to carry the features, and not
   enough data to learn which ones matter in general.

2. **Data** — the live suspect. 96 images is a toy training set for a codec.
   The full-COCO run in `notebooks/colab_validation.ipynb` (train2017 →
   val2017) is this same experiment with three orders of magnitude more data
   and a real held-out split, and it is the run that should happen next. A
   `sweep` over latent channels on 96 images would only re-measure the table
   above.

The mAP numbers are measured over images the codec partly trained on, so they
flatter the codec; the comparison *between* the two objectives is unaffected,
since both trained on the same 96 frames. The latent-width table above is
held-out.

## Two issues an earlier run surfaced (both handled)

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
