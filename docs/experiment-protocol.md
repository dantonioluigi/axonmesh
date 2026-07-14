# Experiment protocol: is the backbone cut worth it?

## Hypothesis

For a YOLO11l detector on the kitting_v4 domain, shipping the INT8-quantised
backbone output (P3/P4/P5 pyramid) from edge to cloud uses less bandwidth than
shipping the JPEG frame, at a negligible mAP cost.

**Success criterion:** INT8 wire bytes < JPEG bytes at production quality, with
< 1–2 points of mAP50-95 lost. If raw INT8 already loses on size, the finding is
still useful: it quantifies the gap a learned bottleneck has to close.

## Protocol

1. **Inspect** — `yolosplit inspect --model teacher_best.pt`. Confirms the
   backbone/neck boundary (layer 10, `C2PSA` in YOLO11) and prices every
   candidate cut, since the wire set changes with the cut point: early cuts ship
   one tensor, the backbone cut ships three (P3/P4/P5 via skip connections).
2. **Measure bandwidth** — `yolosplit measure` on a representative sample of
   kitting_v4 validation images, JPEG quality set to what the production camera
   pipeline uses. Compare: JPEG (letterboxed at 640, i.e. the pixels the model
   actually sees), fp32, fp16, INT8, INT8+zlib.
3. **Measure accuracy** — `yolosplit evaluate` on the kitting_v4 validation
   split: ultralytics `val()` twice, unsplit baseline vs split with the INT8
   round-trip injected at the cut. Same dataloader, same NMS, same metrics.
4. **Decide** — fill the results table in the README. Compression ratio and
   ΔmAP are the two numbers that decide whether to build the next stage.

## Threats to validity

- **Tensor > JPEG risk.** The P3 tap alone is `C×80×80` at 640 input; without
  quantisation the wire set dwarfs a JPEG. Quantisation/bottlenecking is the
  core of feasibility, not an optimisation.
- **Skip connections.** A sequential slice of the module list produces silently
  wrong results; the splitter resolves the graph instead and the test suite
  checks bit-exactness of the unquantised split.
- **Off-device numbers.** Byte counts are hardware-independent; latency and
  edge GPU load are not. Anything time-related must be re-measured on the Orin.
- **JPEG fairness.** Compare against the letterboxed inference-resolution JPEG,
  not the full camera frame, and state the quality factor with the result.

## Where this goes next (out of scope here)

- Learned bottleneck (autoencoder) at the cut to push past INT8.
- Adaptive policy: confidence high → ship detections only; medium → ship
  features; drift/low confidence → ship the full frame and enqueue it for the
  retraining dataset (kitting_v5 built from hard cases).
- Dynamic split point driven by live bandwidth/GPU metrics; packaged as a
  Kubernetes operator with GitOps rollout once the numbers justify it.
