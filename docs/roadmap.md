# Roadmap

This used to be a gated plan toward a generic split-computing platform, each
phase unlocked by the one before. Phase 1 was "validate the numbers", and
validating them refuted the premise the later phases stood on. The plan is
rewritten around what the measurements support.

Nothing below is listed as done because it was implemented — only because it
was measured, with a link to where.

## Done

**The splitter.** Graph-aware cut selection, bit-identical split inference,
INT8 wire, per-cut byte accounting. Any traceable `nn.Module` through the
`torch.fx` backend; YOLO11, ResNet-18, MobileNetV3 and ViT-B/16 verified
([architecture.md](architecture.md)).

**The measurement layer.** `evaluate`, `measure`, `benchmark`, `sweep`, `plan`,
`replan`, `allocate` — bytes, mAP, per-stage latency, power, and a Pareto front
over codec configurations. This is the part that produced every result in the
repo, including the unwelcome ones.

**The finding that redirected the project.** Compressing intermediate features
loses to sending a JPEG frame, and no cut of the network is smaller than the
image it came from. Longer training, 50x the data, 4x the latent width and a
measured bit allocation were each tried and quantified; none close the gap
([validation.md](validation.md)).

**Edge-first inference.** A small model answers locally and escalates what it is
unsure about: 1.5–8.6x the mAP of the obvious alternative at matched bytes, and
38 bytes/frame for 86% of the cloud's accuracy ([cascade.md](cascade.md)).

**Label-free threshold calibration.** The routing threshold is measured against
unlabelled deployment footage rather than guessed, and reproduces the labelled
choice ([cascade.md](cascade.md)).

**Deployment.** Wire protocol v2 with role negotiation, `serve`/`edge`,
Prometheus metrics, two Helm charts, a `SplitInference` operator with
autoscaling, and a kind e2e that runs the real image
([deployment.md](deployment.md)).

## Next, in order

1. **A second task.** Everything measured here is object detection. The protocol
   carries opaque result bytes and the server's postprocess is one function, so
   segmentation or classification *should* be a plugin — but "should be" is a
   design claim, not a result. It becomes one when a second task runs end to end.

2. **Latency, on the device.** Byte counts are hardware-independent; every
   latency number here was taken on a laptop CPU. A cascade's value includes
   response time, and that number does not exist yet for a Jetson or a Pi.

3. **Drift in the loop.** `ConfidenceEMADrift` escalates when confidence sinks,
   and every measurement here disabled it to isolate routing. A deployment
   running for weeks is exactly where it matters and where it is unmeasured.

4. **Calibration that ages.** The threshold is measured once, against footage
   from the site. Scenes change with season, weather and time of day, and
   nothing currently notices that the calibration has gone stale.

## Tried, measured, dropped

Kept so they are not re-attempted on intuition.

- **A deeper codec** (GroupNorm, residual blocks, PixelShuffle decoder) came out
  slightly *behind* the plain two-convolution stack at equal budget, and would
  have put that depth in the encoder — which runs on the edge device.
- **Per-channel INT8 quantisation** of the latent: 0% of the error, 8% more
  bytes. Quantisation is not where the accuracy goes; the autoencoder is.
- **Reconstruction error as the quality axis.** It moves independently of
  accuracy and sometimes opposite to it. Replaced by the error a codec induces
  on the model's own output.
- **A retraining-orchestration phase**, in the original plan and out of scope
  for this project.

## The rule this runs on

A number is only reported against the same baseline as the number it is traded
against. Bandwidth measured against JPEG while accuracy is measured against the
unsplit model is how a losing design looks like a winning one. That mistake hid
the central result here for months, and it recurred three more times during the
work that followed — see [experiment-protocol.md](experiment-protocol.md).
