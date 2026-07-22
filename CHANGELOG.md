# Changelog

## Unreleased

**Usable as a library, not only as a CLI.** `py.typed` ships, so a consumer's
type checker sees the annotations that were already there instead of treating
every symbol as `Any`; `cascade`, `allocate` and `calibrate` are exported from
the package root.

**`axonmesh calibrate` picks the routing threshold by measuring it, without
labels.** `conf_high` is compared against a detector confidence score, which is
not a probability -- 0.6 does not mean "right six times in ten", and the mapping
moves between models and scenes, so a threshold chosen by intuition does not
transfer. Rather than calibrating the score, this measures what each threshold
does: run both models over frames from the deployment and ask, per frame, would
the cloud have disagreed. Agreement is symmetric F1 over IoU-matched boxes, so
it needs no annotations at all -- only footage from the camera that will run,
which is both the distribution that matters and the one a site actually has.

`--max-kb` returns the most faithful threshold that fits; `--min-agreement` the
cheapest that clears the floor; a constraint nothing satisfies raises rather
than returning the closest miss, because a returned threshold implies its budget
was met. On public frames the label-free sweep selects 0.60 -- the same
threshold the labelled mAP measurement picked -- and reads slightly more
conservatively than mAP retention (0.951 vs 0.982), which is the right direction
for a routing decision to err in.

**Edge-first inference (`axonmesh cascade`, `axonmesh.cascade`) — the
configuration that does win on bandwidth.** A small model runs on the device
and the cloud is consulted only for frames it is unsure about; a confident
frame ships its detections, eleven bytes each. Measured against the honest
alternative, which is not "raw tensors" but "keep sending every frame, just
send a worse one":

  KB/frame   cascade   JPEG-quality-only
      0.04     0.385   -- (no frame fits)
      ~3.2     0.412   0.048
      ~5.0     0.440   0.152
      ~7.0     0.436   0.294
     ~11.2     0.448   0.448

The cascade curve is above the other at every matched bandwidth, by 1.5x to
8.6x mAP, and its cheapest point ships 38 bytes/frame for 86% of the cloud's
accuracy. The reason is asymmetric damage: the edge answers easy frames on the
*original* image, while turning JPEG quality down degrades every frame
including the ones that needed nothing. See docs/cascade.md.

- Escalated frames go through a real JPEG round-trip before the cloud scores
  them (`jpeg_roundtrip`). Charging for a codec without applying it is what
  made an earlier reading of this same experiment show 0.556 instead of 0.448.
- `AdaptivePolicy.decide_confidence` splits the thresholds from the statistic
  they are applied to, and `Cascade` takes `frame_confidence`. The default,
  the *minimum* detection confidence, suits a few known objects and is close to
  a constant on a crowded scene — some box is always marginal. On coco128 it
  buys the same mAP as the mean for 12% more bandwidth.

## Earlier in this cycle

**The bottleneck is trained against the head output, not just the features**
(`train_bottleneck(task_weight=...)`, `--task-weight`, default 0.5). Half the
loss is now the error on the model's own output, with the gradient taken back
through the frozen tail — `SplitRunner.cloud(grad=True)` exists to make that
expressible; inference keeps the `no_grad` path. On yolo11n / coco128 at 320,
150 epochs, at an identical 3.8 KB on the wire: **mAP50-95 0.170 → 0.236**
(`docs/validation.md`). Both sides of that pair trained on frames inside the
evaluation set, so the two absolute numbers are flattered and only their
difference is load-bearing; the uncontaminated figures are further down.

- **Reconstruction error is no longer the quality axis.** The task-aware codec
  reconstructs *worse* while scoring better on everything measured downstream:
  it stops spending capacity on activations no prediction depends on. `sweep`
  now draws its Pareto front against `output_error` — the relative error a
  codec induces on the model output, one extra pass through the cloud half —
  and reports reconstruction error only as a diagnostic. Anything that ranked
  configurations by reconstruction was ranking the wrong thing.
- `axonmesh.train.output_error(runner, bottleneck, paths, ...)` is the metric
  as a public function: an affordable stand-in for mAP when a full validation
  run per configuration is not.
- The sweep keeps `task_weight=0` by default: it ranks configurations by size,
  and the backward pass through the tail would triple its cost for a ranking
  it does not use. Retrain the winning configuration with the task loss on.

Measured and worth knowing before spending GPU hours: widening the latent does
not buy accuracy back. Four times the wire (8 → 32 channels) moves held-out
output error 0.101 → 0.097 while the training loss falls 0.291 → 0.247 — a
generalisation gap, not a rate constraint.

That reading pointed at training data, and training data turned out to be worth
about a tenth: 50x more images at matched compute moves held-out output error
0.0959 → 0.0866. Redistributing the same bytes across wire levels
(`axonmesh allocate`) is worth another 4%. Both real, neither the size of the
problem.

The size of the problem, measured with no train/eval overlap: a JPEG frame
costs 11.3 KB and gets the *full* 0.385 mAP50-95, because the cloud then runs
the unsplit model. The codec gets 0.154 at 3.8 KB and 0.195 at 14.1 KB -- at a
JPEG-comparable rate it ships more bytes and returns half the accuracy, and
3.7x the wire is worth 0.041 mAP. `inspect` shows the structural reason: across
all 23 cuts of YOLO11n the smallest wire set is 100 KB as INT8. No cut of this
network is smaller than the coded image it came from. Split inference at these
cuts does not win on bandwidth; docs/validation.md says what it does win.

Not shipped, recorded so it is not re-tried blindly: a deeper codec (GroupNorm,
residual blocks, PixelShuffle decoder) came out slightly *behind* the plain
two-convolution stack at equal budget, and would have put that depth in the
encoder — which runs on the edge device. Worth revisiting only with enough GPU
budget to train it to convergence.

## 0.8.0 — 2026-07-21

**Renamed to `axonmesh`.** The package is now `axonmesh`, the CLI is `axonmesh`,
the Helm chart is `axonmesh-cloud`, the Prometheus metrics are prefixed
`axonmesh_` and the CRD API group is `axonmesh.dev/v1alpha1`. Update imports
(`yolosplit`/`splitflow` -> `axonmesh`), any scraping rules, and re-apply the
CRD — the old `split.dev` group is gone, not aliased. The project is a generic
split-inference framework, not a YOLO demo, and the name now says so.

- Generic `FxAdapter` (`axonmesh.adapters.fx`): recovers the graph of any
  traceable `nn.Module` with `torch.fx` and interprets a span of it, so the
  planner, codecs, wire protocol and facade apply to architectures the project
  was never written for — verified bit-identical on ResNet-18, MobileNetV3 and
  ViT-B/16 alongside YOLO11. Registered as a *fallback*: `register_adapter` now
  sorts fallbacks last, so a purpose-built backend added later still wins
  (without that, the catch-all would have made plugins pointless).
  `enumerate_cuts` is adapter-based too, so planning is no longer YOLO-only.
- Adapter contract (`axonmesh.adapters`): `ModelAdapter` (`graph`,
  `default_cut`, `probe_shapes`, `run_span`) plus a detector registry, with the
  ultralytics graph reader moved behind `UltralyticsAdapter`. `SplitRunner` runs
  on any adapter; layer caching is derived from the graph instead of
  ultralytics' `.save`. Adding a model family is a registration, not a fork.
- `SplitModel` facade: `split()` to configure cut and codec, `plan()` to choose
  a cut from a bandwidth/FPS budget, `edge()`/`cloud()`/`run()` to execute, and
  `deploy()` to emit the `SplitInference` custom resource.
- Unified benchmark (`axonmesh benchmark`, `axonmesh.benchmark`): one command
  reports the numbers a deployment decision needs together — per-stage latency
  (preprocess / edge half / wire codec / cloud half), FPS, bytes on the wire vs
  the JPEG baseline, the link rate that implies, optional power (Jetson INA3221
  rails) and, with `--data`, the mAP cost. Markdown table plus JSON.

- `train-bottleneck` now shows a per-epoch tqdm progress bar with the running
  loss (`train_bottleneck(..., progress=True)`, the default), so a long run is
  no longer a silent black box; the sweep and tests pass `progress=False`.
- SplitInference Kubernetes operator (`operator/`): a `SplitInference` CRD
  (`axonmesh.dev/v1alpha1`) and a kopf controller that renders and reconciles the
  cloud-half Deployment + Service and an edge-facing ConfigMap (the resolved
  cut — fixed, or a bandwidth/FPS budget for the edge to plan against live —
  plus policy thresholds). Children carry ownerReferences, so deleting the CR
  garbage-collects them. The reconcile logic is pure (spec in, manifests out)
  and unit-tested without a cluster; a kind e2e (`deploy/kind/e2e.sh`, run in
  CI) drives create/update/delete against a real API server. Ships CRD, RBAC,
  operator Deployment and a small operator image (no torch — it renders
  manifests, it does not load the model).

## 0.5.0 — 2026-07-20

The split becomes real — and Kubernetes-ready.

- Wire protocol v1 (`splitflow.protocol`): framed TCP messages, INT8 tensor
  packing, and a HELLO/ACK handshake that exchanges model + bottleneck weight
  fingerprints and the cut point — mismatched halves fail loudly at connect
  time instead of silently producing wrong detections.
- `splitflow serve` (`CloudServer`): the cloud half as a long-running service
  with `/healthz` and dependency-free Prometheus `/metrics`, an optional
  retraining queue for FRAME uploads, and a **pluggable postprocess** — the
  protocol carries opaque result bytes, so a different head/task can be served
  by swapping one function (YOLO NMS is just the default codec).
- `splitflow edge` (`EdgeClient`, `run_edge`): local inference + adaptive
  policy against a live server; same per-frame accounting as the offline
  `stream` simulator, so simulated and real numbers are directly comparable.
- `deploy/`: Dockerfiles for both halves (multi-arch friendly, model-agnostic
  images) and a Helm chart for the cloud half — `helm install` with a model
  URL is enough: initContainer download, health probes, Service, optional
  ServiceMonitor.

## 0.4.0 — 2026-07-14

- Bottleneck sweep (`splitflow sweep`, `splitflow.sweep`): trains one
  bottleneck per (latent channels × stride) configuration and prices each on
  the same frames — serialised INT8 latent bytes (plain and zlib), the JPEG
  baseline, feature reconstruction error — then marks the Pareto front.
  Indivisible strides are skipped, not fatal. This is the Phase 1 tooling:
  run it on GPU, pick the smallest Pareto config, validate its mAP with
  `evaluate --bottleneck`.

## 0.3.0 — 2026-07-14

- Cut planner (`splitflow plan`, `splitflow.planner`): given link bandwidth and
  target FPS, prices every candidate cut (wire bytes per transport, share of
  parameters run on the edge) and picks the cut that fits the budget with the
  least edge compute. Returns a clear "nothing fits" verdict when raw feature
  shipping cannot meet the link (i.e. the bottleneck is required).

## 0.2.0 — 2026-07-14

The adaptive layer: learned bottleneck + transmission policy.

- `Bottleneck`/`LevelCodec`: per-level convolutional autoencoder at the cut
  (configurable latent channels and stride), trained by feature distillation
  with the detector frozen and simulated INT8 noise on the latents
  (`splitflow train-bottleneck`). Checkpoints load with `weights_only=True`.
- `BottleneckTransport`: encode → INT8 → decode, byte counts on the serialised
  latents; `splitflow evaluate --bottleneck` measures its mAP cost.
- Adaptive policy (`AdaptivePolicy`, `ConfidenceEMADrift`): per frame ships
  serialised detections (11 bytes each), features, or the full JPEG (enqueued
  for retraining) based on minimum detection confidence and a drift signal.
- `splitflow stream`: simulates the adaptive stream over a directory of frames
  and reports bytes saved vs always-JPEG plus the retraining queue.
- Community files: issue/PR templates, CODEOWNERS, Dependabot, maintenance
  playbook (`docs/maintenance.md`).

## 0.1.0 — 2026-07-14

Initial release: the feasibility probe.

- Graph-aware splitter for ultralytics detection models: resolves `m.f`/`m.i`
  wiring, computes the wire set for any cut, bit-exact split inference
  (edge backbone + cloud neck/head) with skip connections handled.
- Affine INT8 quantisation (per-tensor and per-channel) with a serialised wire
  format whose byte counts include scale/zero-point overhead.
- Transports: raw fp32/fp16, INT8, INT8+zlib.
- Bandwidth measurement: JPEG at production quality vs wire set, per frame and
  aggregated.
- End-to-end mAP comparison via a transparent patch of ultralytics `val()`.
- CLI: `splitflow inspect | measure | evaluate`.
