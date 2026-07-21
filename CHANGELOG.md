# Changelog

## Unreleased

- Unified benchmark (`yolosplit benchmark`, `yolosplit.benchmark`): one command
  reports the numbers a deployment decision needs together — per-stage latency
  (preprocess / edge half / wire codec / cloud half), FPS, bytes on the wire vs
  the JPEG baseline, the link rate that implies, optional power (Jetson INA3221
  rails) and, with `--data`, the mAP cost. Markdown table plus JSON.

- `train-bottleneck` now shows a per-epoch tqdm progress bar with the running
  loss (`train_bottleneck(..., progress=True)`, the default), so a long run is
  no longer a silent black box; the sweep and tests pass `progress=False`.
- SplitInference Kubernetes operator (`operator/`): a `SplitInference` CRD
  (`split.dev/v1alpha1`) and a kopf controller that renders and reconciles the
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

- Wire protocol v1 (`yolosplit.protocol`): framed TCP messages, INT8 tensor
  packing, and a HELLO/ACK handshake that exchanges model + bottleneck weight
  fingerprints and the cut point — mismatched halves fail loudly at connect
  time instead of silently producing wrong detections.
- `yolosplit serve` (`CloudServer`): the cloud half as a long-running service
  with `/healthz` and dependency-free Prometheus `/metrics`, an optional
  retraining queue for FRAME uploads, and a **pluggable postprocess** — the
  protocol carries opaque result bytes, so a different head/task can be served
  by swapping one function (YOLO NMS is just the default codec).
- `yolosplit edge` (`EdgeClient`, `run_edge`): local inference + adaptive
  policy against a live server; same per-frame accounting as the offline
  `stream` simulator, so simulated and real numbers are directly comparable.
- `deploy/`: Dockerfiles for both halves (multi-arch friendly, model-agnostic
  images) and a Helm chart for the cloud half — `helm install` with a model
  URL is enough: initContainer download, health probes, Service, optional
  ServiceMonitor.

## 0.4.0 — 2026-07-14

- Bottleneck sweep (`yolosplit sweep`, `yolosplit.sweep`): trains one
  bottleneck per (latent channels × stride) configuration and prices each on
  the same frames — serialised INT8 latent bytes (plain and zlib), the JPEG
  baseline, feature reconstruction error — then marks the Pareto front.
  Indivisible strides are skipped, not fatal. This is the Phase 1 tooling:
  run it on GPU, pick the smallest Pareto config, validate its mAP with
  `evaluate --bottleneck`.

## 0.3.0 — 2026-07-14

- Cut planner (`yolosplit plan`, `yolosplit.planner`): given link bandwidth and
  target FPS, prices every candidate cut (wire bytes per transport, share of
  parameters run on the edge) and picks the cut that fits the budget with the
  least edge compute. Returns a clear "nothing fits" verdict when raw feature
  shipping cannot meet the link (i.e. the bottleneck is required).

## 0.2.0 — 2026-07-14

The adaptive layer: learned bottleneck + transmission policy.

- `Bottleneck`/`LevelCodec`: per-level convolutional autoencoder at the cut
  (configurable latent channels and stride), trained by feature distillation
  with the detector frozen and simulated INT8 noise on the latents
  (`yolosplit train-bottleneck`). Checkpoints load with `weights_only=True`.
- `BottleneckTransport`: encode → INT8 → decode, byte counts on the serialised
  latents; `yolosplit evaluate --bottleneck` measures its mAP cost.
- Adaptive policy (`AdaptivePolicy`, `ConfidenceEMADrift`): per frame ships
  serialised detections (11 bytes each), features, or the full JPEG (enqueued
  for retraining) based on minimum detection confidence and a drift signal.
- `yolosplit stream`: simulates the adaptive stream over a directory of frames
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
- CLI: `yolosplit inspect | measure | evaluate`.
