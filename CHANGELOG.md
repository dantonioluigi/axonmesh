# Changelog

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
