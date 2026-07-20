# Handoff — yolo-split-computing

Snapshot for whoever picks this up next. Living state, not documentation;
the README and `docs/` are the durable references.

## Where things stand (2026-07-20)

The project has grown from a feasibility probe into a **deployable split
inference system**. Current version **0.5.0**: the split runs for real over the
network, with a wire protocol, a cloud service, Docker images and a Helm chart.

Phases completed (see `docs/roadmap.md` for the full gated plan):

- **0.1.0** — graph-aware splitter, bit-exact split inference, INT8 wire,
  bandwidth/mAP measurement. Finding: raw INT8 at the backbone cut ships ~30x
  more than JPEG → a learned bottleneck is mandatory.
- **0.2.0** — learned bottleneck (per-level autoencoder, feature distillation),
  adaptive transmission policy + offline stream simulator with retraining queue.
- **0.3.0** — cut planner: pick the split point from a bandwidth/FPS budget.
- **0.4.0** — bottleneck sweep: bytes-vs-mAP Pareto tooling.
- **0.5.0** — real network split: wire protocol v1, `serve`/`edge`, Docker +
  Helm. **← current**

## What 0.5.0 added

- `src/yolosplit/protocol.py` — framed TCP messages, INT8 tensor packing, and a
  HELLO/ACK handshake exchanging model+bottleneck weight **fingerprints** and
  the cut point. Mismatched halves are rejected at connect time (the classic
  split-computing failure: two halves on different weights, silently wrong).
  The wire carries *opaque* result bytes, so it is not tied to detection.
- `src/yolosplit/server.py` — `CloudServer`: the cloud half as a service.
  Dependency-free Prometheus `/metrics` + `/healthz`, optional `--retrain-dir`
  queue for FRAME uploads, and a **pluggable `postprocess`** (YOLO NMS is only
  the default codec — swap it to serve a different head/task).
- `src/yolosplit/edge.py` — `EdgeClient` / `run_edge`: local inference +
  adaptive policy against a live server; same per-frame accounting as the
  offline `stream` simulator.
- `deploy/` — `Dockerfile.cloud`, `Dockerfile.edge` (model-agnostic, weights at
  runtime; edge image builds amd64/arm64), and Helm chart `yolosplit-cloud`
  (initContainer model download, health probes, Service, optional
  ServiceMonitor).
- CLI now: `inspect | measure | evaluate | train-bottleneck | stream | plan |
  sweep | serve | edge`.

## Genericity (not just YOLO / not just Jetson)

Deliberate seams, documented in the README "Not just YOLO / not just Jetson":

- **Model** — splitter reads any ultralytics detection graph via `m.f`/`m.i`
  wiring; the cut is derived, not hardcoded. Other heads plug in via the
  server's `postprocess`.
- **Edge device** — "edge" is any host that runs the backbone and speaks the
  protocol; the edge image is plain Python (arm64/amd64). Jetson is the
  reference target, not a dependency.
- **Transport** — INT8 / bottleneck / raw are pluggable `Transport` objects.

## Verified this session

- `133 passed`, coverage 98% (`.venv/bin/python -m pytest`).
- Fixed a real bug: ultralytics moved `non_max_suppression` from `utils.ops`
  to `utils.nms` around 8.4; the server thread crashed on FEATURES/FRAME and
  the edge saw a closed connection. Import is now resilient across both.
- `helm lint` clean; `helm template` renders valid YAML.
- Live end-to-end smoke: `serve` + `edge` over real TCP streamed 5 frames,
  exposed Prometheus counters, and the fingerprint handshake rejected
  mismatched weights. (Compression numbers need the trained model + bottleneck;
  the smoke used random weights, so bandwidth savings were not the point.)

## ⚠️ Compliance: the old work reference is still reachable on GitHub

The history rewrite cleaned `main` and all active branches, but the old commit
`835e662` (message: *"...frames with the production teacher..."*, plus a
dataset name) survives on GitHub via **pull-request head refs**
(`refs/pull/1,2,3,4,5/head`) — those PRs were opened against the pre-rewrite
`main`. GitHub keeps PR refs **permanently**; deleting branches does not remove
them (verified: after deleting the stale branches, `git ls-remote origin
'refs/pull/*/head'` still resolves histories containing `835e662`). Anyone can
reach it via a closed PR's "Commits" tab or `git fetch origin refs/pull/N/head`.

Options (a user decision — do not action autonomously):
1. **Recreate the repo** — the only self-serve way to fully purge it. The repo
   is young; push clean `main` + tags to a fresh repo. Cost: loses PR/issue
   history, stars, watchers.
2. **GitHub Support** — ask them to purge stale/unreachable refs. Keeps history.
3. **Accept it** — not on any branch or default view; reachable only by someone
   deliberately inspecting old PR refs. Low but non-zero exposure.

The 3 open Dependabot PRs (#1–#3) were also branched off the old base, so they
carry `835e662` too; recreating the repo or closing+recreating them clears it.

## Git state — READ BEFORE PUSHING

- **Push only outside working hours** (standing constraint from the owner).
- `feat/network-split` (0.5.0) is **pushed** (`6f229fb` on origin). No PR is
  open for it yet — open one against `main`.
- PRs still to open + merge into `main` (still at `76b9f65`, 0.4.0), via the
  web UI (`gh` not installed, no API token here):
  - `docs/k8s-roadmap` — the roadmap doc.
  - `feat/network-split` — the 0.5.0 code.
- Stale branches `feat/cut-planner` and `feat/bottleneck-sweep` (merged via
  #4/#6) were **deleted** this session, local and remote.
- `main` history was cleaned of private project names earlier; keep it that
  way — generic model/dataset placeholders only, no internal names.
- Commit identity is repo-local: `Luigi D'Antonio
  <265034275+dantonioluigi@users.noreply.github.com>`. Co-author trailer for
  Claude is allowed **in this repo only**.

## Next steps

1. Push `feat/network-split` (off-hours) and merge both open PRs into `main`.
2. **Real numbers** (Phase 1 gate, needs a GPU): train the bottleneck and run
   `yolosplit sweep` on a real dataset for the bytes-vs-mAP Pareto, then
   `evaluate --bottleneck` for the mAP cost. Put the table in the README.
3. **P4 — live re-planning**: feed measured bandwidth/GPU load into the planner
   with hysteresis (simulate first via a scripted bandwidth trace).
4. **P5 — the Kubernetes operator**: `SplitInference` CRD + kopf controller
   that re-runs the planner and patches the edge config; kind e2e in CI.
5. **P6 — retraining loop**: drift-driven `batch/v1` Jobs + GitOps promotion.

## Environment notes

- venv at `.venv` (torch 2.4.1+cpu, ultralytics 8.4.95). It broke once when the
  repo directory was moved; if the venv shebangs point at a stale path,
  `pip install -e ".[dev]"` fixes it. Invoke as `.venv/bin/python -m pytest` or
  `.venv/bin/yolosplit ...`.
- `helm` is available locally; `gh` CLI is not (open PRs via the web UI).
- The sandbox mangles shell `&` backgrounding and `$VAR` across a backgrounded
  process; use a proper background runner for long-lived processes.
