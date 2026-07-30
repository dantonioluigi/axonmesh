# Edge-first inference: the configuration that does win on bandwidth

[docs/validation.md](validation.md) establishes that compressing intermediate
features does not beat sending a JPEG frame. This is the measurement of what
does: **running a small model on the device and consulting the cloud only for
the frames it is unsure about.**

The saving does not come from a better codec. It comes from the frames that
never travel at all — a confident frame ships its detections, eleven bytes
each.

## Setup

- Edge: `yolo11n.pt`. Cloud: `yolo11m.pt`. Both public COCO weights.
- Data: `coco128`, 128 images, 320px, CPU.
- Escalated frames are shipped as JPEG q50 **and the cloud scores the decoded
  image**, not the pristine one. Charging for a codec without applying it is
  the single easiest way to flatter this comparison.
- Frame confidence is the *mean* detection confidence (see "the statistic
  matters" below); the threshold `conf_high` is the knob swept.

```bash
axonmesh cascade --edge yolo11n.pt --cloud yolo11m.pt \
    --data coco128.yaml --imgsz 320 --conf-high 0.6 --statistic mean
```

## Result: two curves, and one is above the other everywhere

The alternative to a cascade is not "send raw tensors" — it is "keep sending
every frame, just send a worse one". Both curves trade accuracy for bandwidth,
so the only question is which dominates.

| KB/frame | cascade mAP50-95 | JPEG-quality-only mAP50-95 |
|---:|---:|---:|
| 0.04 | **0.385** | — (no frame fits) |
| 1.25 | **0.378** | — |
| ~3.2 | **0.412** | 0.048 |
| ~5.0 | **0.440** | 0.152 |
| ~7.0 | **0.436** | 0.294 |
| ~8.3 | — | 0.390 |
| ~11.2 | 0.448 | 0.448 |

At every matched bandwidth the cascade returns between **1.5x and 8.6x** the
mAP of simply turning the JPEG quality down. The two curves meet only where
the cascade escalates everything, which is the same configuration.

The extreme point is the interesting one: **38 bytes per frame for 86% of the
cloud's accuracy** — three hundred times less bandwidth than the frame it
replaces. That is the edge model answering alone, and for many deployments it
is the whole product.

Why the cascade wins is worth being explicit about: on a frame it answers
locally, the edge sees the **original image**. Only escalated frames pay
compression damage. Turning down JPEG quality degrades every frame, including
the easy ones that needed nothing.

## Onsen: when the cameras are inside the cluster, the GPU soaks

*The GPU's hot spring — the accelerator soaks while the edge handles the easy
frames.* The name is a mnemonic; the number under it is the point: on the
reference run the large model does **52% fewer inferences** (`--no-drift`,
confirmed on a laptop CPU and a Tesla T4). The metaphor rests on that
measurement, not the other way round.

On an internal network the bandwidth argument evaporates — 5 KB against 11 KB
per frame is invisible on a datacenter link. What is not free inside a cluster
is the accelerator the large model runs on, and the cascade's other reading is
that **the large model only runs on escalated frames**.

`cascade` measures it rather than deriving it: both models are timed with the
same clock in the same process (with a CUDA synchronise when on GPU, so kernel
execution is measured rather than kernel launch). Same run as above — coco128
at 320, `conf_high=0.6`, mean statistic, drift off:

```
large-model compute: 15.5s on 62 frames (vs 32.2s if every frame escalated,
                     at this run's 250 ms/inference)
                     -> 52% of the large model's compute avoided
edge-model compute:  4.0s over all 129 frames (31 ms/frame)
```

The small model prices at ~1/8 of the large one per frame on this machine, and
it is reported *separately* on purpose: in the deployment this number is for,
the edge model runs on cheap CPU nodes in front of the accelerator, and
folding its seconds into the saving would compare watts on one machine with
watts on another.

### What the 52% is, and what it is not

The load-bearing quantity is the **inference count**: the large model runs on
48% of the frames instead of 100%, so 52% of its forward passes never happen.
That is what maps to a GPU bill — the number of large-model inferences per
second is what fills an accelerator and sets how many you buy — and it is
hardware-independent, because it is a routing decision, not a timing. At the
same input rate the expensive tier needs ~half the accelerator-seconds; on a
fixed GPU pool it serves ~twice the input rate.

It is **not** a claim that each request is half as fast. Per-request latency on
a GPU is dominated by batch size and utilisation, not by which of the two
models ran — at batch 1 both are launch-bound and time about the same, at large
batch the FLOP difference reappears. The saving is throughput and capacity (how
much accelerator you need for a given stream), which is the axis GPU cost
actually lives on; it is not a per-frame latency reduction, and the numbers
here do not pretend to be one.

Two more honest boundaries:

- **The absolute times are the measuring machine's**; only the share transfers
  (below).
- Serving cost is more than model forward time — preprocessing, queueing,
  copies. This measures the forward pass, the part that scales with the number
  of accelerators.

For the throughput saving to be real the small model has to run on cheaper
hardware than the accelerator it shields — a CPU tier in front of the GPU. Run
both on the same GPU and you have added the small model's load to the
accelerator instead of removing the large model's; it is still a net win
(yolo11n is ~4x fewer FLOPs than yolo11m, and it replaces yolo11m on 52% of
frames) but a smaller one, and a different calculation than the one above.

### Confirmed on a GPU

Rerun on a Tesla T4 (the Colab quickstart), the share holds and the
per-inference time collapses — which is the whole claim, demonstrated rather
than asserted. This is the same `--no-drift` configuration as the reference row
above, so the two machines are directly comparable:

| machine | escalation | large-model compute avoided | ms / inference |
|---|---:|---:|---:|
| laptop CPU | 48% | 52% | 250 |
| Tesla T4 | 48% | 52% | 9 |

The routing avoids **52% of the large model's work on both machines** — the
share does not move at all — while the per-inference cost drops ~28x. The edge
model on the T4 runs at 12 ms/frame, still a fraction of the large model: the
triage stays cheap relative to what it shields. This is the load-bearing
result behind "the saving is a decision, not a hardware trick": run it on the
accelerator you will deploy on and the *share* is what you keep, whatever the
milliseconds do.

With drift left on (a live deployment, more escalations) the same pair reads
47% (CPU) and 46% (T4) — less compute avoided because more frames escalate, and
still matched across the hardware. Pass `--no-drift` to reproduce the table
above, leave it on for a number closer to production.

## The statistic matters more than the thresholds

`AdaptivePolicy` reduces a frame to one confidence and thresholds it. The
default is the *minimum* detection confidence, which suits a station holding a
few known objects where every one matters. On a crowded scene it is close to a
constant — some box is always marginal — so the frame is never confident:

| statistic | escalated | KB/frame | mAP50-95 |
|---|---:|---:|---:|
| min (default) | 78% | 8.65 | 0.436 |
| q25 | 76% | 8.37 | 0.436 |
| **mean** | **68%** | **7.73** | **0.436** |

Same accuracy, and `min` pays 12% more bandwidth for it. `Cascade` therefore
takes the statistic as a parameter (`frame_confidence`) while the thresholds
stay in the policy: *how confident is this frame* is a question about the
scene, not about the routing rule.

## Choosing the threshold without labels

`conf_high` is compared against a detector's confidence score, and that score
is not a probability: 0.6 does not mean "right six times in ten", and the
mapping differs between models and shifts with the scene. A threshold picked by
intuition is a guess that does not transfer.

`axonmesh calibrate` removes the need for the score to be calibrated by
measuring what each threshold actually does. It runs both models over frames
from the deployment and asks, per frame, *would the cloud have disagreed?* —
which needs **no annotations at all**, only footage from the camera that will
be running. That is the distribution the threshold has to hold on, and the one
a site actually has.

```bash
axonmesh calibrate --edge yolo11n.pt --cloud yolo11m.pt \
    --images ./footage --imgsz 320 --statistic mean --max-kb 5
```

| threshold | KB/frame | agreement | escalated |
|---:|---:|---:|---:|
| 0.30 | 0.037 | 0.767 | 0% |
| 0.40 | 0.358 | 0.794 | 3% |
| 0.50 | 2.158 | 0.912 | 22% |
| **0.60** | **4.677** | **0.951** | **41%** |
| 0.70 | 6.741 | 0.980 | 59% |
| 0.90 | 10.486 | 1.000 | 94% |

Agreement is the share of the answer an always-escalate deployment would have
given, as symmetric F1 over IoU-matched boxes. Give it a ceiling (`--max-kb`)
and it returns the most faithful threshold that fits; give it a floor
(`--min-agreement`) and it returns the cheapest that clears it. A constraint
nothing satisfies raises rather than returning the closest miss — a returned
threshold implies its budget was met.

**It reproduces the labelled answer.** On the same frames the label-free sweep
selects 0.60, the threshold the mAP measurement above independently picked, and
its two readings bracket it the right way:

| | calibration (no labels) | mAP measurement (labelled) |
|---|---|---|
| threshold | 0.60 | 0.6 |
| KB/frame | 4.68 | 5.43 |
| escalated | 41% | 47% |
| quality kept | 0.951 agreement | 0.982 of cloud mAP |

Agreement reads slightly *below* mAP retention because it penalises every box
difference while mAP forgives some. For a routing decision that is the right
direction to be wrong in.

## Auditing the routing in production

Calibration is a snapshot of one afternoon's footage. Scenes drift with
season, weather and time of day, and a threshold measured then is a guess
again six months later — silently, because a misrouted frame produces no
error, only a worse answer nobody compares.

`--audit` closes that loop with the same label-free agreement the calibration
is built on: a fraction of the frames the edge would answer *locally* is
escalated as well, and the cloud's answer is compared with the edge's. The
rolling agreement is directly comparable with the number `calibrate` reported
for the chosen threshold, which is what `--audit-floor` should be set to:

```bash
axonmesh edge --model yolo11n.pt --images ./frames \
    --escalate-url http://kserve.internal/detector \
    --conf-high 0.6 --statistic mean \
    --audit 0.05 --audit-floor 0.95        # calibrate reported 0.951
```

```
audit: 4 confident frames re-checked against the cloud, rolling agreement 1.000 (calibrated floor 0.9)
```

When the rolling agreement sinks below the floor — judged over a window, not
on a single frame — the scene has moved and the warning says to recalibrate.
With `--metrics-port` the same number is a Prometheus gauge
(`axonmesh_edge_audit_agreement`, with `audit_stale` beside it), so it lives
in Grafana next to the per-mode frame and byte counters instead of only in a
summary printed after the fact.
The audited frames pay for the frames they ship, and the accounting charges
them honestly: on the 24-frame live run, a 25% audit rate cost ~200 KB over
the baseline cascade. That is the price of knowing the saving is still safe,
and it is a dial.

## Honest notes

- **The bar set before running was not met as written.** The criterion was
  "under 2 KB/frame within 2–3% of the cloud accuracy". At 1.25 KB the cost is
  16%, not 3%; the 2% cost arrives at 5.4 KB. The bar asked for a free lunch.
  What the data supports is the stronger and more useful claim above: one curve
  dominating another at every rate.
- The cascade at 7.7 KB scores marginally *below* the 5.4 KB configuration
  (0.436 vs 0.440). On 128 images that is inside the noise, but the mechanism
  is real and worth naming: escalating a frame replaces an edge answer computed
  on a clean image with a cloud answer computed on a compressed one, and for an
  easy frame that can be a downgrade. More escalation is not monotonically
  better.
- 128 images is a small validation set and coco128 is drawn from COCO train2017,
  which both models were trained on. The *relative* shape of the two curves is
  the load-bearing result; the absolute mAPs are optimistic for both.
- Drift detection is disabled in these runs (`warmup` set beyond the run), so
  the numbers measure routing alone. A live deployment adds drift escalations
  on top.
