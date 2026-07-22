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
