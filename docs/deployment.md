# Running it: two processes, then a cluster

`serve` and `edge` are two ordinary processes. They will run on a laptop, a
Jetson, docker-compose or systemd; Kubernetes is one backend, and the library,
the measurement commands and the wire protocol do not import it.

## Run it over the network

The cloud half runs as a service; the edge connects to it. The wire protocol
carries three payload kinds — serialised detections, quantised feature tensors,
and full JPEG frames — chosen per frame by the policy. FRAME uploads can be
queued as hard-frame samples with `serve --retrain-dir /retrain`.

A HELLO/ACK handshake settles what the two ends have to agree on, and that
depends on the **role**:

```bash
# split: two halves of one network. Identical weights are mandatory, because
# halves on different weights produce confidently wrong output and nothing
# else would catch it. Mismatches are rejected at connect time.
axonmesh serve --model yolo11l.pt --bottleneck bottleneck.pt --port 9095
axonmesh edge  --model yolo11l.pt --bottleneck bottleneck.pt \
    --host <cloud-host> --port 9095 --images path/to/frames

# cascade: two independent models. Differing weights are the entire point, so
# only the protocol, role and image size are compared.
axonmesh serve --model yolo11n.pt --escalate-to yolo11m.pt --port 9095
axonmesh edge  --model yolo11n.pt --host <cloud-host> --port 9095 \
    --images path/to/frames --cascade --statistic mean
```

Roles cannot be mixed: a split client talking to a cascade server fails at
connect, because the two ends would disagree about what the fingerprints mean.
Under `--cascade` an escalation always ships the frame — the cloud runs a
*different* model and cannot consume this one's activations, and the server
refuses feature payloads rather than answering with the wrong half.

`/metrics` shows the routing as it happens:

```
axonmesh_frames_total{mode="detections"} 11
axonmesh_frames_total{mode="frame"} 13
axonmesh_wire_bytes_total{mode="detections"} 286
axonmesh_wire_bytes_total{mode="frame"} 646028
```

## Deploy on Kubernetes

`deploy/` ships Dockerfiles for both halves and a Helm chart for the cloud
half. The images are model-agnostic (weights are provided at runtime, not
baked in), so a single build serves any checkpoint:

```bash
helm install detector deploy/helm/axonmesh-cloud \
    --set image.repository=ghcr.io/you/axonmesh-cloud \
    --set model.url=https://your-store/model.pt \
    --set model.sha256=$(sha256sum model.pt | cut -d' ' -f1) \
    --set bottleneck.url=https://your-store/bottleneck.pt
```

An initContainer downloads the checkpoint (or mount a PVC via
`model.existingClaim`); the pod exposes the wire port plus `/healthz` and
Prometheus `/metrics` (set `serviceMonitor.enabled=true` with the Prometheus
Operator). Point edge devices at the resulting Service DNS name.

**Set `model.sha256`.** `torch.load` unpickles the checkpoint, so whatever that
URL serves is code that runs in the pod. The weight fingerprint in the
handshake catches *mismatched* halves, but it is computed after loading —
too late to be a defence against a swapped file.

### One command, then a resource

```bash
helm install axonmesh-operator deploy/helm/axonmesh-operator
kubectl apply -f operator/examples/cascade.yaml
```

The chart brings the CRD, the RBAC and the controller; the images are published
by CI to `ghcr.io/dantonioluigi/axonmesh-{cloud,edge,operator}`, so nothing has
to be built first.

**What this does and does not give you.** If an object detector already runs
inside your cluster and clients POST it frames, installing this changes
nothing — the saving comes from work moving to the device, and the device has
to participate. The case it is for is *frames arriving from outside the
cluster*: cameras or gateways running `axonmesh edge`, answering locally when
confident and escalating when not. Then the cluster side is one resource, and
the wire drops by the escalation rate.

Scaling follows from that: the cloud half's load is the escalation traffic of
every edge pointed at it, which moves with the scenes those cameras look at
rather than with anything in the cluster. `cloud.autoscaling` hands the replica
count to an HPA; a fixed number is either waste or a queue, and which one it is
changes during the day.

### Operator (declarative)

For fleets, `operator/` provides a `SplitInference` custom resource and a kopf
controller that manages the cloud Deployment/Service and an edge-facing
ConfigMap for you:

```yaml
apiVersion: axonmesh.dev/v1alpha1
kind: SplitInference
metadata: { name: detector }
spec:
  model: { url: https://your-store/model.pt }
  bottleneck: { url: https://your-store/bottleneck.pt }
  cut: { mode: auto, auto: { bandwidthMbps: 50, fps: 10 } }
  cloud: { image: ghcr.io/you/axonmesh-cloud:0.5.0, replicas: 2 }
```

Declaring `escalateTo` makes it a **cascade** instead — the operator writes
`role=cascade` into the edge ConfigMap and passes `--escalate-to` to the cloud,
so the winning configuration is as declarative as the split one
([operator/examples/cascade.yaml](operator/examples/cascade.yaml)):

```yaml
spec:
  model: { url: https://store/yolo11n.pt, sha256: "..." }
  escalateTo: { url: https://store/yolo11m.pt, sha256: "..." }
  policy: { confHigh: 0.6, statistic: mean }   # from `axonmesh calibrate`
```

Set `sha256` on every download. `torch.load` unpickles the checkpoint, so the
URL is code that runs in the pod, and the initContainer refuses a digest
mismatch rather than starting on a file nobody vouched for.

`cut.mode: fixed` pins a layer; `auto` writes the budget into the edge
ConfigMap for the edge to plan against live (`axonmesh replan`). Install the
CRD + RBAC from `operator/manifests/`, run the operator (image in
`operator/Dockerfile`), and `kubectl apply` the resource. The reconcile logic
is pure and unit-tested; `deploy/kind/e2e.sh` exercises it end-to-end on a kind
cluster (also run in CI).
