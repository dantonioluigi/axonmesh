"""Pure rendering: a SplitInference spec -> the Kubernetes objects it implies.

No kopf, no cluster, no I/O — just dicts in, dicts out — so the operator's core
decision (what resources should exist for this CR) is unit-testable in CI
without a Kubernetes API. The kopf handlers in :mod:`handlers` are a thin layer
that applies whatever these functions return.

A SplitInference owns three objects:

- a **ConfigMap** the edge devices read: the cut (fixed, or a budget for the
  edge to plan against live) and the policy thresholds;
- a **Deployment** running the cloud half (the model-agnostic cloud image,
  weights pulled at runtime);
- a **Service** exposing the wire and metrics ports.

Everything derives deterministically from the spec, so re-rendering after a
spec change and diffing against the live objects is the whole reconcile.
"""

from __future__ import annotations

import shlex
from typing import Any

API_GROUP = "axonmesh.dev"
API_VERSION = "v1alpha1"
KIND = "SplitInference"
_MANAGED_BY = {"app.kubernetes.io/managed-by": "splitinference-operator"}


class SpecError(ValueError):
    """The SplitInference spec is missing a field or is internally inconsistent."""


def _labels(name: str) -> dict[str, str]:
    return {"app.kubernetes.io/name": "axonmesh-cloud", "axonmesh.dev/instance": name} | _MANAGED_BY


def owner_reference(name: str, uid: str) -> dict[str, Any]:
    """OwnerReference so the children are garbage-collected with the CR."""
    return {
        "apiVersion": f"{API_GROUP}/{API_VERSION}",
        "kind": KIND,
        "name": name,
        "uid": uid,
        "controller": True,
        "blockOwnerDeletion": True,
    }


def resolve_cut(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalise ``spec.cut`` into the config the edge/cloud consume.

    ``fixed`` pins a layer index; ``auto`` hands the edge a bandwidth/FPS budget
    to plan against live (see :mod:`axonmesh.replanning`). Unknown modes raise.
    """
    cut = spec.get("cut", {"mode": "auto"})
    mode = cut.get("mode", "auto")
    if mode == "fixed":
        if "fixed" not in cut:
            raise SpecError("cut.mode=fixed requires cut.fixed")
        return {"mode": "fixed", "cut": int(cut["fixed"])}
    if mode == "auto":
        auto = cut.get("auto", {})
        return {
            "mode": "auto",
            "bandwidthMbps": float(auto.get("bandwidthMbps", 50)),
            "fps": float(auto.get("fps", 10)),
            "transport": auto.get("transport", "int8"),
        }
    raise SpecError(f"unknown cut.mode {mode!r} (want 'fixed' or 'auto')")


def resolve_role(spec: dict[str, Any]) -> str:
    """``cascade`` when an escalation model is declared, ``split`` otherwise.

    The two are not interchangeable and the edge has to know which it is in:
    under split the handshake demands identical weights, under cascade it
    demands they differ. Deriving the role from the presence of
    ``spec.escalateTo`` keeps a CR from declaring one and configuring the other.
    """
    return "cascade" if spec.get("escalateTo", {}).get("url") else "split"


def render_configmap(name: str, spec: dict[str, Any], owner: dict | None = None) -> dict[str, Any]:
    """The edge-facing config: cut decision, role and policy thresholds."""
    policy = spec.get("policy", {})
    data = resolve_cut(spec) | {
        "role": resolve_role(spec),
        "confHigh": float(policy.get("confHigh", 0.75)),
        "confLow": float(policy.get("confLow", 0.4)),
        "statistic": policy.get("statistic", "min"),
        "driftThreshold": float(policy.get("driftThreshold", 0.5)),
        "modelFingerprint": spec.get("model", {}).get("fingerprint", ""),
        "bottleneckFingerprint": spec.get("bottleneck", {}).get("fingerprint", ""),
    }
    meta: dict[str, Any] = {"name": f"{name}-edge-config", "labels": _labels(name)}
    if owner:
        meta["ownerReferences"] = [owner]
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": meta,
        # Stringify: ConfigMap values are strings.
        "data": {k: str(v) for k, v in data.items()},
    }


def _fetch(source: dict[str, Any], destination: str) -> list[str]:
    """Shell lines that download a checkpoint and refuse to trust it blindly.

    The URL is quoted because it comes from a custom resource, and a checkpoint
    is unpickled by torch.load — whatever that URL serves is code running in the
    pod. ``sha256`` is optional but the only thing that makes the download
    trustworthy; the handshake fingerprint is computed after loading, far too
    late to be a defence.
    """
    lines = [f"curl -fsSL -o {destination} {shlex.quote(str(source['url']))}"]
    if source.get("sha256"):
        expected = f"{source['sha256']}  {destination}"
        lines.append(f"echo {shlex.quote(expected)} | sha256sum -c -")
    return lines


def render_deployment(name: str, spec: dict[str, Any], owner: dict | None = None) -> dict[str, Any]:
    """The cloud half Deployment (model-agnostic image, weights at runtime)."""
    cloud = spec.get("cloud", {})
    model = spec.get("model", {})
    if not model.get("url"):
        raise SpecError("spec.model.url is required")
    image = cloud.get("image")
    if not image:
        raise SpecError("spec.cloud.image is required")

    cut = resolve_cut(spec)
    args = [
        "--model=/models/model.pt",
        "--host=0.0.0.0",
        "--port=9095",
        "--metrics-port=9090",
        f"--imgsz={int(cloud.get('imgsz', 640))}",
    ]
    if cut["mode"] == "fixed":
        args.append(f"--cut={cut['cut']}")
    bottleneck = spec.get("bottleneck", {})
    escalate = spec.get("escalateTo", {})
    fetches = _fetch(model, "/models/model.pt")
    if bottleneck.get("url"):
        fetches += _fetch(bottleneck, "/models/bottleneck.pt")
        args.append("--bottleneck=/models/bottleneck.pt")
    if escalate.get("url"):
        fetches += _fetch(escalate, "/models/escalate.pt")
        args.append("--escalate-to=/models/escalate.pt")
    init_cmd = "\n".join(["set -eu", *fetches])

    meta: dict[str, Any] = {"name": f"{name}-cloud", "labels": _labels(name)}
    if owner:
        meta["ownerReferences"] = [owner]
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": meta,
        "spec": {
            # Omitted entirely under autoscaling: a Deployment that keeps
            # setting replicas and an HPA that keeps changing them fight on
            # every reconcile, and the CR looks like it is being ignored.
            **(
                {}
                if cloud.get("autoscaling", {}).get("enabled")
                else {"replicas": int(cloud.get("replicas", 1))}
            ),
            "selector": {"matchLabels": _labels(name)},
            "template": {
                "metadata": {"labels": _labels(name)},
                "spec": {
                    "initContainers": [
                        {
                            "name": "fetch-model",
                            "image": "curlimages/curl:8.10.1",
                            "command": ["sh", "-c", init_cmd],
                            "volumeMounts": [{"name": "models", "mountPath": "/models"}],
                        }
                    ],
                    "containers": [
                        {
                            "name": "cloud",
                            "image": image,
                            "args": args,
                            "ports": [
                                {"name": "wire", "containerPort": 9095},
                                {"name": "metrics", "containerPort": 9090},
                            ],
                            "livenessProbe": {
                                "httpGet": {"path": "/healthz", "port": "metrics"},
                                "initialDelaySeconds": 20,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/healthz", "port": "metrics"},
                                "initialDelaySeconds": 5,
                            },
                            "volumeMounts": [{"name": "models", "mountPath": "/models"}],
                        }
                    ],
                    "volumes": [{"name": "models", "emptyDir": {}}],
                },
            },
        },
    }


def render_service(name: str, spec: dict[str, Any], owner: dict | None = None) -> dict[str, Any]:
    """The cloud Service exposing the wire and metrics ports."""
    meta: dict[str, Any] = {"name": f"{name}-cloud", "labels": _labels(name)}
    if owner:
        meta["ownerReferences"] = [owner]
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": meta,
        "spec": {
            "selector": _labels(name),
            "ports": [
                {"name": "wire", "port": 9095, "targetPort": "wire"},
                {"name": "metrics", "port": 9090, "targetPort": "metrics"},
            ],
        },
    }


def render_hpa(name: str, spec: dict[str, Any], owner: dict | None = None) -> dict[str, Any] | None:
    """A HorizontalPodAutoscaler for the cloud half, when one is asked for.

    The cloud half's load is the escalation traffic of every edge pointed at
    it, which moves with the scenes those cameras are looking at rather than
    with anything inside the cluster. A fixed replica count is therefore either
    wasteful or a queue, and which one it is changes during the day.

    Returns ``None`` when autoscaling is off, so ``replicas`` stays the
    Deployment's own field — setting both would have the two fight, with the
    HPA winning and the CR appearing to be ignored.
    """
    autoscaling = spec.get("cloud", {}).get("autoscaling", {})
    if not autoscaling.get("enabled"):
        return None
    minimum = int(autoscaling.get("minReplicas", 1))
    maximum = int(autoscaling.get("maxReplicas", 10))
    if maximum < minimum:
        raise SpecError(
            f"cloud.autoscaling.maxReplicas ({maximum}) is below minReplicas ({minimum})"
        )
    meta: dict[str, Any] = {"name": f"{name}-cloud", "labels": _labels(name)}
    if owner:
        meta["ownerReferences"] = [owner]
    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": meta,
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": f"{name}-cloud",
            },
            "minReplicas": minimum,
            "maxReplicas": maximum,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": int(
                                autoscaling.get("targetCPUUtilizationPercentage", 70)
                            ),
                        },
                    },
                }
            ],
        },
    }


def render_all(name: str, spec: dict[str, Any], uid: str | None = None) -> list[dict[str, Any]]:
    """Every object a SplitInference owns, ready to apply."""
    owner = owner_reference(name, uid) if uid else None
    objects = [
        render_configmap(name, spec, owner),
        render_deployment(name, spec, owner),
        render_service(name, spec, owner),
    ]
    hpa = render_hpa(name, spec, owner)
    if hpa is not None:
        objects.append(hpa)
    return objects
