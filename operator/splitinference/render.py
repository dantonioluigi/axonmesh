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


def render_configmap(name: str, spec: dict[str, Any], owner: dict | None = None) -> dict[str, Any]:
    """The edge-facing config: cut decision + policy thresholds."""
    policy = spec.get("policy", {})
    data = resolve_cut(spec) | {
        "confHigh": float(policy.get("confHigh", 0.75)),
        "confLow": float(policy.get("confLow", 0.4)),
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
    init_cmd = f"curl -fsSL -o /models/model.pt {model['url']}"
    if bottleneck.get("url"):
        init_cmd += f" && curl -fsSL -o /models/bottleneck.pt {bottleneck['url']}"
        args.append("--bottleneck=/models/bottleneck.pt")

    meta: dict[str, Any] = {"name": f"{name}-cloud", "labels": _labels(name)}
    if owner:
        meta["ownerReferences"] = [owner]
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": meta,
        "spec": {
            "replicas": int(cloud.get("replicas", 1)),
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


def render_all(name: str, spec: dict[str, Any], uid: str | None = None) -> list[dict[str, Any]]:
    """Every object a SplitInference owns, ready to apply."""
    owner = owner_reference(name, uid) if uid else None
    return [
        render_configmap(name, spec, owner),
        render_deployment(name, spec, owner),
        render_service(name, spec, owner),
    ]
