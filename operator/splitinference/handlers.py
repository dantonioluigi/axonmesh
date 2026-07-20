"""kopf handlers: apply what :mod:`render` computes, and report status.

This is the thin event layer. All the decisions live in ``render`` (pure,
unit-tested); here we just create-or-patch the rendered objects on
create/update/resume and write back a status subresource. Child objects carry
an ownerReference to the CR, so Kubernetes garbage-collects them on delete —
no explicit teardown handler needed.

Run locally against any cluster (the e2e does this):

    kopf run -m operator.splitinference.handlers --standalone
"""

from __future__ import annotations

from typing import Any

import kopf
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from .render import API_GROUP, API_VERSION, owner_reference, render_all, resolve_cut


def _load_kube() -> None:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


_load_kube()


def _create_or_patch(create, patch, name: str, namespace: str, obj: dict[str, Any]) -> None:
    """Create the object, or patch it if it already exists."""
    try:
        create(namespace=namespace, body=obj)
    except ApiException as err:
        if err.status != 409:
            raise
        patch(name=name, namespace=namespace, body=obj)


def _apply(obj: dict[str, Any], namespace: str) -> None:
    """Dispatch one rendered object to the right typed API as create-or-patch."""
    name = obj["metadata"]["name"]
    kind = obj["kind"]
    if kind == "ConfigMap":
        core = client.CoreV1Api()
        _create_or_patch(
            core.create_namespaced_config_map,
            core.patch_namespaced_config_map,
            name,
            namespace,
            obj,
        )
    elif kind == "Service":
        core = client.CoreV1Api()
        _create_or_patch(
            core.create_namespaced_service, core.patch_namespaced_service, name, namespace, obj
        )
    elif kind == "Deployment":
        apps = client.AppsV1Api()
        _create_or_patch(
            apps.create_namespaced_deployment,
            apps.patch_namespaced_deployment,
            name,
            namespace,
            obj,
        )
    else:  # pragma: no cover - render only emits the three kinds above
        raise kopf.PermanentError(f"cannot apply unknown kind {kind}")


@kopf.on.create(API_GROUP, API_VERSION, "splitinferences")
@kopf.on.update(API_GROUP, API_VERSION, "splitinferences")
@kopf.on.resume(API_GROUP, API_VERSION, "splitinferences")
def reconcile(spec, name, namespace, uid, patch, logger, **_):
    """Render the owned objects, apply them, and report the resolved cut."""
    owner = owner_reference(name, uid)
    for obj in render_all(name, dict(spec), uid=uid):
        obj["metadata"]["ownerReferences"] = [owner]
        _apply(obj, namespace)
        logger.info("applied %s/%s", obj["kind"], obj["metadata"]["name"])

    cut = resolve_cut(dict(spec))
    patch.status["cut"] = cut
    patch.status["cloudService"] = f"{name}-cloud.{namespace}.svc:9095"
    patch.status["edgeConfig"] = f"{name}-edge-config"
    patch.status["phase"] = "Ready"
    return {"cut": cut}
