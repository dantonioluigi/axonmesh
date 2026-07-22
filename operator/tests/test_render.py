"""Pure-logic tests for the operator's rendering. No kopf, no cluster."""

from __future__ import annotations

import pytest
from splitinference.render import (
    SpecError,
    owner_reference,
    render_all,
    render_configmap,
    render_deployment,
    render_service,
    resolve_cut,
    resolve_role,
)

BASE = {
    "model": {"url": "https://store/model.pt"},
    "cloud": {"image": "ghcr.io/x/axonmesh-cloud:0.5.0"},
}


class TestResolveCut:
    def test_fixed(self):
        assert resolve_cut({"cut": {"mode": "fixed", "fixed": 10}}) == {"mode": "fixed", "cut": 10}

    def test_fixed_requires_value(self):
        with pytest.raises(SpecError, match=r"cut\.fixed"):
            resolve_cut({"cut": {"mode": "fixed"}})

    def test_auto_defaults(self):
        assert resolve_cut({}) == {
            "mode": "auto",
            "bandwidthMbps": 50,
            "fps": 10,
            "transport": "int8",
        }

    def test_auto_explicit(self):
        cut = resolve_cut({"cut": {"mode": "auto", "auto": {"bandwidthMbps": 20, "fps": 5}}})
        assert cut["bandwidthMbps"] == 20 and cut["fps"] == 5

    def test_unknown_mode(self):
        with pytest.raises(SpecError, match=r"unknown cut\.mode"):
            resolve_cut({"cut": {"mode": "sideways"}})


class TestConfigMap:
    def test_carries_cut_and_policy_as_strings(self):
        cm = render_configmap("det", BASE | {"policy": {"confHigh": 0.9}})
        assert cm["kind"] == "ConfigMap"
        assert cm["metadata"]["name"] == "det-edge-config"
        assert cm["data"]["mode"] == "auto"
        assert cm["data"]["confHigh"] == "0.9"
        assert all(isinstance(v, str) for v in cm["data"].values())

    def test_fixed_cut_lands_in_config(self):
        cm = render_configmap("det", BASE | {"cut": {"mode": "fixed", "fixed": 8}})
        assert cm["data"]["mode"] == "fixed"
        assert cm["data"]["cut"] == "8"


class TestDeployment:
    def test_basic_shape(self):
        dep = render_deployment("det", BASE | {"cloud": {"image": "img:1", "replicas": 3}})
        assert dep["kind"] == "Deployment"
        assert dep["metadata"]["name"] == "det-cloud"
        assert dep["spec"]["replicas"] == 3
        c = dep["spec"]["template"]["spec"]["containers"][0]
        assert c["image"] == "img:1"
        assert "--model=/models/model.pt" in c["args"]
        assert any(p["name"] == "wire" for p in c["ports"])

    def test_fixed_cut_passed_as_arg(self):
        dep = render_deployment("det", BASE | {"cut": {"mode": "fixed", "fixed": 12}})
        args = dep["spec"]["template"]["spec"]["containers"][0]["args"]
        assert "--cut=12" in args

    def test_auto_cut_has_no_cut_arg(self):
        dep = render_deployment("det", BASE)
        args = dep["spec"]["template"]["spec"]["containers"][0]["args"]
        assert not any(a.startswith("--cut=") for a in args)

    def test_bottleneck_wires_initcontainer_and_arg(self):
        dep = render_deployment("det", BASE | {"bottleneck": {"url": "https://store/bn.pt"}})
        init = dep["spec"]["template"]["spec"]["initContainers"][0]
        assert "bn.pt" in init["command"][2]
        args = dep["spec"]["template"]["spec"]["containers"][0]["args"]
        assert "--bottleneck=/models/bottleneck.pt" in args

    def test_requires_model_url(self):
        with pytest.raises(SpecError, match=r"model\.url"):
            render_deployment("det", {"cloud": {"image": "img:1"}})

    def test_requires_cloud_image(self):
        with pytest.raises(SpecError, match=r"cloud\.image"):
            render_deployment("det", {"model": {"url": "u"}, "cloud": {}})


def test_service_targets_named_ports():
    svc = render_service("det", BASE)
    assert svc["kind"] == "Service"
    ports = {p["name"]: p["targetPort"] for p in svc["spec"]["ports"]}
    assert ports == {"wire": "wire", "metrics": "metrics"}


def test_owner_reference_and_render_all():
    objs = render_all("det", BASE, uid="uid-123")
    kinds = [o["kind"] for o in objs]
    assert kinds == ["ConfigMap", "Deployment", "Service"]
    for o in objs:
        ref = o["metadata"]["ownerReferences"][0]
        assert ref["uid"] == "uid-123" and ref["controller"] is True


def test_render_all_without_uid_has_no_owner():
    objs = render_all("det", BASE)
    assert all("ownerReferences" not in o["metadata"] for o in objs)


def test_owner_reference_shape():
    ref = owner_reference("det", "u1")
    assert ref["kind"] == "SplitInference"
    assert ref["blockOwnerDeletion"] is True


def init_script(deployment) -> str:
    """The shell the initContainer will actually run."""
    return deployment["spec"]["template"]["spec"]["initContainers"][0]["command"][-1]


def test_a_spec_without_an_escalation_model_is_a_split():
    assert resolve_role({"model": {"url": "https://x/m.pt"}}) == "split"


def test_declaring_an_escalation_model_makes_it_a_cascade():
    """The role is derived, not declared: a CR cannot say one and configure the
    other, and the edge needs it to know what the handshake will enforce."""
    spec = {"model": {"url": "https://x/m.pt"}, "escalateTo": {"url": "https://x/big.pt"}}
    assert resolve_role(spec) == "cascade"


def test_the_edge_config_carries_the_role_and_the_statistic():
    spec = {
        "model": {"url": "https://x/m.pt"},
        "escalateTo": {"url": "https://x/big.pt"},
        "policy": {"statistic": "mean"},
    }
    data = render_configmap("det", spec)["data"]
    assert data["role"] == "cascade"
    assert data["statistic"] == "mean"


def test_the_cloud_downloads_and_serves_the_escalation_model():
    spec = {
        "model": {"url": "https://x/m.pt"},
        "escalateTo": {"url": "https://x/big.pt"},
        "cloud": {"image": "ghcr.io/you/cloud:1"},
    }
    deployment = render_deployment("det", spec)
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert "--escalate-to=/models/escalate.pt" in container["args"]
    assert "https://x/big.pt" in init_script(deployment)


def test_a_declared_digest_is_verified_before_the_pod_starts():
    spec = {
        "model": {"url": "https://x/m.pt", "sha256": "abc123"},
        "cloud": {"image": "ghcr.io/you/cloud:1"},
    }
    script = init_script(render_deployment("det", spec))
    assert "sha256sum -c -" in script
    assert "abc123  /models/model.pt" in script
    assert "set -eu" in script  # a failed check must stop the pod, not be logged


def test_a_url_cannot_smuggle_shell_into_the_init_container():
    """The URL comes from a custom resource, and the init container is a shell."""
    spec = {
        "model": {"url": "https://x/m.pt; touch /pwned"},
        "cloud": {"image": "ghcr.io/you/cloud:1"},
    }
    script = init_script(render_deployment("det", spec))
    assert "; touch /pwned" not in script.replace("'https://x/m.pt; touch /pwned'", "")
    assert "'https://x/m.pt; touch /pwned'" in script  # quoted, so it stays one argument
