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
)

BASE = {
    "model": {"url": "https://store/model.pt"},
    "cloud": {"image": "ghcr.io/x/splitflow-cloud:0.5.0"},
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
