#!/usr/bin/env bash
# End-to-end test for the SplitInference operator on a real (kind) cluster.
#
# Spins up a throwaway kind cluster, installs the CRD + RBAC, runs the operator
# (kopf, locally against the cluster), then drives a SplitInference through its
# lifecycle and asserts the operator reconciles it: create -> children exist &
# status Ready; update -> children re-rendered; delete -> children garbage
# collected via ownerReferences.
#
#   PYTHON=.venv/bin/python deploy/kind/e2e.sh          # run it
#   KEEP=1 PYTHON=.venv/bin/python deploy/kind/e2e.sh   # keep the cluster
set -euo pipefail

CLUSTER="${CLUSTER:-splitinference-e2e}"
PYTHON="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OP_LOG="${OP_LOG:-$REPO/deploy/kind/operator.log}"
OP_IMAGE="${OP_IMAGE:-axonmesh-operator:e2e}"

red() { printf '\033[31m%s\033[0m\n' "$*"; }
grn() { printf '\033[32m%s\033[0m\n' "$*"; }

cleanup() {
  if [ "${KEEP:-0}" != "1" ]; then
    kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

assert() { # <description> <actual> <expected>
  if [ "$2" = "$3" ]; then grn "  ok: $1 = $3"; else red "  FAIL: $1: got '$2', want '$3'"; exit 1; fi
}

wait_for() { # <description> <cmd...>  — poll up to 90s
  for _ in $(seq 1 90); do "${@:2}" >/dev/null 2>&1 && return 0; sleep 1; done
  red "timed out waiting for: $1"
  echo "---- operator log ----"
  kubectl logs -l app.kubernetes.io/name=axonmesh-operator --tail=40 2>/dev/null || true
  exit 1
}

echo "== create kind cluster '$CLUSTER' =="
kind get clusters 2>/dev/null | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER" --wait 90s
kubectl config use-context "kind-$CLUSTER" >/dev/null

# Build and run the *image*, installed by the chart, as a user would.
#
# This used to start kopf locally with PYTHONPATH=operator and the developer's
# own kubeconfig. That exercised the handlers and nothing else, and hid three
# bugs at once: the image collapsed its command into ENTRYPOINT so Kubernetes
# `args` were appended rather than substituted, it never put /app on sys.path
# so `kopf run -m` could not find the handlers, and the ClusterRole was missing
# the CRD and namespace reads kopf performs at startup. None of the three can
# appear when the operator runs as the developer instead of as its
# ServiceAccount, out of its own image.
echo "== build + load the operator image =="
docker build -q -f "$REPO/operator/Dockerfile" -t "$OP_IMAGE" "$REPO/operator"
kind load docker-image "$OP_IMAGE" --name "$CLUSTER"

echo "== helm install the operator (chart brings the CRD and RBAC) =="
helm upgrade --install axonmesh-operator "$REPO/deploy/helm/axonmesh-operator" \
  --set image.repository="${OP_IMAGE%%:*}" --set image.tag="${OP_IMAGE##*:}" --wait --timeout 120s
kubectl wait --for=condition=established crd/splitinferences.axonmesh.dev --timeout=30s
kubectl rollout status deployment/axonmesh-operator --timeout=120s
# A running pod is not a watching operator: kopf authenticates first, and a CR
# created in that window is missed. Gate on the log line, not on a sleep.
wait_for "operator authenticated and watching" bash -c \
  "kubectl logs -l app.kubernetes.io/name=axonmesh-operator --tail=20 | grep -q 'authentication has finished'"

echo "== apply a SplitInference =="
kubectl apply -f "$REPO/operator/examples/splitinference.yaml"
# The ConfigMap appearing IS the proof the operator connected and reconciled.
wait_for "edge ConfigMap created (operator reconciled)" kubectl get configmap detector-edge-config

echo "== assert reconcile output =="
assert "configmap mode" "$(kubectl get cm detector-edge-config -o jsonpath='{.data.mode}')" "auto"
assert "configmap confHigh" "$(kubectl get cm detector-edge-config -o jsonpath='{.data.confHigh}')" "0.75"
assert "deployment replicas" "$(kubectl get deploy detector-cloud -o jsonpath='{.spec.replicas}')" "2"
assert "service wire port" "$(kubectl get svc detector-cloud -o jsonpath='{.spec.ports[0].port}')" "9095"
CR_UID="$(kubectl get splitinference detector -o jsonpath='{.metadata.uid}')"
assert "configmap ownerRef uid" "$(kubectl get cm detector-edge-config -o jsonpath='{.metadata.ownerReferences[0].uid}')" "$CR_UID"
wait_for "status Ready" bash -c "kubectl get splitinference detector -o jsonpath='{.status.phase}' | grep -q Ready"
assert "status cut mode" "$(kubectl get splitinference detector -o jsonpath='{.status.cut.mode}')" "auto"

# The operator recovers from missing permissions by retrying, so an RBAC gap
# shows up as a wall of 403s rather than a failure. Assert on the log.
echo "== assert the ServiceAccount can do its job =="
if kubectl logs -l app.kubernetes.io/name=axonmesh-operator --tail=200 | grep -q "Forbidden"; then
  echo "FAIL: the operator hit RBAC denials (it retries, so reconcile still worked)"
  kubectl logs -l app.kubernetes.io/name=axonmesh-operator --tail=200 | grep -m3 "Forbidden"
  exit 1
fi
echo "  ok: no RBAC denials"

echo "== update: switch to a fixed cut, scale down =="
kubectl patch splitinference detector --type merge \
  -p '{"spec":{"cut":{"mode":"fixed","fixed":8},"cloud":{"image":"ghcr.io/dantonioluigi/axonmesh-cloud:0.5.0","replicas":1}}}'
wait_for "configmap became fixed" bash -c "kubectl get cm detector-edge-config -o jsonpath='{.data.mode}' | grep -q fixed"
assert "configmap fixed cut" "$(kubectl get cm detector-edge-config -o jsonpath='{.data.cut}')" "8"
wait_for "deployment scaled" bash -c "kubectl get deploy detector-cloud -o jsonpath='{.spec.replicas}' | grep -qx 1"
assert "deployment has --cut arg" "$(kubectl get deploy detector-cloud -o jsonpath='{.spec.template.spec.containers[0].args[5]}')" "--cut=8"

echo "== delete: children are garbage-collected =="
kubectl delete splitinference detector
wait_for "configmap GC'd" bash -c "! kubectl get cm detector-edge-config >/dev/null 2>&1"
wait_for "deployment GC'd" bash -c "! kubectl get deploy detector-cloud >/dev/null 2>&1"

grn "== E2E PASSED =="
