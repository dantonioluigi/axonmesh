# SplitInference operator image — tiny: kopf + kubernetes client + the handlers.
# The operator renders manifests from the CR; it does NOT load the model, so no
# torch/ultralytics here (that keeps it small and fast to start).
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir "kopf>=1.37" "kubernetes>=28"
COPY splitinference ./splitinference

# kopf watches the CRD and reconciles; --standalone skips peering (single replica).
ENTRYPOINT ["kopf", "run", "-m", "splitinference.handlers", "--standalone", "--all-namespaces"]
