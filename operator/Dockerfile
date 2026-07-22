# SplitInference operator image — tiny: kopf + kubernetes client + the handlers.
# The operator renders manifests from the CR; it does NOT load the model, so no
# torch/ultralytics here (that keeps it small and fast to start).
FROM python:3.12-slim

WORKDIR /app
# `kopf` is a console script, so sys.path[0] is /usr/local/bin, not the working
# directory: without this the handlers module is invisible to `kopf run -m`.
ENV PYTHONPATH=/app
RUN pip install --no-cache-dir "kopf>=1.37" "kubernetes>=28"
COPY splitinference ./splitinference

# ENTRYPOINT is the binary and CMD the default arguments, so a deployment can
# override the arguments without repeating the binary. Collapsing both into
# ENTRYPOINT means Kubernetes `args` are appended rather than substituted, and
# kopf is handed its own subcommand twice.
#
# --standalone skips peer election: the operator runs as a single replica.
ENTRYPOINT ["kopf"]
CMD ["run", "-m", "splitinference.handlers", "--standalone", "--all-namespaces"]
