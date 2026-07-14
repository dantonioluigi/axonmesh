# Contributing

Issues and pull requests are welcome. This is an experimental research codebase;
keep changes small and measurable.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

- `pytest` — the suite must pass and coverage must stay ≥ 85%. Tests build
  YOLO11n from YAML with random weights, so they need no downloads and no GPU.
- `ruff check . && ruff format .` — lint and formatting are enforced in CI.
- New wire formats or transports need a test that bounds their reconstruction
  error, not just a smoke test.
- Claims about bandwidth or accuracy belong in the README results table with
  the exact command that produced them.

## Scope

Bug fixes and new measurement angles (codecs, cut points, transports) are in
scope. The adaptive policy / Kubernetes operator work is intentionally out of
scope until the feasibility numbers justify it — see
[docs/experiment-protocol.md](docs/experiment-protocol.md).
