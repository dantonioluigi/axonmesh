# Maintenance playbook

How this repository is kept healthy as forks and pull requests arrive.

## Branch model

- `main` is the only long-lived branch and must always be green.
- Every change — including the maintainer's own — lands via a short-lived
  feature branch and a pull request. Nobody pushes to `main` directly once
  branch protection is enabled.
- Recommended GitHub settings (Settings → Branches → protect `main`):
  - Require a pull request before merging.
  - Require status checks: `lint`, `test (3.10)`, `test (3.12)`.
  - Require branches to be up to date before merging.
  - Prefer **squash merge** (Settings → General → Pull Requests: enable squash,
    disable merge commits) so `main` history stays one-commit-per-change.

## Reviewing pull requests from forks

1. CI runs automatically on the PR; nothing gets reviewed until it is green.
   First-time contributors need a maintainer click to start workflows
   (GitHub's default `Require approval for first-time contributors` — keep it,
   it prevents workflow abuse).
2. Check the PR against the template checklist: tests for new behaviour,
   bounded reconstruction error for new transports/codecs, measured numbers
   with the command that produced them.
3. Never merge weights, datasets, or result artifacts — `.gitignore` blocks the
   common cases, `check-added-large-files` (pre-commit) blocks the rest.
4. Security note: only load checkpoints with `weights_only=True` (the code
   already does) and be wary of PRs that change `torch.load`/`subprocess`/CI
   files — review those line by line.

## Releases

1. Update `CHANGELOG.md` and bump the version in **both** `pyproject.toml` and
   `src/axonmesh/__init__.py` (they must match).
2. Tag: `git tag -a v0.x.0 -m "..." && git push origin v0.x.0`, then create a
   GitHub Release from the tag pasting the changelog entry.
3. Versioning is semver-flavoured: breaking API/CLI changes bump minor while
   0.x, patch for fixes. Anything touching the wire format
   (`QuantizedTensor.to_bytes`, detection serialisation) is a breaking change —
   edge and cloud must speak the same version.

## Dependencies

- Dependabot opens weekly grouped PRs for pip and GitHub Actions; merge them
  when CI is green.
- `ultralytics` is the fragile dependency: the splitter relies on `m.f`/`m.i`
  wiring and `_predict_once`. The test suite covers exactly those contact
  points, so a green CI on a Dependabot bump is a real compatibility signal.
  If a bump breaks, pin the upper bound in `pyproject.toml` and open an issue.

## Issue triage

- `bug` with a reproduction → fix or label `help wanted`.
- `enhancement` → check against the scope gate in the feature template
  (measurement angles yes; the K8s operator waits for the numbers).
- Questions about experimental results → answer with commands, not opinions;
  the README results table is the single source of truth and every row must be
  reproducible.

## Keeping experiments honest

- Numbers in the README must state: model, dataset, image size, JPEG quality,
  cut point, transport, and device.
- Byte counts are hardware-independent; latency is not — anything time-related
  is measured on the Jetson or explicitly labelled otherwise.
- When a finding changes a conclusion (like the 30x gap), update the README
  *finding* paragraph, not just the table.
