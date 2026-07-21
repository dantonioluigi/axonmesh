"""Live re-planning: drive the cut planner from measured link and edge load.

The planner (:mod:`splitflow.planner`) is stateless — given a byte budget it
picks a cut. In production the budget is not static: bandwidth drifts and the
edge's own compute headroom changes. This module closes the loop with two
estimators and a controller that re-plans **with hysteresis**, because flapping
between cuts (which forces edge and cloud to renegotiate the wire every frame)
is worse than sitting on a mildly sub-optimal cut.

Hysteresis is asymmetric on purpose:

- **Degrade fast** — if the current plan no longer fits the budget, switch to a
  lighter wire immediately; overrunning the link drops frames.
- **Upgrade slow** — only move to a heavier-wire / more-offloaded plan once it
  has fit the budget *with margin* for ``patience`` consecutive observations,
  so a brief bandwidth spike does not trigger a switch that the next dip undoes.

Edge load is a secondary signal: when the edge is compute-bound past
``load_ceiling``, an upgrade that offloads more (a lighter edge half) is taken
without waiting out the patience window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .planner import CutOption, budget_bytes_per_frame, plan_cut


class BandwidthEstimator:
    """EWMA of achieved throughput in Mbps, fed ``(bytes, seconds)`` samples."""

    def __init__(self, alpha: float = 0.3, initial_mbps: float | None = None) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._mbps = initial_mbps

    @property
    def mbps(self) -> float | None:
        """Current estimate, or ``None`` before the first sample."""
        return self._mbps

    def update(self, nbytes: int, seconds: float) -> float:
        """Fold one transfer sample into the estimate and return it."""
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        sample = nbytes * 8 / seconds / 1e6
        self._mbps = (
            sample if self._mbps is None else ((1 - self.alpha) * self._mbps + self.alpha * sample)
        )
        return self._mbps


def cpu_load() -> float:
    """Default edge-load signal: system CPU utilisation in [0, 1] via psutil."""
    import psutil

    return psutil.cpu_percent(interval=None) / 100.0


@dataclass
class ReplanDecision:
    """Outcome of one :meth:`ReplanningController.observe` call."""

    plan: CutOption
    switched: bool
    reason: str
    budget_bytes: int
    edge_load: float | None = None


@dataclass
class ReplanningController:
    """Stateful wrapper over :func:`plan_cut` with anti-flap hysteresis.

    Args:
        options: candidate cuts (from :func:`splitflow.planner.enumerate_cuts`).
        fps: target frame rate, used to turn Mbps into a per-frame byte budget.
        transport: wire encoding priced by the planner.
        patience: consecutive observations a better plan must persist before an
            upgrade is committed.
        margin: fractional headroom an upgrade plan must fit within before it is
            even considered (0.15 = must fit 15% under budget).
        load_ceiling: edge-load fraction above which upgrades that offload more
            skip the patience window.
    """

    options: list[CutOption]
    fps: float
    transport: str = "int8"
    patience: int = 3
    margin: float = 0.15
    load_ceiling: float = 0.85
    current: CutOption | None = field(default=None, init=False)
    _candidate: CutOption | None = field(default=None, init=False)
    _streak: int = field(default=0, init=False)

    def _reset_pending(self) -> None:
        self._candidate = None
        self._streak = 0

    def observe(self, bandwidth_mbps: float, edge_load: float | None = None) -> ReplanDecision:
        """Fold in a new bandwidth (and optional load) reading; maybe re-plan."""
        budget = budget_bytes_per_frame(bandwidth_mbps, self.fps)

        if self.current is None:
            self.current = plan_cut(self.options, budget, self.transport)
            reason = "initial plan" if self.current else "no plan fits the initial budget"
            return ReplanDecision(self.current, bool(self.current), reason, budget, edge_load)

        # Degrade fast: the active plan overruns the link -> switch down now.
        if self.current.wire_bytes(self.transport) > budget:
            downgraded = plan_cut(self.options, budget, self.transport)
            self.current = downgraded
            self._reset_pending()
            reason = "budget exceeded -> downgraded" if downgraded else "no plan fits -> stalled"
            return ReplanDecision(self.current, True, reason, budget, edge_load)

        # Upgrade slow: best plan that fits with margin headroom.
        conservative = int(budget / (1 + self.margin))
        target = plan_cut(self.options, conservative, self.transport)
        if target is None or target.cut == self.current.cut:
            self._reset_pending()
            return ReplanDecision(self.current, False, "stable", budget, edge_load)

        offloads_more = target.edge_params_share < self.current.edge_params_share
        hot_edge = edge_load is not None and edge_load > self.load_ceiling
        if hot_edge and offloads_more:
            self.current = target
            self._reset_pending()
            return ReplanDecision(
                self.current, True, "edge overloaded -> offloaded now", budget, edge_load
            )

        # Accumulate a stability streak for the candidate before committing.
        if self._candidate is not None and self._candidate.cut == target.cut:
            self._streak += 1
        else:
            self._candidate = target
            self._streak = 1
        if self._streak >= self.patience:
            self.current = target
            self._reset_pending()
            return ReplanDecision(
                self.current, True, f"upgraded after {self.patience} stable obs", budget, edge_load
            )
        return ReplanDecision(
            self.current,
            False,
            f"upgrade pending ({self._streak}/{self.patience})",
            budget,
            edge_load,
        )


def simulate_trace(
    controller: ReplanningController,
    trace: list[tuple[float, float | None]],
) -> list[ReplanDecision]:
    """Run the controller over a scripted ``[(bandwidth_mbps, edge_load), ...]`` trace."""
    return [controller.observe(bw, load) for bw, load in trace]
