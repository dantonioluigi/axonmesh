from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from axonmesh.allocate import (
    LevelSensitivity,
    allocation_cost,
    level_sensitivity,
    propose_allocation,
)
from axonmesh.split import SplitRunner


class TwoScaleNet(nn.Module):
    """A shallow net whose activations stay measurable.

    The shared ``det_model`` fixture is randomly initialised, and a randomly
    initialised deep detector damps its own activations to ~1e-8 by the end of
    the backbone: coarsening a tensor of zeros moves nothing, so there is no
    sensitivity to measure. Anything probing real activations needs a model
    shallow enough to still have some.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.deep = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.head = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        shallow = torch.relu(self.stem(x)) + 1.0  # keep it away from a dead ReLU
        deep = torch.relu(self.deep(shallow)) + 1.0
        return self.head(deep)


@pytest.fixture()
def two_scale():
    torch.manual_seed(15)
    return TwoScaleNet().eval()


def sensitivity(index, error, spatial) -> LevelSensitivity:
    return LevelSensitivity(index=index, output_error=error, spatial=spatial)


def test_sensitivity_is_reported_once_per_wire_level(two_scale):
    runner = SplitRunner(two_scale, cut=1)
    measured = level_sensitivity(runner, torch.rand(1, 3, 32, 32), bits=2)

    assert [s.index for s in measured] == sorted(runner.wire)
    assert all(s.output_error > 0 for s in measured)  # coarsening must move the output
    assert all(s.spatial > 0 for s in measured)


def test_coarser_probing_moves_the_output_further(two_scale):
    """A sanity check on the probe itself: fewer bits, more damage."""
    runner = SplitRunner(two_scale, cut=1)
    frames = torch.rand(1, 3, 32, 32)
    gentle = {s.index: s.output_error for s in level_sensitivity(runner, frames, bits=6)}
    harsh = {s.index: s.output_error for s in level_sensitivity(runner, frames, bits=1)}
    assert all(harsh[i] > gentle[i] for i in gentle)


def test_the_budget_goes_to_the_level_that_is_small_and_sensitive():
    """The point of the whole module: equal sensitivity does not mean equal width.

    Two levels are equally sensitive, but one is 16x larger spatially. A
    channel there costs 16x as much for the same benefit, so the budget
    belongs to the small one.
    """
    measured = [sensitivity(4, error=0.05, spatial=1600), sensitivity(10, error=0.05, spatial=100)]
    allocation = propose_allocation(measured, budget_channels=8)
    assert allocation[10] > allocation[4]


def test_a_proposal_does_not_cost_more_than_the_uniform_width_it_replaces():
    measured = [
        sensitivity(4, error=0.0445, spatial=1600),
        sensitivity(6, error=0.0469, spatial=400),
        sensitivity(10, error=0.0589, spatial=100),
    ]
    proposal = propose_allocation(measured, budget_channels=8)
    uniform = dict.fromkeys(proposal, 8)
    assert allocation_cost(proposal, measured) <= allocation_cost(uniform, measured)


def test_no_level_is_allocated_away_entirely():
    """A level squeezed to zero channels is a level deleted from the wire."""
    measured = [
        sensitivity(4, error=1e-9, spatial=100_000),  # negligible and enormous
        sensitivity(10, error=0.9, spatial=100),
    ]
    allocation = propose_allocation(measured, budget_channels=8, minimum=2)
    assert allocation[4] == 2


def test_a_budget_below_the_floor_is_refused():
    with pytest.raises(ValueError, match="below the"):
        propose_allocation([sensitivity(4, 0.1, 100)], budget_channels=1, minimum=2)


def test_equal_sensitivity_and_size_reproduces_the_uniform_allocation():
    measured = [sensitivity(4, error=0.05, spatial=400), sensitivity(6, error=0.05, spatial=400)]
    assert propose_allocation(measured, budget_channels=8) == {4: 8, 6: 8}


def test_a_model_whose_output_ignores_distortion_falls_back_to_uniform():
    measured = [sensitivity(4, error=0.0, spatial=400), sensitivity(6, error=0.0, spatial=100)]
    assert propose_allocation(measured, budget_channels=8) == {4: 8, 6: 8}


def test_allocation_cost_scales_with_the_latent_stride():
    measured = [sensitivity(4, 0.1, spatial=400)]
    assert allocation_cost({4: 8}, measured, stride=1) == 3200
    assert allocation_cost({4: 8}, measured, stride=2) == 800


def test_sensitivity_probing_leaves_the_model_untouched(det_model, probe):
    """The probe must not be able to perturb the detector it is measuring."""
    runner = SplitRunner(det_model)
    before = torch.cat([p.flatten() for p in det_model.parameters()]).clone()
    level_sensitivity(runner, probe, bits=3)
    after = torch.cat([p.flatten() for p in det_model.parameters()])
    assert torch.equal(before, after)
