"""The prediction losses must not depend on how a group of sequences is split into microbatches.

``update_ppo`` weights each sequence microbatch by its share of the minibatch's sequences, so a loss
that is the group mean of per-sequence means is reproduced exactly by any split.  A loss pooled over
all valid steps would weight sequences by their surviving length and change with the split whenever
episodes end at different times (disk tag captures), which is what the 2026-11-02 review found for
``sequence_microbatch_size`` 128 -> 512."""

from __future__ import annotations

import torch

from sb_pomdp.networks import ConditionMixturePredictor, ObservationPredictor
from sb_pomdp.ppo import belief_mixture_prediction_loss, condition_prediction_loss


def _uneven_dones(steps: int, group: int) -> torch.Tensor:
    dones = torch.zeros(steps, group, dtype=torch.bool)
    dones[1, 0] = True  # sequence 0 ends early
    dones[steps - 2, 2] = True  # sequence 2 ends late
    dones[:, 3] = True  # sequence 3 has no valid transition at all
    return dones


def test_condition_prediction_loss_is_the_group_mean_of_per_sequence_means() -> None:
    torch.manual_seed(0)
    predictor = ConditionMixturePredictor(4, 2, 3, [8], components=2)
    steps, group = 6, 4
    condition = torch.randn(steps * group, 4)
    observations = torch.randn(steps, group, 3)
    previous_actions = torch.randn(steps, group, 2)
    dones = _uneven_dones(steps, group)
    whole = condition_prediction_loss(
        predictor, condition, observations=observations, previous_actions=previous_actions, dones=dones
    )
    singles = torch.stack([
        condition_prediction_loss(
            predictor,
            condition.reshape(steps, group, 4)[:, g].reshape(steps, 4),
            observations=observations[:, g : g + 1],
            previous_actions=previous_actions[:, g : g + 1],
            dones=dones[:, g : g + 1],
        )
        for g in range(group)
    ])
    assert singles[3] == 0.0  # no valid step: contributes zero, still counts as one sequence
    torch.testing.assert_close(whole, singles.mean(), atol=1e-6, rtol=1e-6)
    pairs = torch.stack([
        condition_prediction_loss(
            predictor,
            condition.reshape(steps, group, 4)[:, g : g + 2].reshape(steps * 2, 4),
            observations=observations[:, g : g + 2],
            previous_actions=previous_actions[:, g : g + 2],
            dones=dones[:, g : g + 2],
        )
        for g in (0, 2)
    ])
    torch.testing.assert_close(whole, pairs.mean(), atol=1e-6, rtol=1e-6)


def test_belief_mixture_prediction_loss_is_the_group_mean_of_per_sequence_means() -> None:
    torch.manual_seed(1)
    predictor = ObservationPredictor(2, 1, 3, [8])
    steps, group = 6, 4
    particles = torch.randn(steps, group, 5, 2)
    observations, actions = torch.randn(steps, group, 3), torch.randn(steps, group, 1)
    dones = _uneven_dones(steps, group)
    whole = belief_mixture_prediction_loss(
        predictor, particles, observations=observations, previous_actions=actions, dones=dones
    )
    singles = torch.stack([
        belief_mixture_prediction_loss(
            predictor,
            particles[:, g : g + 1],
            observations=observations[:, g : g + 1],
            previous_actions=actions[:, g : g + 1],
            dones=dones[:, g : g + 1],
        )
        for g in range(group)
    ])
    torch.testing.assert_close(whole, singles.mean(), atol=1e-6, rtol=1e-6)
