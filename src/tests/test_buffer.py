from __future__ import annotations

import pytest
import torch

from sb_pomdp.buffer import RolloutBuilder, generalised_advantage_estimate
from sb_pomdp.ppo import diffusion_advantage_weights


def test_gae_matches_short_hand_calculation() -> None:
    rewards = torch.tensor([[1.0], [2.0]])
    dones = torch.tensor([[0.0], [1.0]])
    values = torch.tensor([[0.5], [0.25]])
    advantage, returns = generalised_advantage_estimate(
        rewards,
        dones,
        values,
        torch.tensor([99.0]),
        gamma=0.9,
        gae_lambda=0.8,
    )
    final_advantage = 2.0 - 0.25
    first_delta = 1.0 + 0.9 * 0.25 - 0.5
    expected_first = first_delta + 0.9 * 0.8 * final_advantage
    torch.testing.assert_close(
        advantage,
        torch.tensor([[expected_first], [final_advantage]]),
    )
    torch.testing.assert_close(returns, advantage + values)


def test_done_transition_does_not_bootstrap() -> None:
    """A closed episode never bootstraps the successor of the reset observation.

    The first case locks ``ppo.bootstrap_on_truncation=false``, the campaign's
    convention.  The other two lock the opt-in convention: a true terminal state
    still contributes nothing, while a time-limit truncation adds exactly
    ``gamma * V(s_T)`` for the post-step state the auto-reset discarded.
    """

    arguments = (
        torch.tensor([[1.0]]),
        torch.tensor([[1.0]]),
        torch.tensor([[0.2]]),
        torch.tensor([1000.0]),
    )
    advantage, _ = generalised_advantage_estimate(
        *arguments,
        gamma=0.99,
        gae_lambda=0.95,
    )
    torch.testing.assert_close(advantage, torch.tensor([[0.8]]))

    terminated, _ = generalised_advantage_estimate(
        *arguments,
        gamma=0.99,
        gae_lambda=0.95,
        truncation_values=torch.tensor([[0.0]]),
    )
    torch.testing.assert_close(terminated, torch.tensor([[0.8]]))

    truncated, returns = generalised_advantage_estimate(
        *arguments,
        gamma=0.99,
        gae_lambda=0.95,
        truncation_values=torch.tensor([[2.0]]),
    )
    torch.testing.assert_close(truncated, torch.tensor([[0.8 + 0.99 * 2.0]]))
    torch.testing.assert_close(returns, truncated + arguments[2])


def _pre_bootstrap_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Reimplement the recursion exactly as it stood before F-1."""

    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for time_index in reversed(range(rewards.shape[0])):
        next_value = last_value if time_index == rewards.shape[0] - 1 else values[time_index + 1]
        nonterminal = 1.0 - dones[time_index].float()
        delta = rewards[time_index] + gamma * next_value * nonterminal - values[time_index]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[time_index] = running
    return advantages


_SYNTHETIC_ROLLOUT = {
    "rewards": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    # Environment 0 truncates at t=0 and truly terminates at t=1; environment 1
    # runs through t=0 and truncates at t=1.
    "dones": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
    "values": torch.tensor([[0.5, 0.25], [0.75, 1.0]]),
    "last_value": torch.tensor([100.0, 200.0]),
}


def test_omitted_truncation_values_reproduce_the_pre_bootstrap_recursion() -> None:
    advantage, returns = generalised_advantage_estimate(
        _SYNTHETIC_ROLLOUT["rewards"],
        _SYNTHETIC_ROLLOUT["dones"],
        _SYNTHETIC_ROLLOUT["values"],
        _SYNTHETIC_ROLLOUT["last_value"],
        gamma=0.9,
        gae_lambda=0.8,
    )
    expected = _pre_bootstrap_advantages(
        _SYNTHETIC_ROLLOUT["rewards"],
        _SYNTHETIC_ROLLOUT["dones"],
        _SYNTHETIC_ROLLOUT["values"],
        _SYNTHETIC_ROLLOUT["last_value"],
        gamma=0.9,
        gae_lambda=0.8,
    )
    torch.testing.assert_close(advantage, expected)
    torch.testing.assert_close(advantage, torch.tensor([[0.5, 4.81], [2.25, 3.0]]))
    torch.testing.assert_close(returns, advantage + _SYNTHETIC_ROLLOUT["values"])


def test_truncation_values_bootstrap_only_the_truncated_steps() -> None:
    truncation_values = torch.tensor([[5.0, 0.0], [0.0, 7.0]])
    advantage, returns = generalised_advantage_estimate(
        _SYNTHETIC_ROLLOUT["rewards"],
        _SYNTHETIC_ROLLOUT["dones"],
        _SYNTHETIC_ROLLOUT["values"],
        _SYNTHETIC_ROLLOUT["last_value"],
        gamma=0.9,
        gae_lambda=0.8,
        truncation_values=truncation_values,
    )

    # t=1: delta = r - V, plus gamma * V(s_T) where the horizon cut the episode.
    last_step = torch.tensor([3.0 - 0.75, 4.0 - 1.0 + 0.9 * 7.0])
    # t=0: environment 0 is closed by its own done, so only its bootstrap adds;
    # environment 1 continues and still accumulates the lambda-return of t=1.
    first_step = torch.tensor(
        [
            1.0 - 0.5 + 0.9 * 5.0,
            2.0 + 0.9 * 1.0 - 0.25 + 0.9 * 0.8 * float(last_step[1]),
        ]
    )
    torch.testing.assert_close(advantage, torch.stack((first_step, last_step)))
    torch.testing.assert_close(returns, advantage + _SYNTHETIC_ROLLOUT["values"])


def test_truncation_values_must_match_the_rollout_shape() -> None:
    with pytest.raises(ValueError, match="truncation_values"):
        generalised_advantage_estimate(
            _SYNTHETIC_ROLLOUT["rewards"],
            _SYNTHETIC_ROLLOUT["dones"],
            _SYNTHETIC_ROLLOUT["values"],
            _SYNTHETIC_ROLLOUT["last_value"],
            gamma=0.9,
            gae_lambda=0.8,
            truncation_values=torch.zeros(2),
        )


def test_rollout_builder_rejects_a_partially_bootstrapped_rollout() -> None:
    builder = RolloutBuilder()
    common = {
        "observations": torch.zeros(1, 1),
        "previous_particles": torch.zeros(1, 2, 1),
        "previous_actions": torch.zeros(1, 1),
        "initial": torch.ones(1, dtype=torch.bool),
        "particles": torch.zeros(1, 2, 1),
        "ula_initial_noise": torch.zeros(1, 2, 1),
        "ula_step_noises": torch.zeros(1, 2, 2, 1),
        "actions": torch.zeros(1),
        "old_log_prob": torch.zeros(1),
        "old_values": torch.zeros(1),
        "rewards": torch.zeros(1),
        "dones": torch.zeros(1, dtype=torch.bool),
    }
    builder.append(**common, truncation_values=torch.zeros(1))
    with pytest.raises(ValueError, match="bootstrapped"):
        builder.append(**common)
    with pytest.raises(ValueError, match="rewards' \\[batch\\] shape"):
        RolloutBuilder().append(**common, truncation_values=torch.zeros(2))


def test_diffusion_advantage_downweights_the_noisiest_step() -> None:
    weights = diffusion_advantage_weights(4, 0.5)
    torch.testing.assert_close(weights, torch.tensor([0.125, 0.25, 0.5, 1.0]))


def test_rollout_builder_rejects_categorical_then_diffusion_entries() -> None:
    builder = RolloutBuilder()
    common = {
        "observations": torch.zeros(1, 1),
        "previous_particles": torch.zeros(1, 2, 1),
        "previous_actions": torch.zeros(1, 1),
        "initial": torch.ones(1, dtype=torch.bool),
        "particles": torch.zeros(1, 2, 1),
        "ula_initial_noise": torch.zeros(1, 2, 1),
        "ula_step_noises": torch.zeros(1, 2, 2, 1),
        "actions": torch.zeros(1),
        "old_log_prob": torch.zeros(1),
        "old_values": torch.zeros(1),
        "rewards": torch.zeros(1),
        "dones": torch.zeros(1, dtype=torch.bool),
    }
    builder.append(**common, diffusion_chains=None)
    with pytest.raises(ValueError, match="cannot mix"):
        builder.append(**common, diffusion_chains=torch.zeros(1, 3, 1))


def test_rollout_builder_stores_detached_ula_noises_and_flattens_them() -> None:
    builder = RolloutBuilder()
    initial_noises = []
    step_noises = []
    for time_index in range(2):
        initial_noise = (
            torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) + 100 * time_index
        ).requires_grad_()
        step_noise = (
            torch.arange(48, dtype=torch.float32).reshape(2, 4, 3, 2) + 1000 * time_index
        ).requires_grad_()
        initial_noises.append(initial_noise.detach().clone())
        step_noises.append(step_noise.detach().clone())
        builder.append(
            observations=torch.zeros(2, 1),
            previous_particles=torch.zeros(2, 3, 2),
            previous_actions=torch.zeros(2, 1),
            initial=torch.full((2,), time_index == 0, dtype=torch.bool),
            particles=torch.zeros(2, 3, 2),
            ula_initial_noise=initial_noise,
            ula_step_noises=step_noise,
            actions=torch.zeros(2),
            old_log_prob=torch.zeros(2),
            old_values=torch.zeros(2),
            rewards=torch.ones(2),
            dones=torch.zeros(2, dtype=torch.bool),
        )

    rollout = builder.finish(last_value=torch.zeros(2), gamma=0.99, gae_lambda=0.95)

    assert rollout.ula_initial_noise.shape == (2, 2, 3, 2)
    assert rollout.ula_step_noises.shape == (2, 2, 4, 3, 2)
    assert not rollout.ula_initial_noise.requires_grad
    assert not rollout.ula_step_noises.requires_grad
    torch.testing.assert_close(rollout.ula_initial_noise, torch.stack(initial_noises))
    torch.testing.assert_close(rollout.ula_step_noises, torch.stack(step_noises))

    flattened = rollout.flatten()
    assert flattened["ula_initial_noise"] is not None
    assert flattened["ula_step_noises"] is not None
    assert flattened["ula_initial_noise"].shape == (4, 3, 2)
    assert flattened["ula_step_noises"].shape == (4, 4, 3, 2)


@pytest.mark.parametrize(
    ("initial_shape", "step_shape", "message"),
    [
        ((2, 2, 2), (2, 4, 3, 2), "initial noise"),
        ((2, 3, 2), (2, 4, 2, 2), "step noises"),
        ((2, 3, 2), (2, 0, 3, 2), "step noises"),
    ],
)
def test_rollout_builder_validates_ula_noise_shapes(
    initial_shape: tuple[int, ...],
    step_shape: tuple[int, ...],
    message: str,
) -> None:
    builder = RolloutBuilder()
    with pytest.raises(ValueError, match=message):
        builder.append(
            observations=torch.zeros(2, 1),
            previous_particles=torch.zeros(2, 3, 2),
            previous_actions=torch.zeros(2, 1),
            initial=torch.ones(2, dtype=torch.bool),
            particles=torch.zeros(2, 3, 2),
            ula_initial_noise=torch.zeros(initial_shape),
            ula_step_noises=torch.zeros(step_shape),
            actions=torch.zeros(2),
            old_log_prob=torch.zeros(2),
            old_values=torch.zeros(2),
            rewards=torch.zeros(2),
            dones=torch.zeros(2, dtype=torch.bool),
        )
