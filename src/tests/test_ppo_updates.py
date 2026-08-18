from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from sb_pomdp.baselines import GRUActorCritic, ObservationActorCritic
from sb_pomdp.buffer import BeliefReplayPrefix, RolloutBatch
from sb_pomdp.ppo import _clipped_policy_loss as score_clipped_policy_loss
from sb_pomdp.ppo import update_ppo
from sb_pomdp.train_baselines import (
    BaselineRollout,
    RecurrentReplayPrefix,
    _update_feedforward,
    _update_recurrent,
)
from sb_pomdp.train_baselines import _clipped_policy_loss as baseline_clipped_policy_loss


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters: Any, *, lr: float) -> None:
        super().__init__(parameters, lr=lr)
        self.step_calls = 0

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        self.step_calls += 1
        return super().step(closure)


def _ppo_config() -> dict[str, Any]:
    return {
        "minibatch_size": 1,
        "normalize_advantage": False,
        "epochs": 2,
        "clip_coef": 0.2,
        "value_coef": 0.0,
        "entropy_coef": 0.0,
        "max_grad_norm": 100.0,
    }


@pytest.mark.parametrize(
    "loss_function",
    [score_clipped_policy_loss, baseline_clipped_policy_loss],
)
def test_policy_ratio_uses_the_unclamped_log_ratio(loss_function: Callable[..., Any]) -> None:
    _, approximate_kl, _ = loss_function(
        torch.tensor([30.0]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        0.2,
    )

    assert approximate_kl > torch.exp(torch.tensor(25.0))


class _IdentityBelief:
    def score_at(self, particles: torch.Tensor, *_: torch.Tensor, **__: bool) -> torch.Tensor:
        return particles

    def particles_from_noise(
        self,
        *_: torch.Tensor,
        initial_noise: torch.Tensor,
        **__: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return initial_noise, initial_noise


class _TinyScoreActorCritic(nn.Module):
    action_kind = "discrete"

    def __init__(self) -> None:
        super().__init__()
        self.belief = _IdentityBelief()
        self.categorical_head = nn.Linear(1, 2, bias=False)
        self.value_head = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.categorical_head.weight)
        nn.init.zeros_(self.value_head.weight)

    @staticmethod
    def encode(particles: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        del scores
        return particles.mean(dim=1)

    def value(self, condition: torch.Tensor) -> torch.Tensor:
        return self.value_head(condition).squeeze(-1)


def test_score_ppo_applies_every_configured_minibatch() -> None:
    model = _TinyScoreActorCritic()
    optimizer = _CountingSGD(model.parameters(), lr=0.1)
    particles = torch.ones(1, 2, 1, 1)
    empty_prefix = BeliefReplayPrefix(
        observations=torch.empty(0, 1),
        previous_actions=torch.empty(0, 1),
        initial=torch.empty(0, dtype=torch.bool),
        ula_initial_noise=torch.empty(0, 1, 1),
        ula_step_noises=torch.empty(0, 1, 1, 1),
    )
    rollout = RolloutBatch(
        observations=torch.zeros(1, 2, 1),
        previous_particles=torch.zeros_like(particles),
        previous_actions=torch.zeros(1, 2, 1),
        initial=torch.ones(1, 2, dtype=torch.bool),
        particles=particles,
        ula_initial_noise=particles.clone(),
        ula_step_noises=torch.zeros(1, 2, 1, 1, 1),
        actions=torch.zeros(1, 2, dtype=torch.long),
        old_log_prob=torch.full((1, 2), -torch.log(torch.tensor(2.0))),
        old_values=torch.zeros(1, 2),
        rewards=torch.zeros(1, 2),
        dones=torch.zeros(1, 2, dtype=torch.bool),
        advantages=torch.ones(1, 2),
        returns=torch.zeros(1, 2),
        diffusion_chains=None,
        pre_tanh_actions=None,
        belief_prefixes=(empty_prefix, empty_prefix),
    )

    metrics = update_ppo(
        model,
        optimizer,
        rollout,
        {"belief_gradient_mode": "full", "langevin_step_size": 0.1},
        _ppo_config(),
    )  # type: ignore[arg-type]

    assert optimizer.step_calls == 4
    assert torch.isfinite(torch.tensor(metrics.approximate_kl))


def _feedforward_rollout(model: ObservationActorCritic) -> BaselineRollout:
    inputs = torch.ones(1, 2, 1)
    actions = torch.zeros(1, 2, dtype=torch.long)
    with torch.no_grad():
        evaluation = model.evaluate_actions(inputs.reshape(2, 1), actions.reshape(2))
    return BaselineRollout(
        inputs=inputs,
        actions=actions,
        old_log_prob=evaluation.log_prob.reshape(1, 2),
        old_values=evaluation.value.reshape(1, 2),
        rewards=torch.zeros(1, 2),
        dones=torch.zeros(1, 2, dtype=torch.bool),
        advantages=torch.ones(1, 2),
        returns=evaluation.value.reshape(1, 2),
        episode_starts=torch.ones(1, 2, dtype=torch.bool),
        pre_tanh_actions=None,
        diffusion_chains=None,
        hidden_states=None,
    )


def test_feedforward_ppo_applies_every_configured_minibatch() -> None:
    torch.manual_seed(1)
    model = ObservationActorCritic(
        1,
        action_kind="discrete",
        hidden_dims=[2],
        discrete_actions=2,
    )
    optimizer = _CountingSGD(model.parameters(), lr=0.1)

    metrics = _update_feedforward(model, optimizer, _feedforward_rollout(model), _ppo_config())

    assert optimizer.step_calls == 4
    assert torch.isfinite(torch.tensor(metrics.approximate_kl))


def _recurrent_rollout(model: GRUActorCritic) -> BaselineRollout:
    inputs = torch.ones(2, 2, 1)
    actions = torch.zeros(2, 2, dtype=torch.long)
    episode_starts = torch.tensor([[True, True], [False, False]])
    hidden_states = torch.zeros(2, 1, 2, 2)
    with torch.no_grad():
        features, _ = model.encode_sequence(inputs, hidden_states[0], episode_starts)
        evaluation = model.heads.evaluate_actions(features.reshape(4, 2), actions.reshape(4))
    empty_prefix = RecurrentReplayPrefix(
        inputs=torch.empty(0, 1),
        episode_starts=torch.empty(0, dtype=torch.bool),
    )
    return BaselineRollout(
        inputs=inputs,
        actions=actions,
        old_log_prob=evaluation.log_prob.reshape(2, 2),
        old_values=evaluation.value.reshape(2, 2),
        rewards=torch.zeros(2, 2),
        dones=torch.zeros(2, 2, dtype=torch.bool),
        advantages=torch.ones(2, 2),
        returns=evaluation.value.reshape(2, 2),
        episode_starts=episode_starts,
        pre_tanh_actions=None,
        diffusion_chains=None,
        hidden_states=hidden_states,
        recurrent_prefixes=(empty_prefix, empty_prefix),
    )


def test_recurrent_ppo_applies_every_configured_minibatch() -> None:
    torch.manual_seed(2)
    model = GRUActorCritic(
        1,
        action_kind="discrete",
        hidden_dims=[2],
        recurrent_hidden_dim=2,
        discrete_actions=2,
    )
    optimizer = _CountingSGD(model.parameters(), lr=0.1)

    metrics = _update_recurrent(
        model,
            optimizer,
            _recurrent_rollout(model),
            {"diffusion_advantage_discount": 0.95},
            {**_ppo_config(), "minibatch_size": 2},
        temporal_gradient_mode="full",
    )

    assert optimizer.step_calls == 4
    assert torch.isfinite(torch.tensor(metrics.approximate_kl))
