from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import fields
from typing import Any

import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from sb_pomdp.buffer import BeliefReplayPrefix, RolloutBatch
from sb_pomdp.config import ConfigError, ExperimentConfig, load_config
from sb_pomdp.ppo import PPOMetrics, update_ppo


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters: Any, *, lr: float) -> None:
        super().__init__(parameters, lr=lr)
        self.step_calls = 0

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        self.step_calls += 1
        return super().step(closure)


class _RecordingIdentityBelief:
    def __init__(self) -> None:
        self.replay_batch_sizes: list[int] = []

    def particles_from_noise(
        self,
        observation: torch.Tensor,
        *_: torch.Tensor,
        initial_noise: torch.Tensor,
        **__: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.replay_batch_sizes.append(int(observation.shape[0]))
        return initial_noise, initial_noise


class _TinyCategoricalActorCritic(nn.Module):
    action_kind = "discrete"

    def __init__(self) -> None:
        super().__init__()
        self.belief = _RecordingIdentityBelief()
        self.categorical_head = nn.Linear(1, 2)
        self.value_head = nn.Linear(1, 1)
        with torch.no_grad():
            self.categorical_head.weight.copy_(torch.tensor([[0.3], [-0.2]]))
            self.categorical_head.bias.copy_(torch.tensor([0.1, -0.1]))
            self.value_head.weight.fill_(0.25)
            self.value_head.bias.fill_(-0.15)

    @staticmethod
    def encode(particles: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        del scores
        return particles.mean(dim=1)

    def value(self, condition: torch.Tensor) -> torch.Tensor:
        return self.value_head(condition).squeeze(-1)


def _empty_prefix() -> BeliefReplayPrefix:
    return BeliefReplayPrefix(
        observations=torch.empty(0, 1),
        previous_actions=torch.empty(0, 1),
        initial=torch.empty(0, dtype=torch.bool),
        ula_initial_noise=torch.empty(0, 1, 1),
        ula_step_noises=torch.empty(0, 1, 1, 1),
    )


def _categorical_rollout(model: _TinyCategoricalActorCritic) -> RolloutBatch:
    rollout_steps, num_environments = 2, 4
    particles = torch.tensor(
        [
            [[[-1.2]], [[-0.4]], [[0.3]], [[1.1]]],
            [[[-0.8]], [[0.1]], [[0.7]], [[1.6]]],
        ]
    )
    actions = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    condition = particles.reshape(rollout_steps * num_environments, 1)
    with torch.no_grad():
        distribution = Categorical(logits=model.categorical_head(condition))
        old_log_prob = distribution.log_prob(actions.reshape(-1)).reshape(
            rollout_steps,
            num_environments,
        )
        old_values = model.value(condition).reshape(rollout_steps, num_environments)

    prefix = _empty_prefix()
    return RolloutBatch(
        observations=torch.zeros(rollout_steps, num_environments, 1),
        previous_particles=torch.zeros_like(particles),
        previous_actions=torch.zeros(rollout_steps, num_environments, 1),
        initial=torch.ones(rollout_steps, num_environments, dtype=torch.bool),
        particles=particles,
        ula_initial_noise=particles.clone(),
        ula_step_noises=torch.zeros(rollout_steps, num_environments, 1, 1, 1),
        actions=actions,
        old_log_prob=old_log_prob,
        old_values=old_values,
        rewards=torch.zeros(rollout_steps, num_environments),
        dones=torch.zeros(rollout_steps, num_environments, dtype=torch.bool),
        advantages=torch.tensor(
            [[1.0, -0.7, 0.4, 1.3], [-1.1, 0.2, 0.9, -0.3]]
        ),
        returns=old_values
        + torch.tensor([[0.4, -0.2, 0.1, 0.7], [-0.5, 0.3, -0.6, 0.2]]),
        diffusion_chains=None,
        pre_tanh_actions=None,
        belief_prefixes=(prefix,) * num_environments,
    )


def _ppo_config(sequence_microbatch_size: int) -> dict[str, Any]:
    return {
        "minibatch_size": 8,
        "sequence_microbatch_size": sequence_microbatch_size,
        "normalize_advantage": False,
        "epochs": 1,
        "clip_coef": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 100.0,
    }


def _assert_metrics_close(actual: PPOMetrics, expected: PPOMetrics) -> None:
    for field in fields(PPOMetrics):
        torch.testing.assert_close(
            torch.tensor(getattr(actual, field.name)),
            torch.tensor(getattr(expected, field.name)),
            atol=2e-7,
            rtol=2e-6,
            msg=lambda message, name=field.name: f"metric {name}: {message}",
        )


@pytest.mark.parametrize("gradient_mode", ["full", "tbptt_1"])
def test_sequence_microbatch_accumulation_matches_one_effective_minibatch(
    gradient_mode: str,
) -> None:
    reference_model = _TinyCategoricalActorCritic()
    accumulated_model = copy.deepcopy(reference_model)
    rollout = _categorical_rollout(reference_model)
    reference_optimizer = _CountingSGD(reference_model.parameters(), lr=0.05)
    accumulated_optimizer = _CountingSGD(accumulated_model.parameters(), lr=0.05)
    model_config = {
        "belief_gradient_mode": gradient_mode,
        "langevin_step_size": 0.1,
    }

    reference_metrics = update_ppo(
        reference_model,  # type: ignore[arg-type]
        reference_optimizer,
        rollout,
        model_config,
        _ppo_config(sequence_microbatch_size=8),
        generator=torch.Generator().manual_seed(1301),
    )
    accumulated_metrics = update_ppo(
        accumulated_model,  # type: ignore[arg-type]
        accumulated_optimizer,
        rollout,
        model_config,
        _ppo_config(sequence_microbatch_size=2),
        generator=torch.Generator().manual_seed(1301),
    )

    assert reference_optimizer.step_calls == 1
    assert accumulated_optimizer.step_calls == 1
    assert reference_model.belief.replay_batch_sizes == [4, 4]
    assert accumulated_model.belief.replay_batch_sizes == [1] * 8
    for accumulated_parameter, reference_parameter in zip(
        accumulated_model.parameters(),
        reference_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            accumulated_parameter,
            reference_parameter,
            atol=2e-7,
            rtol=2e-6,
        )
    _assert_metrics_close(accumulated_metrics, reference_metrics)


def test_production_uses_one_environment_per_sequence_microbatch() -> None:
    ppo = load_config("config/production.json")["ppo"]

    assert ppo["rollout_steps"] == 128
    assert ppo["minibatch_size"] == 512
    assert ppo["sequence_microbatch_size"] == 128
    assert ppo["minibatch_size"] // ppo["rollout_steps"] == 4
    assert ppo["sequence_microbatch_size"] // ppo["rollout_steps"] == 1


def test_sequence_microbatch_size_is_required_by_the_closed_schema() -> None:
    raw = load_config("config/local.json").to_dict()
    raw["ppo"].pop("sequence_microbatch_size")

    with pytest.raises(ConfigError, match="sequence_microbatch_size"):
        ExperimentConfig(raw)


@pytest.mark.parametrize(
    ("sequence_microbatch_size", "message"),
    [
        (0, "positive"),
        (64, "at least.*rollout_steps"),
        (192, "divisible"),
        (384, "divisible"),
        (640, "must not exceed.*minibatch_size"),
    ],
)
def test_invalid_sequence_microbatch_sizes_fail_fast(
    sequence_microbatch_size: int,
    message: str,
) -> None:
    raw = load_config("config/production.json").to_dict()
    raw["ppo"]["sequence_microbatch_size"] = sequence_microbatch_size

    with pytest.raises(ConfigError, match=message):
        ExperimentConfig(raw)
