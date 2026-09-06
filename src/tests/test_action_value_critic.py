"""Action-value (Q) critic for PPO: heads, expected values, advantages, and updates."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_p3o_wml import OBS_DIM, PARTICLES, STATE_DIM, STEPS, _rollout_for

from sb_pomdp.baselines import make_baseline_actor_critic
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.networks import ActionValueHead, expected_action_value
from sb_pomdp.policies import ScoreBeliefActorCritic
from sb_pomdp.ppo import update_ppo
from sb_pomdp.train import build_optimizer


def test_head_shapes_and_gather() -> None:
    torch.manual_seed(0)
    discrete = ActionValueHead(6, [8], discrete_actions=3)
    condition = torch.randn(5, 6)
    all_values = discrete.all_values(condition)
    assert all_values.shape == (5, 3)
    actions = torch.tensor([0, 2, 1, 2, 0])
    assert torch.allclose(discrete(condition, actions), all_values[torch.arange(5), actions])
    continuous = ActionValueHead(6, [8], action_dim=2)
    assert continuous(condition, torch.randn(5, 2)).shape == (5,)
    with pytest.raises(ValueError):
        ActionValueHead(6, [8])
    with pytest.raises(ValueError):
        continuous(condition, torch.randn(5, 3))


def test_expected_value_is_exact_for_discrete_and_monte_carlo_for_continuous() -> None:
    torch.manual_seed(0)
    head = ActionValueHead(4, [8], discrete_actions=3)
    condition = torch.randn(7, 4)
    logits = torch.randn(7, 3)
    expected = expected_action_value(head, condition, categorical_logits=logits)
    manual = (torch.softmax(logits, -1) * head.all_values(condition)).sum(-1)
    assert torch.allclose(expected, manual)
    assert not expected.requires_grad

    continuous = ActionValueHead(4, [8], action_dim=2)
    sampler_calls: list[int] = []

    def sampler(c: torch.Tensor) -> torch.Tensor:
        sampler_calls.append(c.shape[0])
        return torch.zeros(c.shape[0], 2)

    value = expected_action_value(continuous, condition, sample_actions=sampler, num_samples=5)
    assert sampler_calls == [35] and value.shape == (7,)
    assert torch.allclose(value, continuous(condition, torch.zeros(7, 2)))


def _score_model(**extra: object) -> tuple[ScoreBeliefActorCritic, dict, dict]:
    overrides: dict[str, object] = {
        "model.num_particles": PARTICLES,
        "model.langevin_steps": STEPS,
        "model.critic_kind": "action",
        "model.q_value_samples": 4,
    }
    overrides.update(extra)
    config = load_config("config/local.json").with_overrides(overrides)
    model_config = dict(config["model"])
    ppo_config = dict(config["ppo"])
    ppo_config.update({"epochs": 1, "minibatch_size": 8, "sequence_microbatch_size": 4})
    torch.manual_seed(0)
    model = ScoreBeliefActorCritic(
        observation_dim=OBS_DIM,
        state_dim=STATE_DIM,
        action_kind="discrete",
        action_feature_dim=2,
        discrete_actions=2,
        action_low=None,
        action_high=None,
        model_config=model_config,
    )
    return model, model_config, ppo_config


def test_score_model_replaces_the_state_value_head() -> None:
    model, _, _ = _score_model()
    assert model.value_head is None and model.action_value_head is not None
    condition = torch.randn(3, model.action_value_head.network[0].in_features)
    value = model.value(condition)
    assert value.shape == (3,) and not value.requires_grad
    q = model.action_value(condition, torch.tensor([0, 1, 1]))
    assert q.shape == (3,) and q.requires_grad


@pytest.mark.parametrize("mix", [1.0, 0.5, 0.0])
def test_update_ppo_trains_the_action_value_critic(mix: float) -> None:
    model, model_config, ppo_config = _score_model()
    ppo_config["q_advantage_mix"] = mix
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    optimizer = build_optimizer(model, ppo_config)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("action_value_head")
    }
    assert before
    metrics = update_ppo(model, optimizer, rollout, model_config, ppo_config)
    assert np.isfinite(metrics.policy_loss) and np.isfinite(metrics.value_loss)
    moved = max(
        float((parameter - before[name]).abs().max())
        for name, parameter in model.named_parameters()
        if name in before
    )
    assert moved > 0.0


def test_baseline_heads_gain_an_action_value_critic() -> None:
    torch.manual_seed(0)
    for kind, extra in (("gru", {"recurrent_hidden_dim": 8}), ("observation", {})):
        model = make_baseline_actor_critic(
            kind,
            observation_dim=3,
            state_dim=2,
            action_kind="continuous",
            hidden_dims=[8],
            action_low=[-1.0, -1.0],
            action_high=[1.0, 1.0],
            head_hidden_dims=[8],
            critic_kind="action",
            q_value_samples=3,
            **extra,
        )
        heads = model.heads
        assert heads.value_head is None and heads.action_value_head is not None
        features = torch.randn(4, heads.action_value_head.network[0].in_features - 2)
        evaluation = heads.evaluate_actions(features, torch.zeros(4, 2))
        assert torch.allclose(evaluation.value, heads.action_value(features, torch.zeros(4, 2)))
        assert heads.expected_action_value(features).shape == (4,)


def test_config_validates_the_critic_keys() -> None:
    config = load_config("config/local.json")
    resolved = config.to_dict()
    assert resolved["model"]["critic_kind"] == "state"
    assert resolved["model"]["q_value_samples"] == 8
    assert resolved["ppo"]["q_advantage_mix"] == 1.0
    with pytest.raises(ConfigError):
        config.with_overrides({"model.critic_kind": "advantage"})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.q_value_samples": 0})
    with pytest.raises(ConfigError):
        config.with_overrides({"ppo.q_advantage_mix": 1.5})
    accepted = config.with_overrides({"model.critic_kind": "action", "ppo.q_advantage_mix": 0.5})
    assert accepted["model"]["critic_kind"] == "action"
