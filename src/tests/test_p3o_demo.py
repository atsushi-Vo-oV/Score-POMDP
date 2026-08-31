"""Demonstration seeding for the P3O-style weighted-ML trainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.envs import make_env
from sb_pomdp.policies import DiffusionPolicy
from sb_pomdp.train import _PassiveKalmanDemo

P3O_TASK = {
    "horizon": 30,
    "observation_noise_std": 0.01,
    "dimension": 2,
    "light_position": 5.0,
    "initial_mean": 2.0,
    "initial_std": 1.0,
    "state_cost": 0.0,
    "action_cost": 0.001,
    "goal_radius": 1e-06,
    "goal_bonus": 0.0,
    "noise_gain": 5.0,
    "terminal_cost": 10.0,
    "process_noise_std": 0.01,
}


def _policy(action_dim: int = 2, steps: int = 5) -> DiffusionPolicy:
    torch.manual_seed(0)
    return DiffusionPolicy(
        8,
        [-1.0] * action_dim,
        [1.0] * action_dim,
        [16],
        num_steps=steps,
        beta_start=0.01,
        beta_end=0.2,
        min_std=0.05,
    )


def test_forward_noised_chain_matches_the_sample_layout() -> None:
    policy = _policy()
    actions = torch.tensor([[0.3, -0.7], [0.0, 0.9], [-0.99, 0.5]])
    chain = policy.forward_noised_chain(actions, generator=torch.Generator().manual_seed(1))
    assert chain.shape == (3, policy.num_steps + 1, 2)
    # The clean endpoint reproduces the demonstration action through tanh.
    recovered = torch.tanh(chain[:, -1]) * policy.action_scale + policy.action_bias
    assert torch.allclose(recovered, actions.clamp(-1 + 1e-6, 1 - 1e-6), atol=1e-5)
    # Earlier elements are noisier: index 0 has the smallest clean fraction.
    assert policy.alpha_bars[-1] < policy.alpha_bars[0]


def test_forward_noised_chain_supports_the_imitation_loss() -> None:
    policy = _policy()
    condition = torch.randn(4, 8)
    actions = torch.rand(4, 2) * 1.6 - 0.8
    chain = policy.forward_noised_chain(actions, generator=torch.Generator().manual_seed(2))
    log_prob, _entropy = policy.log_prob_chain(condition, chain)
    assert log_prob.shape == (4, policy.num_steps)
    loss = -log_prob.sum(dim=-1).mean()
    loss.backward()
    gradient_norms = [
        parameter.grad.abs().sum()
        for parameter in policy.denoiser.parameters()
        if parameter.grad is not None
    ]
    assert gradient_norms and all(torch.isfinite(value) for value in gradient_norms)
    assert float(sum(gradient_norms)) > 0.0


def test_passive_kalman_demo_beats_doing_nothing_on_the_p3o_task() -> None:
    episodes, horizon = 30, P3O_TASK["horizon"]
    returns = []
    for episode in range(episodes):
        environment = make_env("light_dark", P3O_TASK, seed=4000 + episode)
        observation, _info = environment.reset(seed=4000 + episode)
        demo = _PassiveKalmanDemo(environment, slots=1)
        total, initial = 0.0, np.array([True])
        for _ in range(horizon):
            action = demo.actions(observation[None, :], initial)[0]
            observation, reward, terminated, truncated, _info = environment.step(action)
            total += reward
            initial = np.array([False])
            if terminated or truncated:
                break
        returns.append(total)
    average = float(np.mean(returns))
    # Passive-Kalman reference is about -15.5; doing nothing costs about -80.
    assert average > -40.0


def test_demo_slot_config_is_validated() -> None:
    config = load_config("config/local.json")
    assert config.to_dict()["ppo"]["p3o_demo_slots"] == 0
    with pytest.raises(ConfigError):
        config.with_overrides({"ppo.p3o_demo_slots": -1})
    accepted = config.with_overrides({"ppo.p3o_demo_slots": 8})
    assert accepted["ppo"]["p3o_demo_slots"] == 8
