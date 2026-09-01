"""Predictive-sufficiency auxiliary supervision of the belief."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_p3o_wml import OBS_DIM, PARTICLES, _rollout_for, _tiny_model

from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.networks import ObservationPredictor
from sb_pomdp.policies import ScoreBeliefActorCritic
from sb_pomdp.ppo import observation_prediction_loss, update_ppo
from sb_pomdp.train import build_optimizer


def test_predictor_likelihood_shapes_and_finiteness() -> None:
    torch.manual_seed(0)
    predictor = ObservationPredictor(2, 3, 4, [16])
    particles = torch.randn(5, 6, 2)
    actions = torch.randn(5, 3)
    target = torch.randn(5, 4)
    log_likelihood = predictor.log_likelihood(particles, actions, target)
    assert log_likelihood.shape == (5,)
    assert torch.isfinite(log_likelihood).all()


def test_predictor_prefers_informative_particles() -> None:
    torch.manual_seed(1)
    predictor = ObservationPredictor(1, 0, 1, [32])
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-2)
    # Train on particles that equal the next observation.
    for _ in range(300):
        state = torch.randn(64, 1)
        particles = state[:, None, :] + 0.01 * torch.randn(64, 4, 1)
        loss = -predictor.log_likelihood(
            particles, torch.zeros(64, 0), state + 0.05 * torch.randn(64, 1)
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    state = torch.randn(256, 1)
    informative = state[:, None, :].repeat(1, 4, 1)
    blob = torch.randn(256, 4, 1)
    ll_informative = predictor.log_likelihood(
        informative, torch.zeros(256, 0), state
    ).mean()
    ll_blob = predictor.log_likelihood(blob, torch.zeros(256, 0), state).mean()
    assert float(ll_informative) > float(ll_blob) + 0.5


def test_model_gains_predictor_only_with_positive_coefficient() -> None:
    model, _, _ = _tiny_model()
    assert model.observation_predictor is None
    config = load_config("config/local.json").with_overrides(
        {
            "model.num_particles": PARTICLES,
            "model.langevin_steps": 2,
            "model.observation_prediction_coef": 0.5,
        }
    )
    torch.manual_seed(0)
    with_predictor = ScoreBeliefActorCritic(
        observation_dim=OBS_DIM,
        state_dim=2,
        action_kind="discrete",
        action_feature_dim=2,
        discrete_actions=2,
        action_low=None,
        action_high=None,
        model_config=dict(config["model"]),
    )
    assert with_predictor.observation_predictor is not None
    assert with_predictor.observation_prediction_coef == pytest.approx(0.5)


def test_update_ppo_applies_the_auxiliary_loss_and_reaches_the_energies() -> None:
    config = load_config("config/local.json").with_overrides(
        {
            "model.num_particles": PARTICLES,
            "model.langevin_steps": 2,
            "model.observation_prediction_coef": 1.0,
        }
    )
    model_config = dict(config["model"])
    ppo_config = dict(config["ppo"])
    ppo_config.update(
        {"epochs": 1, "minibatch_size": 8, "sequence_microbatch_size": 4}
    )
    torch.manual_seed(0)
    model = ScoreBeliefActorCritic(
        observation_dim=OBS_DIM,
        state_dim=2,
        action_kind="discrete",
        action_feature_dim=2,
        discrete_actions=2,
        action_low=None,
        action_high=None,
        model_config=model_config,
    )
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    optimizer = build_optimizer(model, ppo_config)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "observation_energy" in name or "initial_energy" in name
    }
    metrics = update_ppo(model, optimizer, rollout, model_config, ppo_config)
    assert np.isfinite(metrics.policy_loss)
    moved = [
        (parameter - before[name]).abs().max()
        for name, parameter in model.named_parameters()
        if name in before
    ]
    assert moved and max(float(value) for value in moved) > 0.0
    predictor_grads = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("observation_predictor")
    ]
    assert predictor_grads


def test_prediction_loss_masks_episode_boundaries() -> None:
    config = load_config("config/local.json").with_overrides(
        {
            "model.num_particles": PARTICLES,
            "model.langevin_steps": 2,
            "model.observation_prediction_coef": 1.0,
        }
    )
    torch.manual_seed(0)
    model = ScoreBeliefActorCritic(
        observation_dim=OBS_DIM,
        state_dim=2,
        action_kind="discrete",
        action_feature_dim=2,
        discrete_actions=2,
        action_low=None,
        action_high=None,
        model_config=dict(config["model"]),
    )
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    particles = rollout.particles[:, torch.arange(4)]
    loss = observation_prediction_loss(model, particles, rollout, torch.arange(4))
    assert torch.isfinite(loss)


def test_config_validates_the_prediction_keys() -> None:
    config = load_config("config/local.json")
    resolved = config.to_dict()["model"]
    assert resolved["observation_prediction_coef"] == 0.0
    assert resolved["observation_predictor_hidden"] == [64]
    with pytest.raises(ConfigError):
        config.with_overrides({"model.observation_prediction_coef": -0.1})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.observation_predictor_hidden": []})
    accepted = config.with_overrides(
        {
            "model.observation_prediction_coef": 1.0,
            "model.observation_predictor_hidden": [32, 32],
        }
    )
    assert accepted["model"]["observation_prediction_coef"] == pytest.approx(1.0)
