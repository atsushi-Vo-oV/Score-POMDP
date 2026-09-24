"""Set-level mixture-density auxiliary predictor (DeepSets over the particle set -> MDN)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_p3o_wml import OBS_DIM, PARTICLES, STATE_DIM, STEPS, _rollout_for

from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.networks import ObservationPredictor, SetMixturePredictor
from sb_pomdp.policies import ScoreBeliefActorCritic
from sb_pomdp.ppo import observation_prediction_loss, reward_prediction_loss, update_ppo
from sb_pomdp.train import build_optimizer


def test_shapes_permutation_invariance_and_reward_target_shape() -> None:
    torch.manual_seed(0)
    predictor = SetMixturePredictor(2, 3, 4, [16, 16], components=3)
    particles = torch.randn(5, 6, 2)
    actions = torch.randn(5, 3)
    target = torch.randn(5, 4)
    ll = predictor.log_likelihood(particles, actions, target)
    assert ll.shape == (5,) and torch.isfinite(ll).all()
    permuted = particles[:, torch.randperm(6)]
    assert torch.allclose(predictor.log_likelihood(permuted, actions, target), ll, atol=1e-5)
    log_w, means, log_std = predictor.mixture_parameters(particles, actions)
    assert log_w.shape == (5, 3) and means.shape == (5, 3, 4) and log_std.shape == (5, 3, 4)
    assert torch.allclose(log_w.exp().sum(-1), torch.ones(5), atol=1e-5)
    reward_predictor = SetMixturePredictor(2, 3, 1, [8], components=2)
    assert reward_predictor.log_likelihood(particles, actions, torch.randn(5)).shape == (5,)
    with pytest.raises(ValueError):
        predictor.log_likelihood(particles, actions, torch.randn(5, 3))


def test_gradient_reaches_the_particles() -> None:
    torch.manual_seed(0)
    predictor = SetMixturePredictor(2, 0, 2, [16])
    particles = torch.randn(4, 5, 2, requires_grad=True)
    predictor.log_likelihood(particles, torch.zeros(4, 0), torch.randn(4, 2)).sum().backward()
    assert particles.grad is not None and float(particles.grad.abs().sum()) > 0.0


def test_mixture_captures_a_bimodal_target_that_one_gaussian_per_particle_cannot() -> None:
    """Target is +-2 with equal probability, independent of the (identical) particles."""

    torch.manual_seed(2)
    mixture = SetMixturePredictor(1, 0, 1, [32], components=2)
    single = ObservationPredictor(1, 0, 1, [32])
    opt_m = torch.optim.Adam(mixture.parameters(), lr=1e-2)
    opt_s = torch.optim.Adam(single.parameters(), lr=1e-2)
    for _ in range(400):
        particles = torch.zeros(128, 4, 1)  # all particles identical: the per-particle
        target = torch.where(torch.rand(128, 1) < 0.5, -2.0, 2.0)  # mixture collapses to 1 Gaussian
        for predictor, opt in ((mixture, opt_m), (single, opt_s)):
            loss = -predictor.log_likelihood(particles, torch.zeros(128, 0), target).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    particles = torch.zeros(512, 4, 1)
    target = torch.where(torch.rand(512, 1) < 0.5, -2.0, 2.0)
    ll_mixture = float(mixture.log_likelihood(particles, torch.zeros(512, 0), target).mean())
    ll_single = float(single.log_likelihood(particles, torch.zeros(512, 0), target).mean())
    assert ll_mixture > ll_single + 0.5


def _model(kind: str) -> tuple[ScoreBeliefActorCritic, dict, dict]:
    config = load_config("config/local.json").with_overrides(
        {
            "model.num_particles": PARTICLES,
            "model.langevin_steps": STEPS,
            "model.observation_prediction_coef": 1.0,
            "model.reward_prediction_coef": 1.0,
            "model.aux_predictor_kind": kind,
            "model.aux_mixture_components": 3,
            "model.observation_predictor_hidden": [8, 8],
            "model.reward_predictor_hidden": [8],
        }
    )
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


def test_model_builds_set_mixture_predictors_and_update_ppo_trains_them() -> None:
    model, model_config, ppo_config = _model("set_mixture")
    assert isinstance(model.observation_predictor, SetMixturePredictor)
    assert isinstance(model.reward_predictor, SetMixturePredictor)
    assert model.observation_predictor.components == 3
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    indices = torch.arange(4)
    assert torch.isfinite(observation_prediction_loss(model, rollout.particles, rollout, indices))
    assert torch.isfinite(reward_prediction_loss(model, rollout.particles, rollout, indices))
    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n.startswith(("observation_predictor", "reward_predictor", "belief."))
    }
    metrics = update_ppo(
        model, build_optimizer(model, ppo_config), rollout, model_config, ppo_config
    )
    assert np.isfinite(metrics.policy_loss)
    moved = {
        n: float((p - before[n]).abs().max()) for n, p in model.named_parameters() if n in before
    }
    for prefix in ("observation_predictor", "reward_predictor", "belief."):
        assert max(v for n, v in moved.items() if n.startswith(prefix)) > 0.0, prefix


def test_default_kind_is_the_per_particle_predictor_and_config_validates() -> None:
    model, _, _ = _model("particle")
    assert isinstance(model.observation_predictor, ObservationPredictor)
    config = load_config("config/local.json")
    resolved = config.to_dict()["model"]
    assert resolved["aux_predictor_kind"] == "particle" and resolved["aux_mixture_components"] == 4
    with pytest.raises(ConfigError):
        config.with_overrides({"model.aux_predictor_kind": "flow"})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.aux_mixture_components": 0})
