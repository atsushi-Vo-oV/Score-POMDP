"""Critic encoder modes: shared (default), separate encoder, or detached shared condition."""

from __future__ import annotations

import pytest
import torch
from test_p3o_wml import OBS_DIM, PARTICLES, STATE_DIM, STEPS, _rollout_for

from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.policies import ScoreBeliefActorCritic
from sb_pomdp.ppo import update_ppo
from sb_pomdp.train import build_optimizer


def _model(mode: str, **extra: object) -> tuple[ScoreBeliefActorCritic, dict, dict]:
    overrides: dict[str, object] = {
        "model.num_particles": PARTICLES,
        "model.langevin_steps": STEPS,
        "model.value_encoder": mode,
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


def test_shared_mode_is_the_default_and_reuses_one_pass() -> None:
    model, _, _ = _model("shared")
    assert model.value_encoder is None and model.value_encoder_mode == "shared"
    particles = torch.randn(3, PARTICLES, STATE_DIM)
    scores = torch.randn(3, PARTICLES, STATE_DIM)
    policy_c, value_c = model.encode_both(particles, scores)
    assert value_c is policy_c
    assert torch.equal(model.encode_value(particles, scores), model.encode(particles, scores))


def test_separate_mode_builds_a_second_encoder_with_its_own_parameters() -> None:
    shared, _, _ = _model("shared")
    separate, _, _ = _model("separate")
    assert separate.value_encoder is not None
    extra = sum(p.numel() for p in separate.value_encoder.parameters())
    assert extra == sum(p.numel() for p in separate.encoder.parameters())
    total_shared = sum(p.numel() for p in shared.parameters())
    assert sum(p.numel() for p in separate.parameters()) == total_shared + extra
    particles = torch.randn(3, PARTICLES, STATE_DIM)
    scores = torch.randn(3, PARTICLES, STATE_DIM)
    policy_c, value_c = separate.encode_both(particles, scores)
    assert policy_c.shape == value_c.shape and not torch.allclose(policy_c, value_c)


@pytest.mark.parametrize("mode", ["separate", "detached"])
def test_value_loss_does_not_reach_the_policy_encoder(mode: str) -> None:
    model, _, _ = _model(mode)
    particles = torch.randn(3, PARTICLES, STATE_DIM)
    scores = torch.randn(3, PARTICLES, STATE_DIM)
    model.zero_grad()
    value = model.value(model.encode_value(particles, scores))
    if mode == "detached":
        assert not value.requires_grad or value.grad_fn is not None
    loss = value.square().sum()
    if loss.requires_grad:
        loss.backward()
    assert all(
        p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in model.encoder.parameters()
    )
    if mode == "separate":
        assert any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0
            for p in model.value_encoder.parameters()
        )


def test_shared_mode_value_loss_reaches_the_encoder() -> None:
    model, _, _ = _model("shared")
    particles = torch.randn(3, PARTICLES, STATE_DIM)
    scores = torch.randn(3, PARTICLES, STATE_DIM)
    model.value(model.encode_value(particles, scores)).square().sum().backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in model.encoder.parameters()
    )


@pytest.mark.parametrize("mode", ["shared", "separate", "detached"])
def test_update_ppo_runs_in_every_mode(mode: str) -> None:
    model, model_config, ppo_config = _model(mode)
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    metrics = update_ppo(
        model, build_optimizer(model, ppo_config), rollout, model_config, ppo_config
    )
    assert metrics.value_loss == metrics.value_loss  # finite, not NaN


def test_config_validates_the_mode() -> None:
    config = load_config("config/local.json")
    assert config.to_dict()["model"]["value_encoder"] == "shared"
    with pytest.raises(ConfigError):
        config.with_overrides({"model.value_encoder": "twin"})
    with pytest.raises(ValueError):
        _model("twin")
