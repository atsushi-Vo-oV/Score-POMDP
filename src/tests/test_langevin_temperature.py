"""Tempered, learned-schedule, and warm-start invariants of the ULA belief."""

from __future__ import annotations

import math

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.policies import ScoreBeliefActorCritic


def _belief(**overrides: object) -> EnergyBelief:
    return EnergyBelief(3, 2, 2, [8], **overrides)  # type: ignore[arg-type]


def _context(batch: int = 2, num_particles: int = 4, steps: int = 4):
    torch.manual_seed(0)
    observation = torch.randn(batch, 3)
    previous_particles = torch.randn(batch, num_particles, 2)
    previous_action = torch.randn(batch, 2)
    initial = torch.zeros(batch, dtype=torch.bool)
    initial_noise = torch.randn(batch, num_particles, 2)
    langevin_noise = torch.randn(batch, steps, num_particles, 2)
    return (
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise,
        langevin_noise,
    )


def _run(model: EnergyBelief, context, *, langevin_noise=None, track_grad=False):
    observation, previous, action, initial, initial_noise, default_noise = context
    return model.particles_from_noise(
        observation,
        previous,
        action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=default_noise if langevin_noise is None else langevin_noise,
        step_size=0.05,
        track_grad=track_grad,
    )


def test_default_flags_reproduce_the_original_update_exactly() -> None:
    torch.manual_seed(1)
    original = _belief()
    torch.manual_seed(1)
    flagged = _belief(
        langevin_temperature=1.0,
        langevin_temperature_learnable=False,
        langevin_warm_start=False,
    )
    context = _context()
    particles_a, scores_a = _run(original, context)
    particles_b, scores_b = _run(flagged, context)
    assert torch.equal(particles_a, particles_b)
    assert torch.equal(scores_a, scores_b)
    assert "langevin_log_temperature" not in flagged.state_dict()


def test_fixed_temperature_equals_prescaling_the_stored_noise() -> None:
    temperature = 0.25
    torch.manual_seed(2)
    tempered = _belief(langevin_temperature=temperature)
    torch.manual_seed(2)
    reference = _belief()
    context = _context()
    langevin_noise = context[-1]
    particles_t, _ = _run(tempered, context)
    particles_ref, _ = _run(
        reference, context, langevin_noise=langevin_noise * math.sqrt(temperature)
    )
    assert torch.allclose(particles_t, particles_ref, atol=1e-6)


def test_learnable_schedule_is_a_parameter_receiving_gradient() -> None:
    torch.manual_seed(3)
    model = _belief(
        langevin_temperature=0.5,
        langevin_temperature_learnable=True,
        langevin_steps=4,
    )
    assert isinstance(model.langevin_log_temperature, torch.nn.Parameter)
    assert model.langevin_log_temperature.shape == (4,)
    assert torch.allclose(
        model.langevin_log_temperature,
        torch.full((4,), math.log(0.5)),
    )
    assert "langevin_log_temperature" in model.state_dict()

    context = _context()
    particles, _ = _run(model, context, track_grad=True)
    (gradient,) = torch.autograd.grad(
        particles.sum(), model.langevin_log_temperature, allow_unused=False
    )
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    with pytest.raises(ValueError, match="schedule length"):
        _run(model, _context(steps=3))


def test_warm_start_carries_previous_particles_and_respects_episode_starts() -> None:
    torch.manual_seed(4)
    model = _belief(langevin_warm_start=True, langevin_temperature=1e-3)
    observation, previous, action, _, initial_noise, _ = _context()
    initial = torch.tensor([True, False])
    tiny_noise = torch.randn(2, 4, 4, 2)
    particles, _ = model.particles_from_noise(
        observation,
        previous,
        action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=tiny_noise,
        step_size=1e-8,
        track_grad=False,
    )
    assert torch.allclose(particles[0], initial_noise[0], atol=1e-3)
    assert torch.allclose(particles[1], previous[1], atol=1e-3)

    mismatched = previous[:, :3]
    with pytest.raises(ValueError, match="particle counts"):
        model.particles_from_noise(
            observation,
            mismatched,
            action,
            initial,
            initial_noise=initial_noise,
            langevin_noise=tiny_noise,
            step_size=1e-8,
            track_grad=False,
        )


def test_flagged_update_stays_a_deterministic_function_of_its_inputs() -> None:
    torch.manual_seed(5)
    model = _belief(
        langevin_warm_start=True,
        langevin_temperature=0.1,
        langevin_temperature_learnable=True,
        langevin_steps=4,
    )
    context = _context()
    rollout_particles, _ = _run(model, context, track_grad=False)
    replayed_particles, _ = _run(model, context, track_grad=True)
    assert torch.allclose(rollout_particles, replayed_particles, atol=1e-6)


def test_learnable_step_size_schedule_is_a_parameter_receiving_gradient() -> None:
    torch.manual_seed(6)
    model = _belief(
        langevin_step_size_learnable=True,
        langevin_step_size=0.05,
        langevin_steps=4,
    )
    torch.manual_seed(6)
    reference = _belief()
    assert isinstance(model.langevin_log_step_size, torch.nn.Parameter)
    assert model.langevin_log_step_size.shape == (4,)
    assert torch.allclose(
        model.langevin_log_step_size, torch.full((4,), math.log(0.05))
    )
    assert "langevin_log_step_size" in model.state_dict()
    assert "langevin_log_step_size" not in reference.state_dict()

    context = _context()
    particles, _ = _run(model, context)
    reference_particles, _ = _run(reference, context)
    assert torch.allclose(particles, reference_particles, atol=1e-5)

    tracked, _ = _run(model, context, track_grad=True)
    (gradient,) = torch.autograd.grad(
        tracked.sum(), model.langevin_log_step_size, allow_unused=False
    )
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    with pytest.raises(ValueError, match="schedule length"):
        _run(model, _context(steps=3))


def test_constructor_rejects_invalid_temperature_settings() -> None:
    with pytest.raises(ValueError):
        _belief(langevin_temperature=0.0)
    with pytest.raises(ValueError):
        _belief(langevin_temperature=-1.0)
    with pytest.raises(ValueError):
        _belief(langevin_temperature_learnable=True)
    with pytest.raises(ValueError):
        _belief(langevin_step_size_learnable=True, langevin_steps=4)
    with pytest.raises(ValueError):
        _belief(
            langevin_step_size_learnable=True,
            langevin_step_size=0.05,
        )


def test_actor_critic_forwards_the_flags_to_its_belief() -> None:
    model_config = {
        "num_particles": 4,
        "langevin_steps": 3,
        "langevin_step_size": 0.05,
        "langevin_temperature": 0.1,
        "langevin_temperature_learnable": True,
        "langevin_warm_start": True,
        "langevin_step_size_learnable": True,
        "particle_clip": 0.0,
        "score_clip": 0.0,
        "energy_hidden": [8],
        "d_model": 8,
        "num_heads": 2,
        "num_transformer_layers": 1,
        "transformer_ff_dim": 16,
        "dropout": 0.0,
        "policy_hidden": [8],
    }
    model = ScoreBeliefActorCritic(
        observation_dim=3,
        state_dim=2,
        action_kind="discrete",
        action_feature_dim=2,
        discrete_actions=2,
        action_low=None,
        action_high=None,
        model_config=model_config,
    )
    assert model.belief.langevin_warm_start
    assert model.belief.langevin_temperature == pytest.approx(0.1)
    assert model.belief.langevin_log_temperature is not None
    assert model.belief.langevin_log_temperature.shape == (3,)
    assert model.belief.langevin_log_step_size is not None
    assert model.belief.langevin_log_step_size.shape == (3,)
    assert torch.allclose(
        model.belief.langevin_log_step_size, torch.full((3,), math.log(0.05))
    )


def test_config_schema_validates_the_temperature_keys() -> None:
    config = load_config("config/local.json")
    resolved = config.to_dict()
    assert resolved["model"]["langevin_temperature"] == 1.0
    assert resolved["model"]["langevin_temperature_learnable"] is False
    assert resolved["model"]["langevin_warm_start"] is False
    assert resolved["model"]["langevin_step_size_learnable"] is False
    with pytest.raises(ConfigError):
        config.with_overrides({"model.langevin_temperature": 0.0})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.langevin_warm_start": "yes"})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.langevin_step_size_learnable": "yes"})
    accepted = config.with_overrides(
        {
            "model.langevin_temperature": 0.1,
            "model.langevin_temperature_learnable": True,
            "model.langevin_warm_start": True,
            "model.langevin_step_size_learnable": True,
        }
    )
    assert accepted["model"]["langevin_temperature"] == pytest.approx(0.1)
