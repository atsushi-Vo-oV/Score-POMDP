"""Soft (sigmoid) versus hard (clamp) bounds on the learned Langevin schedules."""

from __future__ import annotations

import math

import pytest
import torch

from sb_pomdp.belief import (
    _LOG_STEP_SIZE_MAX,
    _LOG_STEP_SIZE_MIN,
    _LOG_TEMPERATURE_MAX,
    _LOG_TEMPERATURE_MIN,
    EnergyBelief,
)
from sb_pomdp.config import ConfigError, load_config


def _belief(bound: str) -> EnergyBelief:
    torch.manual_seed(0)
    return EnergyBelief(
        3,
        2,
        2,
        [8],
        langevin_temperature=1.0,
        langevin_temperature_learnable=True,
        langevin_step_size_learnable=True,
        langevin_step_size=0.02,
        langevin_steps=4,
        langevin_schedule_bound=bound,
    )


def _context(batch: int = 2, num_particles: int = 4, steps: int = 4):
    torch.manual_seed(1)
    return (
        torch.randn(batch, 3),
        torch.randn(batch, num_particles, 2),
        torch.randn(batch, 2),
        torch.zeros(batch, dtype=torch.bool),
        torch.randn(batch, num_particles, 2),
        torch.randn(batch, steps, num_particles, 2),
    )


def _run(model: EnergyBelief, context, *, track_grad: bool = False):
    observation, previous, action, initial, initial_noise, noise = context
    return model.particles_from_noise(
        observation,
        previous,
        action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=noise,
        step_size=0.02,
        track_grad=track_grad,
    )


def test_sigmoid_initialisation_reproduces_the_configured_schedule() -> None:
    clamp = _belief("clamp")
    sigmoid = _belief("sigmoid")
    sigmoid.load_state_dict(
        {
            key: value
            for key, value in clamp.state_dict().items()
            if not key.startswith("langevin_log_")
        },
        strict=False,
    )
    assert torch.allclose(
        sigmoid.step_size_schedule(), torch.full((4,), 0.02), rtol=1e-5
    )
    assert torch.allclose(sigmoid.temperature_schedule(), torch.ones(4), rtol=1e-5)
    # The raw parameters differ (logit space) but the effective schedules match,
    # so both modes produce the same particles from the same noise.
    context = _context()
    particles_clamp, scores_clamp = _run(clamp, context)
    particles_sigmoid, scores_sigmoid = _run(sigmoid, context)
    assert torch.allclose(particles_clamp, particles_sigmoid, atol=1e-5)
    assert torch.allclose(scores_clamp, scores_sigmoid, atol=1e-5)


def test_sigmoid_schedule_stays_strictly_inside_the_bounds() -> None:
    model = _belief("sigmoid")
    with torch.no_grad():
        model.langevin_log_step_size.fill_(60.0)
        model.langevin_log_temperature.fill_(-60.0)
    alphas = model.step_size_schedule()
    taus = model.temperature_schedule()
    tolerance = 1e-5
    assert torch.all(alphas <= math.exp(_LOG_STEP_SIZE_MAX) * (1 + tolerance))
    assert torch.all(alphas >= math.exp(_LOG_STEP_SIZE_MIN) * (1 - tolerance))
    assert torch.all(taus >= math.exp(_LOG_TEMPERATURE_MIN) * (1 - tolerance))
    assert torch.all(taus <= math.exp(_LOG_TEMPERATURE_MAX) * (1 + tolerance))
    assert torch.allclose(
        alphas, torch.full((4,), math.exp(_LOG_STEP_SIZE_MAX)), rtol=tolerance
    )
    assert torch.allclose(
        taus, torch.full((4,), math.exp(_LOG_TEMPERATURE_MIN)), rtol=tolerance
    )


def test_clamp_kills_the_gradient_past_the_bound_but_sigmoid_keeps_it() -> None:
    context = _context()
    grads = {}
    for bound in ("clamp", "sigmoid"):
        model = _belief(bound)
        with torch.no_grad():
            # Push alpha past the 0.5 ceiling: log(0.75) for clamp, and a raw
            # logit that maps to alpha ~ 0.49 for sigmoid (inside but near it).
            if bound == "clamp":
                model.langevin_log_step_size.fill_(math.log(0.75))
            else:
                model.langevin_log_step_size.fill_(4.0)
        particles, _ = _run(model, context, track_grad=True)
        (grad,) = torch.autograd.grad(
            particles.square().sum(), model.langevin_log_step_size, allow_unused=True
        )
        grads[bound] = grad
    assert grads["clamp"] is None or torch.all(grads["clamp"] == 0.0)
    assert grads["sigmoid"] is not None
    assert torch.any(grads["sigmoid"] != 0.0)


def test_config_validates_the_schedule_bound() -> None:
    config = load_config("config/local.json")
    assert config.to_dict()["model"]["langevin_schedule_bound"] == "clamp"
    with pytest.raises(ConfigError):
        config.with_overrides({"model.langevin_schedule_bound": "tanh"})
    accepted = config.with_overrides({"model.langevin_schedule_bound": "sigmoid"})
    assert accepted["model"]["langevin_schedule_bound"] == "sigmoid"
