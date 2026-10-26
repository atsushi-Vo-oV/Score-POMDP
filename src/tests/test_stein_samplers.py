"""SVGD and SMI particle transports as drop-in replacements for the ULA chain of the score belief."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from sb_pomdp.belief import EnergyBelief, _mixture_entropy_gradient, _rbf_stein_direction
from sb_pomdp.config import ConfigError, load_config, validate_config


def _belief(sampler: str, **kwargs: Any) -> EnergyBelief:
    torch.manual_seed(0)
    return EnergyBelief(2, 2, 2, [8], langevin_warm_start=True, particle_sampler=sampler, **kwargs)


def _run(belief: EnergyBelief, *, track_grad: bool, noise_seed: int = 2, initial: bool = False):
    torch.manual_seed(1)
    observation = torch.randn(3, 2)
    previous = torch.randn(3, 6, 2)
    action = torch.zeros(3, 2)
    flags = torch.full((3,), initial, dtype=torch.bool)
    initial_noise, step_noise = belief.draw_particle_noise(
        observation, num_particles=6, num_steps=4, generator=torch.Generator().manual_seed(noise_seed)
    )
    return belief.particles_from_noise(observation, previous, action, flags, initial_noise=initial_noise,
                                       langevin_noise=step_noise, step_size=0.2, track_grad=track_grad)


def test_svgd_direction_matches_the_kernel_formula() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 2)
    drive = torch.randn(2, 5, 2)
    phi = _rbf_stein_direction(x, drive)
    for b in range(2):
        d2 = torch.cdist(x[b], x[b]).square()
        h = d2[~torch.eye(5, dtype=torch.bool)].median() / math.log(6.0)
        k = torch.exp(-d2 / h)
        expected = (k @ drive[b] + (2.0 / h) * (k[..., None] * (x[b][:, None] - x[b][None])).sum(1)) / 5
        assert torch.allclose(phi[b], expected, atol=1e-6)


def test_svgd_repulsion_spreads_particles_without_noise() -> None:
    x = torch.tensor([[[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]]])
    phi = _rbf_stein_direction(x, torch.zeros_like(x))
    spread_before = x.var(dim=1).sum()
    spread_after = (x + 0.1 * phi).var(dim=1).sum()
    assert spread_after > spread_before


def test_mixture_entropy_gradient_matches_autograd_of_the_mixture_log_density() -> None:
    torch.manual_seed(0)
    means = torch.randn(1, 4, 2, dtype=torch.float64)
    eps = torch.randn(1, 4, 2, dtype=torch.float64)
    scale = 0.3
    gradient = _mixture_entropy_gradient(means, means + scale * eps, scale, create_graph=False)
    theta = means.clone().requires_grad_(True)
    draws = theta + scale * eps
    log_q = torch.logsumexp(-0.5 * (draws[:, :, None] - theta[:, None]).square().sum(-1) / scale**2, dim=-1)
    (expected,) = torch.autograd.grad(log_q.sum(), theta)
    assert torch.allclose(gradient, -expected)


@pytest.mark.parametrize("sampler", ["svgd", "smi"])
def test_stein_samplers_replay_exactly_and_differ_from_ula(sampler: str) -> None:
    stein = _belief(sampler)
    ula = _belief("ula")
    ula.load_state_dict(stein.state_dict())
    first, _ = _run(stein, track_grad=False)
    again, _ = _run(stein, track_grad=False)
    assert torch.equal(first, again)
    assert not torch.allclose(first, _run(ula, track_grad=False)[0])
    other_noise, _ = _run(stein, track_grad=False, noise_seed=3)
    if sampler == "svgd":  # deterministic transport: only the (cold) chain start uses noise
        assert torch.equal(first, other_noise)
    else:  # SMI draws its component samples from the saved step noise
        assert not torch.equal(first, other_noise)


@pytest.mark.parametrize("sampler", ["svgd", "smi"])
@pytest.mark.parametrize("initial", [False, True])
def test_stein_samplers_carry_gradients_to_the_energies_and_the_step_size(sampler: str, initial: bool) -> None:
    belief = _belief(sampler, langevin_step_size_learnable=True, langevin_step_size=0.2, langevin_steps=4)
    particles, scores = _run(belief, track_grad=True, initial=initial)
    (particles.square().sum() + scores.square().sum()).backward()
    energy = belief.initial_energy if initial else belief.observation_energy
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in energy.parameters())
    assert belief.langevin_log_step_size.grad is not None and belief.langevin_log_step_size.grad.abs().sum() > 0
    no_grad, _ = _run(belief, track_grad=False, initial=initial)
    assert torch.allclose(particles.detach(), no_grad, atol=1e-6)


def test_stein_samplers_reject_ula_noise_settings() -> None:
    with pytest.raises(ValueError):
        _belief("svgd", langevin_noise_scale=0.2)
    with pytest.raises(ValueError):
        _belief("nuts")
    resolved = load_config("config/core6_tps_b02.json").to_dict()
    resolved["model"]["particle_sampler"] = "smi"
    with pytest.raises(ConfigError):
        validate_config(resolved)  # core6 sets a decoupled noise scale
    resolved["model"]["langevin_noise_scale"] = None
    validate_config(resolved)
    resolved["model"]["smi_component_scale"] = 0.0
    with pytest.raises(ConfigError):
        validate_config(resolved)
