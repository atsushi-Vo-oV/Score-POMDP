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
    log_scales = torch.randn(1, 4, 2, dtype=torch.float64) * 0.3 - 1.0
    eps = torch.randn(1, 4, 2, dtype=torch.float64)
    grad_theta, grad_log_scale = _mixture_entropy_gradient(means, log_scales, eps, create_graph=False)
    theta = means.clone().requires_grad_(True)
    log_s = log_scales.clone().requires_grad_(True)
    draws = theta + log_s.exp() * eps
    z = (draws[:, :, None] - theta[:, None]) / log_s.exp()[:, None]
    log_q = torch.logsumexp(-0.5 * z.square().sum(-1) - log_s.sum(-1)[:, None, :], dim=-1)
    expected_theta, expected_log_s = torch.autograd.grad(log_q.sum(), (theta, log_s))
    assert torch.allclose(grad_theta, -expected_theta) and torch.allclose(grad_log_scale, -expected_log_s)


def test_smi_returns_the_component_log_scales_next_to_the_scores() -> None:
    belief = _belief("smi", smi_component_scale=0.1)
    assert belief.score_feature_dim == 4 and _belief("svgd").score_feature_dim == 2
    particles, features = _run(belief, track_grad=False)
    assert particles.shape == (3, 6, 2) and features.shape == (3, 6, 4)
    log_scales = features[..., 2:]
    assert not torch.allclose(log_scales, torch.full_like(log_scales, math.log(0.1)))  # transported
    assert torch.all((log_scales >= math.log(1e-3) - 1e-6) & (log_scales <= math.log(2.0) + 1e-6))


def test_the_policy_encoder_reads_the_smi_log_scales() -> None:
    from sb_pomdp.envs import make_env
    from sb_pomdp.train import model_for_environment

    resolved = load_config("config/core4_tps.json").to_dict()
    model_config = dict(resolved["model"], particle_sampler="smi")
    env = make_env("mountain_hike", resolved["environment"]["tasks"]["mountain_hike"], seed=0)
    model = model_for_environment(env, model_config, torch.device("cpu"))
    assert model.encoder.score_dim == 2 * model.belief.state_dim
    particles = torch.zeros(1, 16, model.belief.state_dim)
    small = model.encode(particles, torch.cat((torch.zeros_like(particles), torch.full_like(particles, -3.0)), -1))
    large = model.encode(particles, torch.cat((torch.zeros_like(particles), torch.full_like(particles, 0.5)), -1))
    assert not torch.allclose(small, large)


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
    no_grad, no_grad_scores = _run(belief, track_grad=False, initial=initial)
    assert torch.allclose(particles.detach(), no_grad, atol=1e-6)
    assert torch.allclose(scores.detach(), no_grad_scores, atol=1e-5)


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
