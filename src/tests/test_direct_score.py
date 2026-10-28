"""Directly parameterised score field (model.score_parameterization="direct", try-svgd-light)."""

from __future__ import annotations

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, load_config, validate_config


def _belief(**kwargs) -> EnergyBelief:
    torch.manual_seed(0)
    return EnergyBelief(3, 2, 2, [8], score_parameterization="direct", **kwargs)


def _context(batch: int = 2, k: int = 5):
    torch.manual_seed(1)
    return torch.randn(batch, k, 2), torch.randn(batch, 3), torch.randn(batch, 4, 2), torch.randn(batch, 2)


def test_direct_networks_output_score_vectors() -> None:
    belief = _belief()
    assert belief.initial_energy(torch.zeros(1, 5)).shape == (1, 2)
    assert belief.observation_energy(torch.zeros(1, 7)).shape == (1, 2)
    assert belief.transition_energy(torch.zeros(1, 6)).shape == (1, 3)


def test_direct_score_matches_its_formula() -> None:
    belief = _belief()
    query, observation, previous, action = _context()
    initial = belief.score_at(query, observation, previous, action, torch.ones(2, dtype=torch.bool), create_graph=False)
    bound = belief.direct_score_bound
    expected = bound * torch.tanh(belief.initial_energy(torch.cat((observation[:, None].expand(-1, 5, -1), query), -1)) / bound) - query
    assert torch.allclose(initial, expected, atol=1e-6)
    recursive = belief.score_at(query, observation, previous, action, torch.zeros(2, dtype=torch.bool), create_graph=False)
    b, i = 1, 3
    field = bound * torch.tanh(belief.observation_energy(torch.cat((observation[b], query[b, i], action[b]))) / bound)
    out = belief.transition_energy(torch.cat((query[b, i].expand(4, -1), previous[b], action[b].expand(4, -1)), -1))
    weights = torch.softmax(out[:, 0], 0)
    assert torch.allclose(recursive[b, i], field + (weights[:, None] * bound * torch.tanh(out[:, 1:] / bound)).sum(0), atol=1e-6)
    mixed = belief.score_at(query, observation, previous, action, torch.tensor([True, False]), create_graph=False)
    assert torch.allclose(mixed[0], initial[0], atol=1e-6) and torch.allclose(mixed[1], recursive[1], atol=1e-6)


def test_direct_score_has_no_scalar_potential() -> None:
    belief = _belief()
    query, observation, previous, action = _context()
    with pytest.raises(ValueError):
        belief.log_potential(query, observation, previous, action, torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
def test_direct_score_trains_through_every_sampler(sampler: str) -> None:
    belief = _belief(particle_sampler=sampler, langevin_noise_scale=0.2 if sampler == "ula" else None)
    _, observation, previous, action = _context(k=4)
    for flag in (True, False):
        belief.zero_grad()
        initial_noise, step_noise = belief.draw_particle_noise(observation, num_particles=4, num_steps=3)
        particles, scores = belief.particles_from_noise(
            observation, previous, action, torch.full((2,), flag), initial_noise=initial_noise,
            langevin_noise=step_noise, step_size=0.2, track_grad=True)
        (particles.square().sum() + scores.square().sum()).backward()
        networks = [belief.initial_energy] if flag else [belief.observation_energy, belief.transition_energy]
        for network in networks:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters())


def test_energy_parameterisation_is_the_default_and_unchanged() -> None:
    torch.manual_seed(0)
    default = EnergyBelief(3, 2, 2, [8])
    assert default.score_parameterization == "energy" and default.transition_energy(torch.zeros(1, 6)).shape == (1, 1)
    resolved = load_config("config/trysvgd_tps_tf-ula.json").to_dict()
    resolved["model"]["score_parameterization"] = "gradient"
    with pytest.raises(ConfigError):
        validate_config(resolved)


def test_direct_score_stays_bounded_far_from_the_origin() -> None:
    belief = _belief(direct_score_bound=10.0)
    query, observation, previous, action = _context()
    far = 1e4 * query
    recursive = belief.score_at(far, observation, previous, action, torch.zeros(2, dtype=torch.bool), create_graph=False)
    assert recursive.abs().max() <= 2 * 10.0 + 1e-4  # field + attention average, each bounded by B
