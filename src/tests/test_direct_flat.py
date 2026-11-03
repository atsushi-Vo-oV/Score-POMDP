"""score_parameterization = direct_flat: one MLP over (o, x, a, all previous particles in order)."""
from __future__ import annotations

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, load_config


def _belief(**kwargs) -> EnergyBelief:
    torch.manual_seed(3)
    return EnergyBelief(3, 2, 2, [16], score_parameterization="direct_flat", num_particles=4,
                        particle_sampler="svgd", **kwargs)


def _context(batch=2, k=4):
    g = torch.Generator().manual_seed(7)
    return (torch.randn(batch, 5, 2, generator=g), torch.randn(batch, 3, generator=g),
            torch.randn(batch, k, 2, generator=g), torch.randn(batch, 2, generator=g))


def test_needs_particle_count_and_has_no_transition_network():
    with pytest.raises(ValueError):
        EnergyBelief(3, 2, 2, [8], score_parameterization="direct_flat")
    belief = _belief()
    assert belief.transition_energy is None
    first = next(m for m in belief.observation_energy.modules() if isinstance(m, torch.nn.Linear))
    assert first.in_features == 3 + 2 + 2 + 4 * 2


def test_score_is_not_permutation_invariant_and_reads_every_previous_particle():
    belief = _belief()
    query, obs, previous, action = _context()
    initial = torch.zeros(2, dtype=torch.bool)
    base = belief.score_at(query, obs, previous, action, initial, create_graph=False)
    assert base.shape == (2, 5, 2)
    permuted = belief.score_at(query, obs, previous[:, [1, 0, 3, 2]], action, initial, create_graph=False)
    assert not torch.allclose(base, permuted)
    for j in range(4):
        moved = previous.clone()
        moved[:, j] += 1.0
        assert not torch.allclose(base, belief.score_at(query, obs, moved, action, initial, create_graph=False)), j
    with pytest.raises(ValueError):
        belief.score_at(query, obs, previous[:, :3], action, initial, create_graph=False)


def test_initial_step_ignores_previous_particles():
    belief = _belief()
    query, obs, previous, action = _context()
    initial = torch.ones(2, dtype=torch.bool)
    a = belief.score_at(query, obs, previous, action, initial, create_graph=False)
    b = belief.score_at(query, obs, previous + 5.0, action, initial, create_graph=False)
    assert torch.allclose(a, b)


def test_transport_runs_with_gradients():
    belief = _belief()
    query, obs, previous, action = _context()
    particles, scores = belief.particles_from_noise(
        obs, previous, action, torch.zeros(2, dtype=torch.bool),
        initial_noise=query[:, :4], langevin_noise=torch.randn(2, 3, 4, 2), step_size=0.1, track_grad=True,
    )
    (particles.square().sum() + scores.square().sum()).backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in belief.observation_energy.parameters())


def test_config_accepts_direct_flat():
    base = load_config("config/trysvgdlight_mirror2_tf-svgd.json")
    base.with_overrides({"model.score_parameterization": "direct_flat"})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.score_parameterization": "direct_flattened"})
