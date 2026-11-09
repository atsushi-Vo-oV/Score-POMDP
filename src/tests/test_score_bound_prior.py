"""model.direct_score_bound = null (no tanh bound) and model.initial_score_prior (weight of the -x term at t = 0)."""
from __future__ import annotations

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, ExperimentConfig, load_config


def _belief(**kwargs) -> EnergyBelief:
    torch.manual_seed(3)
    defaults = dict(
        observation_dim=3,
        state_dim=2,
        action_feature_dim=2,
        hidden_dims=[8],
        score_clip=0.0,
        particle_clip=0.0,
        langevin_step_size_learnable=True,
        langevin_step_size=0.2,
        langevin_steps=3,
        observation_anchor=False,
        particle_sampler="svgd",
        score_parameterization="direct",
    )
    defaults.update(kwargs)
    return EnergyBelief(**defaults)


def _inputs(batch: int = 4, scale: float = 1.0):
    g = torch.Generator().manual_seed(7)
    return dict(
        query=torch.randn(batch, 5, 2, generator=g) * scale,
        observation=torch.randn(batch, 3, generator=g),
        previous_particles=torch.randn(batch, 5, 2, generator=g),
        previous_action=torch.randn(batch, 2, generator=g),
    )


def _score(belief: EnergyBelief, branch: str, **inputs) -> torch.Tensor:
    initial = torch.full((inputs["query"].shape[0],), branch == "initial", dtype=torch.bool)
    return belief.score_at(
        inputs["query"],
        inputs["observation"],
        inputs["previous_particles"],
        inputs["previous_action"],
        initial,
        create_graph=False,
    )


def test_log_squashing_is_identity_near_zero_and_sublinear_far_away():
    import math

    squashed = _belief(direct_score_bound="log")
    raw = _belief(direct_score_bound=None)
    raw.load_state_dict(squashed.state_dict())
    with torch.no_grad():
        for layer in raw.initial_energy.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.mul_(40.0)
        squashed.load_state_dict(raw.state_dict())
    inputs = _inputs(scale=3.0)
    s_raw = _score(raw, "initial", **inputs) + inputs["query"]
    s_log = _score(squashed, "initial", **inputs) + inputs["query"]
    expected = s_raw / torch.log(math.e + s_raw.square())
    assert torch.allclose(s_log, expected, atol=1e-5)
    small = torch.tensor([1e-3, -1e-3])
    assert torch.allclose(small / torch.log(math.e + small.square()), small, rtol=1e-3)
    big = torch.tensor([1e4])
    assert (big / torch.log(math.e + big.square())).item() < 600.0  # unbounded but slow


def test_unbounded_direct_score_returns_the_raw_network_output():
    bounded, unbounded = _belief(direct_score_bound=10.0), _belief(direct_score_bound=None)
    unbounded.load_state_dict(bounded.state_dict())
    with torch.no_grad():
        for layer in unbounded.initial_energy.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.mul_(40.0)
        bounded.load_state_dict(unbounded.state_dict())
    inputs = _inputs(scale=3.0)
    s_b = _score(bounded, "initial", **inputs)
    s_u = _score(unbounded, "initial", **inputs)
    raw = unbounded.initial_energy(
        torch.cat((inputs["observation"][:, None, :].expand(-1, 5, -1), inputs["query"]), dim=-1)
    )
    assert torch.allclose(s_u, raw - inputs["query"], atol=1e-5)
    assert (s_b + inputs["query"]).abs().max() <= 10.0 + 1e-5
    assert s_u.abs().max() > s_b.abs().max()


def test_initial_score_prior_zero_removes_the_restoring_term():
    with_prior, without = _belief(), _belief(initial_score_prior=0.0)
    without.load_state_dict(with_prior.state_dict())
    inputs = _inputs()
    s1 = _score(with_prior, "initial", **inputs)
    s0 = _score(without, "initial", **inputs)
    assert torch.allclose(s1 - s0, -inputs["query"], atol=1e-6)
    # the recursive branch does not carry the term in either model
    assert torch.allclose(_score(with_prior, "recursive", **inputs), _score(without, "recursive", **inputs))


def test_initial_score_prior_scales_the_energy_potential():
    energy1 = _belief(score_parameterization="energy", initial_score_prior=1.0)
    energy0 = _belief(score_parameterization="energy", initial_score_prior=0.0)
    energy0.load_state_dict(energy1.state_dict())
    inputs = _inputs()
    s1 = _score(energy1, "initial", **inputs)
    s0 = _score(energy0, "initial", **inputs)
    assert torch.allclose(s1 - s0, -inputs["query"], atol=1e-5)


def test_validation():
    with pytest.raises(ValueError):
        _belief(direct_score_bound=0.0)
    with pytest.raises(ValueError):
        _belief(direct_score_bound="exp")
    with pytest.raises(ValueError):
        _belief(initial_score_prior=-0.5)
    with pytest.raises(ConfigError):
        load_config("config/trysvgdlight_mirror2_tf-svgd.json").with_overrides({"model.direct_score_bound": "exp"})
    assert isinstance(
        load_config("config/trysvgdlight_mirror2_tf-svgd.json").with_overrides({"model.direct_score_bound": "log"}),
        ExperimentConfig,
    )
    base = load_config("config/trysvgdlight_mirror2_tf-svgd.json")
    model = base.to_dict()["model"]
    assert model["direct_score_bound"] == 10.0 and model["initial_score_prior"] == 1.0
    assert isinstance(
        base.with_overrides({"model.direct_score_bound": None, "model.initial_score_prior": 0.0}), ExperimentConfig
    )
    with pytest.raises(ConfigError):
        base.with_overrides({"model.initial_score_prior": -1.0})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.direct_score_bound": -3.0})
