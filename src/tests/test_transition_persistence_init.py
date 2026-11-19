"""model.transition_persistence_init: the transition network starts as a pull to the previous cloud (experiment 30)."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from sb_pomdp.belief import _SILU_INFLECTION, EnergyBelief, persistence_initialise_
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.networks import make_mlp


def _belief(state_dim: int = 2, hidden: int = 8, **kwargs) -> EnergyBelief:
    torch.manual_seed(5)
    return EnergyBelief(
        observation_dim=3, state_dim=state_dim, action_feature_dim=2, hidden_dims=[hidden], score_clip=0.0,
        particle_clip=0.0, langevin_step_size_learnable=True, langevin_step_size=0.2, langevin_steps=3,
        langevin_proposal="uniform", langevin_initial_noise="uniform", particle_sampler="svgd",
        score_parameterization="direct", direct_score_bound=None, initial_score_prior=0.0, **kwargs,
    )


def _pairs(state_dim: int, count: int = 512, scale: float = 1.5):
    g = torch.Generator().manual_seed(3)
    query = scale * torch.randn(count, state_dim, generator=g)
    previous = scale * torch.randn(count, state_dim, generator=g)
    action = torch.randn(count, 2, generator=g)
    return query, previous, action


def test_default_is_the_usual_initialisation_bit_for_bit():
    a, b = _belief(), _belief(transition_persistence_init=None)
    assert a.transition_persistence_init is None
    state_a, state_b = a.state_dict(), b.state_dict()
    assert state_a.keys() == state_b.keys() and all(torch.equal(state_a[k], state_b[k]) for k in state_a)


@pytest.mark.parametrize("units", ["pair", "single"])
def test_only_the_transition_network_changes_and_no_random_numbers_are_drawn(units):
    plain = _belief()
    after_plain = torch.rand(1)
    started = _belief(transition_persistence_init=4.0, transition_persistence_units=units)
    after_started = torch.rand(1)
    assert torch.equal(after_plain, after_started)
    state_plain, state_started = plain.state_dict(), started.state_dict()
    assert state_plain.keys() == state_started.keys()
    changed = {k for k in state_plain if not torch.equal(state_plain[k], state_started[k])}
    assert changed and all(k.startswith("transition_energy.") for k in changed)
    assert all(p.requires_grad for p in started.transition_energy.parameters())


@pytest.mark.parametrize("strength", [4.0, 11.0])
def test_pair_units_give_the_exact_field_with_uniform_attention(strength):
    belief = _belief(transition_persistence_init=strength)
    query, previous, action = _pairs(2)
    out = belief.transition_energy(torch.cat((query, previous, action), dim=-1))
    assert torch.equal(out[:, 0], torch.zeros_like(out[:, 0]))
    assert torch.allclose(out[:, 1:], strength * (previous - query), atol=1e-5)
    # the action does not enter at the start
    other = belief.transition_energy(torch.cat((query, previous, -action), dim=-1))
    assert torch.equal(out, other)


def test_single_units_are_close_to_the_field_and_fit_six_dimensions_in_eight_hidden_units():
    sigmoid = 1.0 / (1.0 + math.exp(-_SILU_INFLECTION))
    assert abs(2.0 + _SILU_INFLECTION * (1.0 - 2.0 * sigmoid)) < 1e-9  # second derivative of SiLU is 0 there
    for state_dim in (2, 6):
        belief = _belief(state_dim, transition_persistence_init=4.0, transition_persistence_units="single")
        query, previous, action = _pairs(state_dim, scale=0.5)
        out = belief.transition_energy(torch.cat((query, previous, action), dim=-1))
        target = 4.0 * (previous - query)
        assert torch.equal(out[:, 0], torch.zeros_like(out[:, 0]))
        relative = (out[:, 1:] - target).square().mean().sqrt() / target.square().mean().sqrt()
        assert float(relative) < 0.08
        difference = previous - query
        small = difference.abs() < 0.5
        assert float((out[:, 1:] - target)[small].abs().max()) < 0.06
        assert bool((out[:, 1:] * difference)[difference.abs() > 0.05].gt(0).all())  # always towards the previous particle
    with pytest.raises(ValueError):
        _belief(6, transition_persistence_init=4.0)  # pair needs 12 hidden units


@pytest.mark.parametrize("units", ["pair", "single"])
def test_the_recursive_score_starts_as_the_pull_to_the_previous_mean(units):
    belief = _belief(transition_persistence_init=4.0, transition_persistence_units=units)
    g = torch.Generator().manual_seed(9)
    batch, particles = 3, 5
    query = 0.5 * torch.randn(batch, particles, 2, generator=g)
    previous = 0.5 * torch.randn(batch, particles, 2, generator=g)
    observation = torch.randn(batch, 3, generator=g)
    action = torch.randn(batch, 2, generator=g)
    initial = torch.zeros(batch, dtype=torch.bool)
    score = belief._direct_score_for_branch(query, observation, previous, action, initial, "recursive")
    field = belief.observation_energy(
        torch.cat((observation[:, None, :].expand(-1, particles, -1), query, action[:, None, :].expand(-1, particles, -1)), dim=-1)
    )
    pull = 4.0 * (previous.mean(dim=1, keepdim=True) - query)
    assert torch.allclose(score - field, pull, atol=1e-5 if units == "pair" else 0.3)
    # the initial branch is untouched
    plain = _belief()
    first = torch.ones(batch, dtype=torch.bool)
    assert torch.equal(
        belief._direct_score_for_branch(query, observation, previous, action, first, "initial"),
        plain._direct_score_for_branch(query, observation, previous, action, first, "initial"),
    )


def test_kernel_none_is_the_uniform_attention_start_bit_for_bit():
    a, b = _belief(transition_persistence_init=4.0), _belief(transition_persistence_init=4.0, transition_persistence_kernel=None)
    assert a.transition_persistence_kernel is None
    state_a, state_b = a.state_dict(), b.state_dict()
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)


@pytest.mark.parametrize("sharpness", [4.0, 1.0])
def test_kernel_start_gives_a_per_particle_attention_from_the_same_units(sharpness):
    uniform = _belief(transition_persistence_init=4.0)
    after_uniform = torch.rand(1)
    mixture = _belief(transition_persistence_init=4.0, transition_persistence_kernel=sharpness)
    after_mixture = torch.rand(1)
    assert torch.equal(after_uniform, after_mixture)  # no random numbers are drawn
    state_u, state_m = uniform.state_dict(), mixture.state_dict()
    changed = {k for k in state_u if not torch.equal(state_u[k], state_m[k])}
    assert changed == {"transition_energy.2.weight"}  # only the outgoing weights of the logit differ
    assert torch.equal(state_u["transition_energy.2.weight"][1:], state_m["transition_energy.2.weight"][1:])
    query, previous, action = _pairs(2)
    out = mixture.transition_energy(torch.cat((query, previous, action), dim=-1))
    z = previous - query
    assert torch.allclose(out[:, 1:], 4.0 * z, atol=1e-5)  # the values are the same exact field
    assert torch.allclose(out[:, 0], -sharpness * (z * torch.tanh(z / 2)).sum(dim=-1), atol=1e-4)
    assert torch.equal(out, mixture.transition_energy(torch.cat((query, previous, -action), dim=-1)))
    # near a previous particle the logit is the normal kernel of variance 1 / sharpness
    near_query = previous + 0.2 * torch.randn(previous.shape, generator=torch.Generator().manual_seed(4))
    near = mixture.transition_energy(torch.cat((near_query, previous, action), dim=-1))[:, 0]
    exact = -sharpness * (previous - near_query).square().sum(dim=-1) / 2
    assert float((near - exact).abs().max()) < 0.02 * sharpness


def test_kernel_start_pulls_to_the_near_group_and_the_uniform_start_to_the_midpoint():
    g = torch.Generator().manual_seed(9)
    previous = 0.1 * torch.randn(1, 8, 2, generator=g)
    previous[:, :4, 0] += 2.0
    previous[:, 4:, 0] -= 2.0
    query = torch.tensor([[[1.5, 0.0], [-1.5, 0.0]]])
    observation, action = torch.randn(1, 3, generator=g), torch.randn(1, 2, generator=g)
    initial = torch.zeros(1, dtype=torch.bool)

    def transition_term(belief):
        score = belief._direct_score_for_branch(query, observation, previous, action, initial, "recursive")
        field = belief.observation_energy(
            torch.cat((observation[:, None, :].expand(-1, 2, -1), query, action[:, None, :].expand(-1, 2, -1)), dim=-1)
        )
        return score - field

    mixture = transition_term(_belief(transition_persistence_init=4.0, transition_persistence_kernel=4.0))
    z = previous[:, None, :, :] - query[:, :, None, :]  # (1, query, previous, 2)
    weights = torch.softmax(-4.0 * (z * torch.tanh(z / 2)).sum(dim=-1), dim=-1)
    assert torch.allclose(mixture, (weights[..., None] * 4.0 * z).sum(dim=2), atol=1e-4)  # the attention-weighted pull
    assert float(weights[0, 0, 4:].sum()) < 1e-4 and float(weights[0, 1, :4].sum()) < 1e-4  # the far group has no weight
    assert float(mixture[0, 0, 0]) > 1.0 and float(mixture[0, 1, 0]) < -1.0  # each query moves to its own group
    uniform = transition_term(_belief(transition_persistence_init=4.0))
    assert torch.allclose(uniform, 4.0 * (previous.mean(dim=1, keepdim=True) - query), atol=1e-5)


def test_kernel_start_invalid_combinations():
    with pytest.raises(ValueError):
        _belief(transition_persistence_kernel=4.0)  # needs the initialisation
    with pytest.raises(ValueError):
        _belief(transition_persistence_init=4.0, transition_persistence_kernel=0.0)
    with pytest.raises(ValueError):
        _belief(6, transition_persistence_init=4.0, transition_persistence_units="single", transition_persistence_kernel=4.0)
    base = load_config("config/trysvgdlight2_mirror2_tf-svgd.json")
    assert base.to_dict()["model"]["transition_persistence_kernel"] is None
    with pytest.raises(ConfigError):
        base.with_overrides({"model.transition_persistence_kernel": 4.0})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.transition_persistence_init": 4.0, "model.transition_persistence_kernel": -1.0})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.transition_persistence_init": 4.0, "model.transition_persistence_units": "single",
                             "model.transition_persistence_kernel": 4.0})
    started = base.with_overrides({"model.transition_persistence_init": 4.0, "model.transition_persistence_kernel": 4.0})
    assert started.to_dict()["model"]["transition_persistence_kernel"] == 4.0


def test_initial_projection_start_gives_a_narrow_cloud_around_an_image_of_the_observation():
    plain = _belief()
    after_plain = torch.rand(1)
    started = _belief(initial_score_projection_precision=4.0)
    after_started = torch.rand(1)
    assert torch.equal(after_plain, after_started)  # no random numbers are drawn
    state_plain, state_started = plain.state_dict(), started.state_dict()
    changed = {k for k in state_plain if not torch.equal(state_plain[k], state_started[k])}
    assert changed and all(k.startswith("initial_energy.") for k in changed)
    unset = _belief(initial_score_projection_precision=None).state_dict()
    assert all(torch.equal(state_plain[k], unset[k]) for k in state_plain)
    g = torch.Generator().manual_seed(2)
    observation, query = torch.randn(64, 3, generator=g), torch.randn(64, 2, generator=g)
    projection = state_plain["initial_energy.0.weight"][:2, :3]  # the image uses the weights that were already there
    out = started.initial_energy(torch.cat((observation, query), dim=-1))
    assert torch.allclose(out, 4.0 * (observation @ projection.T - query), atol=1e-5)
    # together with the transition start; the recursive networks are untouched by it
    both = _belief(transition_persistence_init=16.0, transition_persistence_kernel=4.0, initial_score_projection_precision=4.0)
    reference = _belief(transition_persistence_init=16.0, transition_persistence_kernel=4.0).state_dict()
    assert all(torch.equal(both.state_dict()[k], reference[k]) for k in reference if not k.startswith("initial_energy."))
    with pytest.raises(ValueError):
        _belief(initial_score_projection_precision=0.0)
    with pytest.raises(ValueError):
        _belief(2, 3, initial_score_projection_precision=4.0)  # 3 hidden units < 4
    with pytest.raises(ValueError):  # the v1 default has a bound and a prior at t = 0
        EnergyBelief(observation_dim=3, state_dim=2, action_feature_dim=2, hidden_dims=[8], score_clip=0.0, particle_clip=0.0,
                     score_parameterization="direct", initial_score_projection_precision=4.0)
    base = load_config("config/trysvgdlight2_mirror2_tf-svgd.json")
    assert base.to_dict()["model"]["initial_score_projection_precision"] is None
    started_config = base.with_overrides({"model.initial_score_projection_precision": 4.0})
    assert started_config.to_dict()["model"]["initial_score_projection_precision"] == 4.0
    with pytest.raises(ConfigError):
        base.with_overrides({"model.initial_score_projection_precision": -1.0})
    with pytest.raises(ConfigError):
        load_config("config/trysvgdlight_mirror2_tf-svgd.json").with_overrides({"model.initial_score_projection_precision": 4.0})


def test_the_started_weights_train():
    belief = _belief(transition_persistence_init=4.0)
    before = {k: v.clone() for k, v in belief.transition_energy.state_dict().items()}
    query, previous, action = _pairs(2, count=64)
    optimizer = torch.optim.Adam(belief.transition_energy.parameters(), lr=1e-2)
    out = belief.transition_energy(torch.cat((query, previous, action), dim=-1))
    (out[:, 0] * action[:, 0]).mean().add(out[:, 1:].square().mean()).backward()
    optimizer.step()
    after = belief.transition_energy.state_dict()
    assert all(not torch.equal(before[k], after[k]) for k in before)


def test_invalid_combinations():
    with pytest.raises(ValueError):
        _belief(transition_persistence_init=-1.0)
    with pytest.raises(ValueError):
        _belief(transition_persistence_init=4.0, transition_persistence_units="triple")
    with pytest.raises(ValueError):
        _belief(2, 3, transition_persistence_init=4.0)  # 3 hidden units < 4
    with pytest.raises(ValueError):
        EnergyBelief(observation_dim=3, state_dim=2, action_feature_dim=2, hidden_dims=[8], score_clip=0.0, particle_clip=0.0,
                     transition_persistence_init=4.0)  # scalar-energy parameterisation
    with pytest.raises(ValueError):
        persistence_initialise_(make_mlp(6, [8, 8], 3), 2, 4.0)  # two hidden layers
    base = load_config("config/trysvgdlight2_mirror2_tf-svgd.json")
    model = base.to_dict()["model"]
    assert model["transition_persistence_init"] is None and model["transition_persistence_units"] == "pair"
    with pytest.raises(ConfigError):
        base.with_overrides({"model.transition_persistence_init": 0.0})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.transition_persistence_units": "triple"})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.transition_persistence_init": 4.0, "model.energy_hidden": [8, 8]})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.transition_persistence_init": 4.0, "model.energy_network_kind": "kan"})


def _generator():
    spec = importlib.util.spec_from_file_location("tinit_generate", Path("scripts/tinit_generate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_arms_differ_from_their_default_only_in_the_listed_keys():
    assets = list(_generator().assets())
    assert len(assets) == 32  # 7 tasks x 2 defaults; the mixture form (x 2 defaults) and the narrow first cloud (v2) on 6 tasks
    assert not any("tinitq" in name and ("_tagr_" in name or name.startswith("trysvgdlight_")) for name, *_ in assets)
    assert not any("tinitmix" in name and "_tagr_" in name for name, *_ in assets)
    for name, source, label, campaign, stem, keys, updates, _ in assets:
        arm, base = load_config(f"config/{name}.json").to_dict(), load_config(f"config/{source}.json").to_dict()
        assert arm["model"]["transition_persistence_init"] == 4.0
        assert arm["model"]["transition_persistence_units"] == ("single" if "_tagr_" in name else "pair")
        assert arm["model"]["transition_persistence_kernel"] == (4.0 if name.endswith("-tinitmix") else None)
        assert arm["model"]["initial_score_projection_precision"] == (4.0 if name.endswith("-tinitq") else None)
        assert arm["experiment"]["label"] == label and campaign == f"{label}-v1"
        for key in keys:
            arm["model"][key] = base["model"][key]
        arm["experiment"]["label"] = base["experiment"]["label"]
        assert arm == base
        for seed in range(10, 15):
            text = Path(f"jobs/tinit/{stem}_s{seed}.sh").read_text(encoding="utf-8")
            assert f'SB_POMDP_CONFIG="config/{name}.json"' in text
            assert f'SB_POMDP_CAMPAIGN_ID="{campaign}"' in text
            assert f"SB_POMDP_SHARD_INDEX={seed - 9}" in text and f"SB_POMDP_SEGMENT_UPDATES={updates}" in text
            assert text.rstrip().endswith("exec bash jobs/genkai_production.sh")


def test_the_model_of_every_arm_builds_with_the_started_transition_network():
    from sb_pomdp.envs import make_env  # noqa: PLC0415
    from sb_pomdp.train import model_for_environment  # noqa: PLC0415

    for name, *_ in _generator().assets():
        config = load_config(f"config/{name}.json").to_dict()
        task = config["experiment"]["tasks"][0]
        environment = make_env(task, config["environment"]["tasks"][task], seed=0)
        model = model_for_environment(environment, config["model"], torch.device("cpu"))
        d = model.belief.state_dim
        g = torch.Generator().manual_seed(1)
        query, previous = 0.5 * torch.randn(64, d, generator=g), 0.5 * torch.randn(64, d, generator=g)
        action = torch.zeros(64, model.belief.action_feature_dim)
        out = model.belief.transition_energy(torch.cat((query, previous, action), dim=-1))
        target = 4.0 * (previous - query)
        assert float((out[:, 1:] - target).square().mean().sqrt() / target.square().mean().sqrt()) < 0.08
