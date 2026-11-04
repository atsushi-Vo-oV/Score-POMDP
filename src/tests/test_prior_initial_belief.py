"""model.initial_belief = "prior": t = 0 is a recursive step over K learned prior particles.

The initial network and the observation anchor disappear; an initial row is scored by the
recursive networks with the prior particles as its previous belief and a zero previous action,
and its chain starts where a recursive chain starts (noise, or the prior particles themselves
under a warm start).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, ExperimentConfig, load_config
from sb_pomdp.run_experiments import run_all

K = 4


def _belief(
    initial_belief: str = "prior",
    parameterization: str = "direct",
    sampler: str = "svgd",
    **kwargs,
) -> EnergyBelief:
    torch.manual_seed(5)
    return EnergyBelief(
        observation_dim=3,
        state_dim=2,
        action_feature_dim=2,
        hidden_dims=[8],
        score_clip=0.0,
        particle_clip=0.0,
        langevin_step_size_learnable=True,
        langevin_step_size=0.2,
        langevin_steps=3,
        num_particles=K,
        particle_sampler=sampler,
        score_parameterization=parameterization,
        initial_belief=initial_belief,
        **kwargs,
    )


def _inputs(initial_rows: tuple[bool, ...]):
    g = torch.Generator().manual_seed(11)
    batch = len(initial_rows)
    return {
        "observation": torch.randn(batch, 3, generator=g),
        "previous_particles": torch.randn(batch, K, 2, generator=g),
        "previous_action": torch.randn(batch, 2, generator=g),
        "initial": torch.tensor(initial_rows),
        "initial_noise": torch.randn(batch, K, 2, generator=g),
        "langevin_noise": torch.randn(batch, 3, K, 2, generator=g),
    }


def _as_explicit_recursive_rows(belief: EnergyBelief, inputs: dict) -> dict:
    """The same batch with every initial row rewritten as a recursive row over the prior."""
    explicit = {name: value.clone() for name, value in inputs.items()}
    for row in range(inputs["initial"].shape[0]):
        if bool(inputs["initial"][row]):
            explicit["previous_particles"][row] = belief.prior_particles.detach()
            explicit["previous_action"][row] = 0.0
    explicit["initial"] = torch.zeros_like(inputs["initial"])
    return explicit


def _context(inputs: dict) -> tuple[torch.Tensor, ...]:
    return (
        inputs["observation"],
        inputs["previous_particles"],
        inputs["previous_action"],
        inputs["initial"],
    )


def test_prior_mode_replaces_the_initial_network_with_prior_particles():
    belief = _belief()
    assert belief.initial_energy is None
    assert belief.prior_particles is not None and tuple(belief.prior_particles.shape) == (K, 2)
    assert "initial" not in belief.energy_hidden_dims
    assert "prior_particles" in dict(belief.named_parameters())
    reference = _belief("branch")
    assert reference.initial_energy is not None and reference.prior_particles is None
    with pytest.raises(ValueError):
        _belief(observation_anchor=True)
    with pytest.raises(ValueError):
        EnergyBelief(3, 2, 2, [8], initial_belief="prior")  # no particle count
    with pytest.raises(ValueError):
        _belief("spline")


@pytest.mark.parametrize("parameterization", ["direct", "energy", "direct_joint", "direct_flat"])
@pytest.mark.parametrize("sampler", ["ula", "svgd"])
def test_initial_rows_are_recursive_rows_over_the_prior(parameterization, sampler):
    belief = _belief(parameterization=parameterization, sampler=sampler)
    inputs = _inputs((True, False, True))
    explicit = _as_explicit_recursive_rows(belief, inputs)
    query = torch.randn(3, K, 2, generator=torch.Generator().manual_seed(3))

    scored = belief.score_at(query, *_context(inputs), create_graph=False)
    expected = belief.score_at(query, *_context(explicit), create_graph=False)
    assert torch.allclose(scored, expected, atol=1e-6)
    if parameterization == "energy":
        assert torch.allclose(
            belief.log_potential(query, *_context(inputs)),
            belief.log_potential(query, *_context(explicit)),
            atol=1e-6,
        )
    particles, scores = belief.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    expected_particles, expected_scores = belief.particles_from_noise(
        **explicit, step_size=0.2, track_grad=False
    )
    assert torch.allclose(particles, expected_particles, atol=1e-6)
    assert torch.allclose(scores, expected_scores, atol=1e-6)


def test_the_recursive_row_keeps_its_own_previous_belief():
    belief = _belief()
    inputs = _inputs((True, False))
    changed = {name: value.clone() for name, value in inputs.items()}
    changed["previous_particles"][0] += 1.0  # the initial row ignores its previous particles ...
    same, _ = belief.particles_from_noise(**changed, step_size=0.2, track_grad=False)
    base, _ = belief.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    assert torch.equal(same, base)
    changed["previous_particles"][1] += 1.0  # ... the recursive row does not
    moved, _ = belief.particles_from_noise(**changed, step_size=0.2, track_grad=False)
    assert torch.equal(moved[0], base[0]) and not torch.allclose(moved[1], base[1])


def test_prior_mode_classifies_every_step_as_recursive_and_ignores_a_passed_initial_branch():
    belief = _belief()
    assert belief.classify_potential_branch(torch.tensor([True, True])) == "recursive"
    assert belief.classify_potential_branches(torch.tensor([[True, False], [True, True]])) == (
        "recursive",
        "recursive",
    )
    inputs = _inputs((True, True))
    automatic, _ = belief.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    # train.py / ppo.py pass the branch of the initial flags literally
    literal, _ = belief.particles_from_noise(
        **inputs, step_size=0.2, track_grad=False, potential_branch="initial"
    )
    assert torch.equal(automatic, literal)
    assert _belief("branch").classify_potential_branch(torch.tensor([True, True])) == "initial"


@pytest.mark.parametrize("gradient_mode", ["full", "tbptt_1"])
def test_prior_particles_are_trained_only_through_initial_rows(gradient_mode):
    belief = _belief()
    inputs = _inputs((False, False))
    particles, _ = belief.particles_from_noise(
        **inputs, step_size=0.2, track_grad=True, temporal_gradient_mode=gradient_mode
    )
    particles.square().sum().backward()
    grad = belief.prior_particles.grad
    assert grad is None or torch.count_nonzero(grad) == 0
    belief.zero_grad()
    inputs = _inputs((True, False))
    particles, _ = belief.particles_from_noise(
        **inputs, step_size=0.2, track_grad=True, temporal_gradient_mode=gradient_mode
    )
    particles.square().sum().backward()
    grad = belief.prior_particles.grad
    assert grad is not None and torch.count_nonzero(grad) > 0


def _chain_start(belief: EnergyBelief, inputs: dict) -> torch.Tensor:
    captured = {}
    original = EnergyBelief._transport_chain

    def spy(self, particles, *args, **kwargs):
        captured["start"] = particles.detach().clone()
        return original(self, particles, *args, **kwargs)

    EnergyBelief._transport_chain = spy
    try:
        belief.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    finally:
        EnergyBelief._transport_chain = original
    return captured["start"]


def test_initial_rows_start_where_recursive_rows_start():
    inputs = _inputs((True, False))
    # plain sampler: standard-normal noise for both rows (no anchor exists in prior mode)
    start = _chain_start(_belief(), inputs)
    assert torch.equal(start, inputs["initial_noise"])
    # warm start: the initial row starts at the prior particles, the recursive row at its previous ones
    warm = _belief(langevin_warm_start=True)
    start = _chain_start(warm, inputs)
    assert torch.allclose(start[0], warm.prior_particles.detach())
    assert torch.equal(start[1], inputs["previous_particles"][1])


def test_state_dict_round_trip_keeps_the_prior():
    source = _belief()
    with torch.no_grad():
        source.prior_particles.fill_(2.5)
    target = _belief()
    target.load_state_dict(source.state_dict())
    assert torch.equal(target.prior_particles, source.prior_particles)
    inputs = _inputs((True,))
    a, _ = source.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    b, _ = target.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    assert torch.equal(a, b)


def test_config_validation():
    base = load_config("config/trysvgdlight_mirror2_tf-svgd.json")
    assert base.to_dict()["model"]["initial_belief"] == "branch"
    with pytest.raises(ConfigError):  # the reference arm keeps the anchor on
        base.with_overrides({"model.initial_belief": "prior"})
    assert isinstance(
        base.with_overrides(
            {"model.initial_belief": "prior", "model.langevin_observation_anchor": False}
        ),
        ExperimentConfig,
    )
    with pytest.raises(ConfigError):
        base.with_overrides({"model.initial_belief": "spline"})


@pytest.mark.parametrize("task", ["mirror2", "tps", "tagr"])
def test_prior_configs_differ_from_the_reference_only_in_the_initial_belief(task):
    base = load_config(f"config/trysvgdlight_{task}_tf-svgd.json").to_dict()
    prior = load_config(f"config/trysvgdlight_{task}_tf-svgd-prior.json").to_dict()
    assert prior["experiment"]["label"] == f"trysvgdlight-{task}-tf-svgd-prior"
    assert prior["model"]["initial_belief"] == "prior"
    assert prior["model"]["langevin_observation_anchor"] is False
    base["experiment"]["label"] = prior["experiment"]["label"]
    base["model"].update({"initial_belief": "prior", "langevin_observation_anchor": False})
    assert base == prior


@pytest.mark.parametrize("task", ["mirror2", "tps", "tagr"])
def test_noanchor_control_configs_differ_from_the_reference_only_in_the_anchor(task):
    base = load_config(f"config/trysvgdlight_{task}_tf-svgd.json").to_dict()
    control = load_config(f"config/trysvgdlight_{task}_tf-svgd-noanchor.json").to_dict()
    assert control["experiment"]["label"] == f"trysvgdlight-{task}-tf-svgd-noanchor"
    assert control["model"]["initial_belief"] == "branch"
    base["experiment"]["label"] = control["experiment"]["label"]
    base["model"]["langevin_observation_anchor"] = False
    assert base == control


def test_prior_mode_trains_one_update(tmp_path):
    config = load_config("config/local.json").with_overrides(
        {
            "experiment.output_dir": str(tmp_path),
            "model.initial_belief": "prior",
            "model.langevin_observation_anchor": False,
        }
    )
    task = list(config["experiment"]["tasks"])[0]
    run, summaries = run_all(
        config, tasks=[task], seeds=[0], max_updates=1, label="pytest-prior", verbose=False
    )
    assert summaries[0]["status"] == "completed"
    seed_directory = run.root / task / "seed_000"
    with np.load(seed_directory / "arrays.npz", allow_pickle=False) as arrays:
        assert np.isfinite(arrays["gradient_norm"]).all()
        assert arrays["initial_energy_gradient_norm"].tolist() == [0.0]
        assert np.isfinite(arrays["transition_energy_gradient_norm"]).all()
        assert arrays["score_nonfinite_count"].tolist() == [0]
    checkpoints = sorted(seed_directory.rglob("*.pt"))
    assert checkpoints
    state = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    keys = set(_flatten_keys(state))
    assert any(key.endswith("belief.prior_particles") for key in keys)
    assert not any("belief.initial_energy" in key for key in keys)


def _flatten_keys(value, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten_keys(item, f"{prefix}{key}.")
    else:
        yield prefix.rstrip(".")
