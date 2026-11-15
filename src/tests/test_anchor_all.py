"""model.langevin_observation_anchor_steps: the observation anchor at every step (experiment 27)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, load_config


def _belief(proposal: str = "noise", **kwargs) -> EnergyBelief:
    torch.manual_seed(5)
    belief = EnergyBelief(
        observation_dim=3, state_dim=2, action_feature_dim=2, hidden_dims=[8], score_clip=0.0, particle_clip=0.0,
        langevin_step_size_learnable=True, langevin_step_size=0.2, langevin_steps=3, observation_anchor=True,
        langevin_proposal=proposal, particle_sampler="svgd", score_parameterization="direct", **kwargs,
    )
    # the anchor starts at the identity (zero last layer); give it weights so that it does something
    torch.manual_seed(7)
    with torch.no_grad():
        for parameter in belief.observation_anchor.parameters():
            parameter.normal_(std=0.3)
    return belief


def _inputs(initial_rows: tuple[bool, ...]):
    g = torch.Generator().manual_seed(11)
    batch = len(initial_rows)
    return {
        "observation": torch.randn(batch, 3, generator=g),
        "previous_particles": torch.randn(batch, 4, 2, generator=g),
        "previous_action": torch.randn(batch, 2, generator=g),
        "initial": torch.tensor(initial_rows),
        "initial_noise": torch.randn(batch, 4, 2, generator=g),
        "langevin_noise": torch.randn(batch, 3, 4, 2, generator=g),
    }


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


def test_default_is_the_initial_only_anchor_bit_for_bit():
    a, b = _belief(), _belief(observation_anchor_steps="initial")
    inputs = _inputs((True, False, False))
    out_a = a.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    out_b = b.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    assert torch.equal(out_a[0], out_b[0]) and torch.equal(out_a[1], out_b[1])
    assert torch.equal(_chain_start(a, inputs)[1:], inputs["initial_noise"][1:])  # recursive rows: plain noise


@pytest.mark.parametrize("proposal", ["noise", "uniform"])
def test_every_row_starts_at_the_anchor(proposal):
    first, every = _belief(proposal), _belief(proposal, observation_anchor_steps="all")
    every.load_state_dict(first.state_dict())
    assert sorted(n for n, _ in every.named_parameters()) == sorted(n for n, _ in first.named_parameters())
    inputs = _inputs((True, False, False))
    start_first, start_every = _chain_start(first, inputs), _chain_start(every, inputs)
    assert torch.equal(start_first[:1], start_every[:1])  # the t = 0 row is unchanged
    mean, log_scale = every.observation_anchor(inputs["observation"]).chunk(2, dim=-1)
    base = inputs["initial_noise"]
    if proposal == "uniform":
        base = (2.0 * torch.special.ndtr(base) - 1.0) * 3.0 ** 0.5
    expected = mean[:, None, :] + log_scale.clamp(-4.0, 2.0).exp()[:, None, :] * base
    assert torch.allclose(start_every[1:], expected[1:].detach(), atol=1e-6)
    assert not torch.allclose(start_every[1:], start_first[1:])


def test_a_fresh_anchor_at_every_step_equals_the_plain_start():
    torch.manual_seed(5)
    kwargs = dict(observation_dim=3, state_dim=2, action_feature_dim=2, hidden_dims=[8], score_clip=0.0, particle_clip=0.0,
                  langevin_step_size_learnable=True, langevin_step_size=0.2, langevin_steps=3, observation_anchor=True,
                  particle_sampler="svgd", score_parameterization="direct")
    first = EnergyBelief(**kwargs)
    every = EnergyBelief(**kwargs, observation_anchor_steps="all")
    every.load_state_dict(first.state_dict())
    inputs = _inputs((True, False, False))
    out_a = first.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    out_b = every.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    assert torch.equal(out_a[0], out_b[0])  # zero last layer: mu = 0, sigma = 1


def test_the_anchor_learns_from_recursive_rows_only_when_it_moves_them():
    inputs = _inputs((False, False, False))
    for steps, expect in (("initial", False), ("all", True)):
        belief = _belief(observation_anchor_steps=steps)
        particles, _ = belief.particles_from_noise(**inputs, step_size=0.2, track_grad=True)
        particles.square().sum().backward()
        moved = any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in belief.observation_anchor.parameters())
        assert moved is expect


def test_invalid_combinations():
    with pytest.raises(ValueError):
        _belief(observation_anchor_steps="recursive")
    with pytest.raises(ValueError):
        _belief("affine", observation_anchor_steps="all")
    with pytest.raises(ValueError):
        _belief(observation_anchor_steps="all", langevin_warm_start=True)
    with pytest.raises(ValueError):
        EnergyBelief(observation_dim=3, state_dim=2, action_feature_dim=2, hidden_dims=[8], score_clip=0.0, particle_clip=0.0,
                     score_parameterization="direct", observation_anchor=False, observation_anchor_steps="all")
    base = load_config("config/trysvgdlight_mirror2_tf-svgd.json")
    assert base.to_dict()["model"]["langevin_observation_anchor_steps"] == "initial"
    with pytest.raises(ConfigError):
        base.with_overrides({"model.langevin_observation_anchor_steps": "recursive"})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.langevin_observation_anchor_steps": "all", "model.langevin_proposal": "network"})
    with pytest.raises(ConfigError):
        load_config("config/trysvgdlight2_mirror2_tf-svgd.json").with_overrides({"model.langevin_observation_anchor_steps": "all"})


def _generator():
    spec = importlib.util.spec_from_file_location("anchor_all_generate", Path("scripts/anchor_all_generate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_arms_differ_from_their_default_only_in_the_listed_keys():
    assets = list(_generator().assets())
    assert len(assets) == 16
    for name, source, label, campaign, stem, keys, updates, _ in assets:
        arm, base = load_config(f"config/{name}.json").to_dict(), load_config(f"config/{source}.json").to_dict()
        assert arm["model"]["langevin_observation_anchor"] is True and arm["model"]["langevin_observation_anchor_steps"] == "all"
        assert arm["model"]["langevin_steps"] == (1 if name.endswith("-l1") else 12) and base["model"]["langevin_steps"] == 12
        assert base["model"]["langevin_observation_anchor"] is (not name.startswith("trysvgdlight2"))
        assert {key: arm["model"][key] for key in keys} == keys
        for key in keys:
            arm["model"][key] = base["model"][key]
        arm["experiment"]["label"] = base["experiment"]["label"]
        assert arm == base
        for seed in range(10, 15):
            text = Path(f"jobs/anchor_all/{stem}_s{seed}.sh").read_text(encoding="utf-8")
            assert f'SB_POMDP_CONFIG="config/{name}.json"' in text and f'SB_POMDP_CAMPAIGN_ID="{campaign}"' in text
            assert f"SB_POMDP_SHARD_INDEX={seed - 9}" in text and f"SB_POMDP_SEGMENT_UPDATES={updates}" in text
            assert text.rstrip().endswith("exec bash jobs/genkai_production.sh")
