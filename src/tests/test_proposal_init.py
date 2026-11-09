"""model.langevin_proposal: learned chain start (affine / network) for the noise-started steps."""
from __future__ import annotations

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, ExperimentConfig, load_config


def _belief(proposal: str, sampler: str = "svgd", anchor: bool = True, **kwargs) -> EnergyBelief:
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
        observation_anchor=anchor,
        langevin_proposal=proposal,
        particle_sampler=sampler,
        score_parameterization="direct",
        **kwargs,
    )


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


@pytest.mark.parametrize("proposal", ["affine", "network"])
@pytest.mark.parametrize("sampler", ["svgd", "ula"])
def test_fresh_proposal_is_identical_to_noise_start(proposal, sampler):
    """Zero-initialised mean / log-scale: the first forward equals the plain sampler exactly."""
    plain, learned = _belief("noise", sampler), _belief(proposal, sampler)
    learned.load_state_dict(plain.state_dict(), strict=False)
    inputs = _inputs((True, False, False))
    for belief in (plain, learned):
        torch.manual_seed(0)
    p0, s0 = plain.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    p1, s1 = learned.particles_from_noise(**inputs, step_size=0.2, track_grad=False)
    assert torch.equal(p0, p1)
    assert torch.equal(s0, s1)


@pytest.mark.parametrize("proposal", ["affine", "network"])
def test_proposal_parameters_receive_gradients_only_through_recursive_rows(proposal):
    belief = _belief(proposal)
    names = [n for n, _ in belief.named_parameters() if n.startswith("proposal_")]
    assert names
    # all-initial batch: the anchor owns every row, the proposal is unused
    inputs = _inputs((True, True))
    particles, _ = belief.particles_from_noise(**inputs, step_size=0.2, track_grad=True)
    particles.square().sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for n, p in belief.named_parameters() if n in names)
    belief.zero_grad()
    inputs = _inputs((False, False))
    particles, _ = belief.particles_from_noise(**inputs, step_size=0.2, track_grad=True)
    particles.square().sum().backward()
    grads = {n: p.grad for n, p in belief.named_parameters() if n in names}
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in grads.values())


def test_affine_proposal_moves_the_chain_start():
    belief = _belief("affine", anchor=False)
    with torch.no_grad():
        belief.proposal_mean.fill_(3.0)
        belief.proposal_log_scale.fill_(-10.0)  # clamped to the anchor range, i.e. a tiny scale
    inputs = _inputs((False,))
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
    assert torch.allclose(captured["start"], torch.full((1, 4, 2), 3.0), atol=0.1)


def test_network_proposal_depends_on_observation_and_previous_particles():
    belief = _belief("network", anchor=False)
    with torch.no_grad():
        for layer in belief.proposal_network.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.normal_(0.0, 0.5)
    a, b = _inputs((False,)), _inputs((False,))
    b["observation"] = b["observation"] + 1.0
    pa, _ = belief.particles_from_noise(**a, step_size=0.2, track_grad=False)
    pb, _ = belief.particles_from_noise(**b, step_size=0.2, track_grad=False)
    assert not torch.allclose(pa, pb)


def test_proposal_conflicts_with_warm_start():
    with pytest.raises(ValueError):
        _belief("affine", langevin_warm_start=True)
    with pytest.raises(ValueError):
        _belief("uniform", langevin_warm_start=True)
    with pytest.raises(ValueError):
        _belief("spline")
    with pytest.raises(ValueError):
        _belief("uniform", langevin_uniform_half_width=0.0)


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


def test_uniform_proposal_starts_recursive_rows_in_the_box_and_keeps_the_anchor():
    """U(-w, w) per coordinate on the noise-started rows; the anchored initial rows are untouched."""
    plain, uniform = _belief("noise"), _belief("uniform", langevin_uniform_half_width=2.0)
    uniform.load_state_dict(plain.state_dict(), strict=False)
    inputs = _inputs((True, False, False))
    inputs["initial_noise"] = torch.randn(3, 4, 2, generator=torch.Generator().manual_seed(3)) * 4.0
    start_plain, start_uniform = _chain_start(plain, inputs), _chain_start(uniform, inputs)
    assert torch.equal(start_plain[0], start_uniform[0])  # initial row: anchor in both arms
    assert start_uniform[1:].abs().max() <= 2.0
    assert start_plain[1:].abs().max() > 2.0  # the wide Gaussian noise left the box
    # monotone map of the stored noise: the same noise gives the same start (replay), order preserved
    expected = (2.0 * torch.special.ndtr(inputs["initial_noise"][1:]) - 1.0) * 2.0
    assert torch.allclose(start_uniform[1:], expected)
    assert not any(n.startswith("proposal_") for n, _ in uniform.named_parameters())


def test_uniform_proposal_default_half_width_matches_unit_variance():
    belief = _belief("uniform", anchor=False)
    g = torch.Generator().manual_seed(1)
    inputs = _inputs((False,) * 64)
    inputs["initial_noise"] = torch.randn(64, 4, 2, generator=g)
    start = _chain_start(belief, inputs)
    assert start.abs().max() <= 3.0 ** 0.5 + 1e-6
    assert abs(start.var().item() - 1.0) < 0.1


def test_config_validation(tmp_path):
    base = load_config("config/trysvgdlight_mirror2_tf-svgd.json")
    assert base.to_dict()["model"]["langevin_proposal"] == "noise"
    assert isinstance(base.with_overrides({"model.langevin_proposal": "network"}), ExperimentConfig)
    assert isinstance(base.with_overrides({"model.langevin_proposal": "uniform"}), ExperimentConfig)
    with pytest.raises(ConfigError):
        base.with_overrides({"model.langevin_proposal": "spline"})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.langevin_proposal": "affine", "model.langevin_warm_start": True})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.langevin_proposal": "uniform", "model.langevin_warm_start": True})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.langevin_uniform_half_width": 0.0})
    uniform = load_config("config/trysvgdlight_mirror2_tf-svgd-uniform-init.json").to_dict()["model"]
    assert uniform["langevin_proposal"] == "uniform"
    assert abs(uniform["langevin_uniform_half_width"] - 3.0 ** 0.5) < 1e-12
