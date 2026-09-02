"""Kolmogorov-Arnold energy networks (model.energy_network_kind=kan)."""

from __future__ import annotations

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.networks import (
    KANLinear,
    count_kan_parameters,
    count_mlp_parameters,
    make_kan,
    make_mlp,
    make_network,
    matched_kan_hidden_dims,
)
from sb_pomdp.policies import ScoreBeliefActorCritic


def test_b_splines_form_a_partition_of_unity_inside_the_grid() -> None:
    layer = KANLinear(3, 2, grid_size=6, spline_order=3, grid_range=2.0)
    inputs = torch.linspace(-1.9, 1.9, 25)[:, None].repeat(1, 3)
    bases = layer.b_splines(inputs)
    assert bases.shape == (25, 3, layer.num_bases)
    assert torch.allclose(bases.sum(dim=-1), torch.ones(25, 3), atol=1e-5)
    outside = layer.b_splines(torch.full((2, 3), 5.0))
    assert torch.all(outside == 0.0)


def test_kan_layer_forward_shape_and_leading_dims() -> None:
    layer = KANLinear(4, 5)
    output = layer(torch.randn(7, 3, 4))
    assert output.shape == (7, 3, 5)
    assert torch.isfinite(output).all()


def test_kan_fits_a_nonlinear_target_that_a_linear_map_cannot() -> None:
    torch.manual_seed(0)
    x = torch.linspace(-2.0, 2.0, 256)[:, None]
    target = torch.sin(2.0 * x) + 0.3 * x.square()
    network = make_kan(1, [8], 1, grid_size=8, spline_order=3, grid_range=3.0)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    for _ in range(400):
        loss = (network(x) - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss) < 0.05
    linear_residual = float((target - target.mean()).square().mean())
    assert float(loss) < 0.2 * linear_residual


def test_kan_supports_first_and_second_order_input_gradients() -> None:
    torch.manual_seed(1)
    network = make_kan(2, [8], 1)
    x = torch.randn(5, 2, requires_grad=True)
    energy = network(x).sum()
    (gradient,) = torch.autograd.grad(energy, x, create_graph=True)
    assert gradient.shape == (5, 2)
    assert torch.isfinite(gradient).all()
    (second,) = torch.autograd.grad(gradient.square().sum(), x, allow_unused=True)
    assert second is not None and torch.isfinite(second).all()


def test_make_network_selects_the_kind() -> None:
    mlp = make_network("mlp", 3, [4], 1)
    kan = make_network("kan", 3, [4], 1, kan_grid_size=5)
    assert isinstance(mlp[0], torch.nn.Linear)
    assert isinstance(kan[0], KANLinear)
    with pytest.raises(ValueError):
        make_network("spline", 3, [4], 1)
    assert sum(p.numel() for p in kan.parameters()) > sum(
        p.numel() for p in make_mlp(3, [4], 1).parameters()
    )


def test_energy_belief_runs_with_kan_energies() -> None:
    torch.manual_seed(0)
    belief = EnergyBelief(3, 2, 2, [8], energy_network_kind="kan", kan_grid_size=6)
    assert belief.energy_network_kind == "kan"
    assert isinstance(belief.initial_energy[0], KANLinear)
    observation = torch.randn(2, 3)
    previous = torch.randn(2, 4, 2)
    action = torch.randn(2, 2)
    initial = torch.tensor([True, False])
    initial_noise = torch.randn(2, 4, 2)
    step_noise = torch.randn(2, 3, 4, 2)
    particles, scores = belief.particles_from_noise(
        observation,
        previous,
        action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=step_noise,
        step_size=0.05,
        track_grad=True,
    )
    assert particles.shape == (2, 4, 2)
    assert torch.isfinite(particles).all() and torch.isfinite(scores).all()
    particles.square().sum().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in belief.observation_energy.parameters()
    )
    with pytest.raises(ValueError):
        EnergyBelief(3, 2, 2, [8], energy_network_kind="spline")


def test_score_model_and_config_accept_kan() -> None:
    config = load_config("config/local.json")
    resolved = config.to_dict()["model"]
    assert resolved["energy_network_kind"] == "mlp"
    assert resolved["kan_grid_size"] == 8
    with pytest.raises(ConfigError):
        config.with_overrides({"model.energy_network_kind": "spline"})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.kan_grid_size": 0})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.kan_grid_range": 0.0})
    accepted = config.with_overrides(
        {
            "model.energy_network_kind": "kan",
            "model.kan_grid_size": 5,
            "model.langevin_steps": 2,
        }
    )
    torch.manual_seed(0)
    model = ScoreBeliefActorCritic(
        observation_dim=3,
        state_dim=2,
        action_kind="discrete",
        action_feature_dim=2,
        discrete_actions=2,
        action_low=None,
        action_high=None,
        model_config=dict(accepted["model"]),
    )
    assert isinstance(model.belief.transition_energy[0], KANLinear)
    assert model.belief.transition_energy[0].grid_size == 5


def test_matched_widths_track_the_mlp_parameter_budget() -> None:
    for input_dim, hidden in ((6, [48, 48]), (12, [48, 48]), (4, [32]), (20, [64, 64, 64])):
        widths = matched_kan_hidden_dims(input_dim, hidden, 1, num_bases=11)
        assert len(widths) == len(hidden)
        mlp = count_mlp_parameters(input_dim, hidden, 1)
        kan = count_kan_parameters(input_dim, widths, 1, 11)
        assert abs(kan - mlp) / mlp < 0.15
        model = make_kan(input_dim, widths, 1, grid_size=8, spline_order=3)
        assert sum(p.numel() for p in model.parameters()) == kan


def test_energy_belief_matches_parameters_by_default() -> None:
    torch.manual_seed(0)
    mlp = EnergyBelief(2, 2, 2, [48, 48])
    kan = EnergyBelief(2, 2, 2, [48, 48], energy_network_kind="kan")
    wide = EnergyBelief(
        2, 2, 2, [48, 48], energy_network_kind="kan", kan_match_parameters=False
    )
    count = lambda module: sum(p.numel() for p in module.parameters())
    assert abs(count(kan) - count(mlp)) / count(mlp) < 0.15
    assert count(wide) > 5 * count(mlp)
    assert kan.energy_hidden_dims["observation"] != [48, 48]
    assert wide.energy_hidden_dims["observation"] == [48, 48]
    assert mlp.energy_hidden_dims["transition"] == [48, 48]
