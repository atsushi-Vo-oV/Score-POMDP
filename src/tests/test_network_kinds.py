"""Fixed orthonormal-basis heads and per-role network kinds (mlp / kan / basis)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sb_pomdp.baselines import make_baseline_actor_critic
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.envs import make_env
from sb_pomdp.networks import (
    KANLinear,
    NetworkKinds,
    OrthonormalBasisHead,
    active_network_kinds,
    make_head_network,
    use_network_kinds,
)
from sb_pomdp.train import model_for_environment
from sb_pomdp.train_baselines import _make_model


def test_basis_features_are_orthonormal_per_coordinate() -> None:
    head = OrthonormalBasisHead(3, 2, projection_dim=3, order=5, seed=1)
    # Feed inputs whose projection is uniform on (-1, 1): invert tanh and Q.
    u = torch.rand(20000, 3) * 2 - 1
    z = torch.atanh(u * 0.999999) @ head.projection  # projection is orthonormal -> Q^T inverse
    feats = head.features(z)  # [N, 1 + 3*5]
    # Gram matrix of the basis functions of one coordinate is ~identity.
    block = feats[:, 1:11]  # coordinate 0: cos k=1..5 then sin k=1..5
    gram = block.T @ block / len(block)
    assert torch.allclose(gram, torch.eye(10), atol=0.05)
    assert torch.allclose((feats[:, :1].T @ block / len(block)), torch.zeros(1, 10), atol=0.05)
    # The projection rows are orthonormal.
    assert torch.allclose(head.projection @ head.projection.T, torch.eye(3), atol=1e-5)


def test_basis_head_trains_only_the_linear_coefficients() -> None:
    head = OrthonormalBasisHead(8, 3, projection_dim=4, order=3)
    trainable = [name for name, parameter in head.named_parameters() if parameter.requires_grad]
    assert sorted(trainable) == ["linear.bias", "linear.weight"]
    assert head.linear.in_features == 1 + 2 * 4 * 3
    output = head(torch.randn(5, 7, 8))
    assert output.shape == (5, 7, 3)
    assert torch.all(output == 0.0)  # zero-initialised coefficients


def test_basis_head_fits_a_smooth_target() -> None:
    torch.manual_seed(0)
    head = OrthonormalBasisHead(2, 1, projection_dim=2, order=6)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-2)
    x = torch.randn(512, 2)
    # An additive function of the (fixed, rotated) projected coordinates - the
    # class this basis represents; interactions across coordinates are not.
    projected = x @ head.projection.T
    target = torch.sin(1.5 * projected[:, :1]) + 0.5 * torch.cos(2.0 * projected[:, 1:])
    for _ in range(300):
        loss = (head(x) - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss) < 0.15 * float(target.var())


def test_network_kind_context_selects_head_classes() -> None:
    assert active_network_kinds().head == "mlp"
    with use_network_kinds(NetworkKinds(head="basis", basis_order=2, basis_projection_dim=4)):
        assert isinstance(make_head_network(6, [8], 2), OrthonormalBasisHead)
    with use_network_kinds(NetworkKinds(head="kan", kan_grid_size=4)):
        assert isinstance(make_head_network(6, [8], 2)[0], KANLinear)
    assert active_network_kinds().head == "mlp"
    with pytest.raises(ValueError):
        NetworkKinds(trunk="basis")


def _score_model(task: str, overrides: dict):
    config = load_config("config/local.json").with_overrides(
        {"model.num_particles": 4, "model.langevin_steps": 2, **overrides}
    )
    env = make_env(task, dict(config["environment"]["tasks"][task]), seed=0)
    return model_for_environment(env, dict(config["model"]), torch.device("cpu")), env, config


def test_score_model_builds_basis_and_kan_heads() -> None:
    model, env, _ = _score_model("masked_cartpole", {"model.head_network_kind": "basis"})
    assert isinstance(model.categorical_head, OrthonormalBasisHead)
    assert isinstance(model.value_head, OrthonormalBasisHead)
    env.reset(seed=0)
    particles = torch.randn(1, 4, env.state_dim)
    sample = model.sample_policy(model.encode(particles, torch.randn_like(particles)), deterministic=False)
    assert sample.action.shape == (1,)
    model, env, _ = _score_model(
        "light_dark",
        {
            "model.head_network_kind": "basis",
            "model.continuous_policy_kind": "gaussian",
        },
    )
    assert isinstance(model.gaussian_policy.parameter_network, OrthonormalBasisHead)
    model, _, _ = _score_model(
        "light_dark", {"model.head_network_kind": "kan", "model.trunk_network_kind": "kan"}
    )
    assert isinstance(model.value_head[0], KANLinear)
    assert isinstance(model.diffusion_policy.denoiser[0], KANLinear)


def test_baselines_build_with_basis_heads_and_kan_trunks() -> None:
    config = load_config("config/local.json").with_overrides(
        {"model.head_network_kind": "basis", "model.trunk_network_kind": "kan"}
    )
    env = make_env("masked_cartpole", dict(config["environment"]["tasks"]["masked_cartpole"]), seed=0)
    for method in ("gru", "rnn", "particle_filter"):
        model = _make_model(env, method, dict(config["model"]), dict(config["comparison"]), torch.device("cpu"))
        assert isinstance(model.heads.categorical_head, OrthonormalBasisHead)
        assert isinstance(model.heads.value_head, OrthonormalBasisHead)
        hidden = model.initial_hidden(1)
        inp = torch.cat((torch.randn(1, env.observation_dim), torch.zeros(1, env.action_spec.feature_dim)), -1)
        feat, _ = model.step(inp, hidden, torch.ones(1, dtype=torch.bool))
        assert torch.isfinite(feat).all()
    with use_network_kinds(NetworkKinds(head="mlp", trunk="mlp")):
        plain = make_baseline_actor_critic(
            "gru",
            observation_dim=3,
            state_dim=2,
            action_kind="discrete",
            hidden_dims=[8],
            discrete_actions=2,
            recurrent_hidden_dim=4,
        )
    assert isinstance(plain.heads.value_head[0], torch.nn.Linear)


def test_config_validates_the_network_kind_keys() -> None:
    config = load_config("config/local.json")
    resolved = config.to_dict()["model"]
    assert resolved["head_network_kind"] == "mlp" and resolved["trunk_network_kind"] == "mlp"
    assert resolved["basis_order"] == 4 and resolved["basis_projection_dim"] == 16
    with pytest.raises(ConfigError):
        config.with_overrides({"model.head_network_kind": "spline"})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.trunk_network_kind": "basis"})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.basis_order": 0})
    accepted = config.with_overrides({"model.head_network_kind": "basis", "model.basis_order": 6})
    assert accepted["model"]["basis_order"] == 6


def test_basis_head_parameter_count_is_modest() -> None:
    head = OrthonormalBasisHead(64, 2, projection_dim=16, order=4)
    count = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert count == (1 + 2 * 16 * 4) * 2 + 2
    assert np.isfinite(count)
