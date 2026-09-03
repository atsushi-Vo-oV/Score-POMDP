"""PF-RNN / DPFRL-faithful particle filter: stochastic transitions, soft resampling, MGF features."""

from __future__ import annotations

import pytest
import torch

from sb_pomdp.baselines import ParticleFilterActorCritic
from sb_pomdp.config import ConfigError, load_config

K, D, F = 6, 5, 3  # particles, latent dim, raw input features


def _model(**overrides: object) -> ParticleFilterActorCritic:
    kwargs: dict[str, object] = {
        "observation_dim": F,
        "action_kind": "continuous",
        "num_particles": K,
        "particle_dim": D,
        "hidden_dims": [8, 8],
        "soft_alpha": 0.9,
        "policy_condition_dim": 8,
        "head_hidden_dims": [8],
        "continuous_policy_kind": "gaussian",
        "action_low": [-1.0],
        "action_high": [1.0],
        "variant": "dpfrl",
        "mgf_features": 4,
    }
    kwargs.update(overrides)
    torch.manual_seed(0)
    return ParticleFilterActorCritic(**kwargs)  # type: ignore[arg-type]


def _inputs(model: ParticleFilterActorCritic, time: int, batch: int):
    features = torch.randn(time, batch, F)
    noise = torch.stack([model.draw_noise(batch) for _ in range(time)])
    starts = torch.zeros(time, batch, dtype=torch.bool)
    starts[0] = True
    return torch.cat((features, noise), dim=-1), starts


def test_noise_layout_and_input_width() -> None:
    model = _model()
    assert model.noise_dim == K * (D + 1)
    assert model.observation_dim == F + K * (D + 1)
    noise = model.draw_noise(7)
    assert noise.shape == (7, K * (D + 1))
    uniforms = noise[:, K * D :]
    assert torch.all(uniforms >= 0) and torch.all(uniforms < 1)
    plain = _model(variant="deterministic", mgf_features=0)
    assert plain.noise_dim == 0 and plain.observation_dim == F


def test_particles_stay_diverse_and_weights_are_informative() -> None:
    model = _model()
    inputs, starts = _inputs(model, time=20, batch=3)
    hidden = model.initial_hidden(3)
    features, hidden = model.encode_sequence(inputs, hidden, starts)
    particles, log_weights = model._unpack(hidden)
    spread = particles.std(dim=1).mean() / (particles.abs().mean() + 1e-8)
    assert float(spread) > 0.05, "stochastic transitions must keep the ensemble spread out"
    weights = log_weights.exp()
    assert torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-5)
    ess = 1.0 / (weights**2).sum(-1)
    assert torch.all(ess <= K + 1e-4)
    assert torch.isfinite(features).all() and features.shape == (20, 3, 8)


def test_replay_is_exact_given_the_stored_noise_and_differs_with_fresh_noise() -> None:
    model = _model()
    inputs, starts = _inputs(model, time=12, batch=2)
    hidden = model.initial_hidden(2)
    a, ha = model.encode_sequence(inputs, hidden, starts)
    b, hb = model.encode_sequence(inputs, hidden, starts)
    assert torch.allclose(a, b) and torch.allclose(ha, hb)
    other = inputs.clone()
    other[..., F:] = torch.stack([model.draw_noise(2) for _ in range(12)])
    c, _ = model.encode_sequence(other, hidden, starts)
    assert not torch.allclose(a, c)


def test_step_matches_encode_sequence_with_noise() -> None:
    model = _model()
    inputs, starts = _inputs(model, time=6, batch=2)
    hidden = model.initial_hidden(2)
    seq, _ = model.encode_sequence(inputs, hidden, starts)
    outputs = []
    for t in range(6):
        feat, hidden = model.step(inputs[t], hidden, starts[t])
        outputs.append(feat)
    assert torch.allclose(torch.stack(outputs), seq, atol=1e-6)


def test_soft_resampling_importance_weights_and_gradients() -> None:
    model = _model(soft_alpha=0.5)
    inputs, starts = _inputs(model, time=8, batch=2)
    hidden = model.initial_hidden(2)
    features, hidden = model.encode_sequence(inputs, hidden, starts)
    _particles, log_weights = model._unpack(hidden)
    assert torch.allclose(log_weights.exp().sum(-1), torch.ones(2), atol=1e-5)
    loss = features.square().mean()
    loss.backward()
    named = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert any("observation_function" in n for n in named), "weights must carry gradient"
    assert any("noise_scale" in n for n in named), "learned noise scale must carry gradient"
    assert any("mgf_vectors" in n for n in named)


def test_missing_noise_columns_are_rejected() -> None:
    model = _model()
    bad = torch.randn(4, 2, F)
    starts = torch.zeros(4, 2, dtype=torch.bool)
    starts[0] = True
    with pytest.raises(ValueError):
        model.encode_sequence(bad, model.initial_hidden(2), starts)


def test_config_validates_pf_variant_keys() -> None:
    config = load_config("config/local.json")
    resolved = config.to_dict()["comparison"]
    assert resolved["pf_variant"] == "deterministic" and resolved["pf_mgf_features"] == 0
    with pytest.raises(ConfigError):
        config.with_overrides({"comparison.pf_variant": "bootstrap"})
    with pytest.raises(ConfigError):
        config.with_overrides({"comparison.pf_mgf_features": -1})
    accepted = config.with_overrides(
        {"comparison.pf_variant": "dpfrl", "comparison.pf_mgf_features": 8}
    )
    assert accepted["comparison"]["pf_variant"] == "dpfrl"
