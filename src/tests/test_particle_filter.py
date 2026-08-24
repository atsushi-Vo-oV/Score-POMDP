"""Invariants of the deterministic particle-filter-style baseline."""

from __future__ import annotations

import pytest
import torch

from sb_pomdp.baselines import ParticleFilterActorCritic
from sb_pomdp.config import ConfigError, load_config


def _make_model(**overrides: object) -> ParticleFilterActorCritic:
    kwargs: dict[str, object] = {
        "observation_dim": 3,
        "action_kind": "continuous",
        "num_particles": 4,
        "particle_dim": 3,
        "hidden_dims": [8, 8],
        "soft_alpha": 0.9,
        "policy_condition_dim": 8,
        "head_hidden_dims": [8],
        "continuous_policy_kind": "diffusion",
        "diffusion_steps": 2,
        "action_low": [-1.0],
        "action_high": [1.0],
    }
    kwargs.update(overrides)
    return ParticleFilterActorCritic(**kwargs)  # type: ignore[arg-type]


def _sequence(time: int, batch: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.randn(time, batch, dim)
    episode_starts = torch.zeros(time, batch, dtype=torch.bool)
    episode_starts[0] = True
    return observations, episode_starts


def test_full_and_tbptt_share_identical_forward_values() -> None:
    torch.manual_seed(0)
    model = _make_model()
    observations, episode_starts = _sequence(5, 2, 3)
    hidden = model.initial_hidden(2)
    features_full, hidden_full = model.encode_sequence(
        observations, hidden, episode_starts, temporal_gradient_mode="full"
    )
    features_tbptt, hidden_tbptt = model.encode_sequence(
        observations, hidden, episode_starts, temporal_gradient_mode="tbptt_1"
    )
    assert torch.allclose(features_full, features_tbptt, atol=1e-6)
    assert torch.allclose(hidden_full, hidden_tbptt, atol=1e-6)


def test_tbptt_detaches_the_temporal_graph_and_full_keeps_it() -> None:
    torch.manual_seed(0)
    model = _make_model()
    observations, episode_starts = _sequence(4, 1, 3)
    hidden = model.initial_hidden(1)

    features, _ = model.encode_sequence(
        observations, hidden, episode_starts, temporal_gradient_mode="full"
    )
    (gradient,) = torch.autograd.grad(
        features[-1].sum(), model.anchors, retain_graph=False, allow_unused=False
    )
    assert torch.isfinite(gradient).all()

    features, _ = model.encode_sequence(
        observations, hidden, episode_starts, temporal_gradient_mode="tbptt_1"
    )
    # In one-step TBPTT the final feature depends on the anchors only through
    # detached particles from step zero, so the anchor gradient must vanish.
    (gradient,) = torch.autograd.grad(
        features[-1].sum(), model.anchors, allow_unused=True
    )
    assert gradient is None or torch.count_nonzero(gradient) == 0


def test_episode_start_ignores_the_incoming_hidden_state() -> None:
    torch.manual_seed(0)
    model = _make_model()
    observations, episode_starts = _sequence(3, 2, 3)
    zeros = model.initial_hidden(2)
    garbage = torch.randn_like(zeros)
    features_a, _ = model.encode_sequence(observations, zeros, episode_starts)
    features_b, _ = model.encode_sequence(observations, garbage, episode_starts)
    assert torch.allclose(features_a, features_b, atol=1e-6)


def test_mid_episode_reset_rebuilds_particles_from_the_new_observation() -> None:
    torch.manual_seed(0)
    model = _make_model()
    observations = torch.randn(4, 1, 3)
    episode_starts = torch.tensor([[True], [False], [True], [False]])
    hidden = model.initial_hidden(1)
    features, _ = model.encode_sequence(observations, hidden, episode_starts)
    fresh_features, _ = model.encode_sequence(
        observations[2:], hidden, episode_starts[:2]
    )
    assert torch.allclose(features[2:], fresh_features, atol=1e-6)


def test_step_matches_encode_sequence() -> None:
    torch.manual_seed(0)
    model = _make_model()
    observations, episode_starts = _sequence(3, 2, 3)
    hidden = model.initial_hidden(2)
    sequence_features, _ = model.encode_sequence(observations, hidden, episode_starts)
    stepped = []
    for time_index in range(observations.shape[0]):
        features, hidden = model.step(
            observations[time_index], hidden, episode_starts[time_index]
        )
        stepped.append(features)
    assert torch.allclose(sequence_features, torch.stack(stepped), atol=1e-6)


def test_log_weights_stay_normalised_and_finite() -> None:
    torch.manual_seed(0)
    model = _make_model()
    observations, episode_starts = _sequence(6, 2, 3)
    hidden = model.initial_hidden(2)
    _, final_hidden = model.encode_sequence(observations, hidden, episode_starts)
    _, log_weights = model._unpack(final_hidden)
    assert torch.isfinite(log_weights).all()
    assert torch.allclose(
        log_weights.logsumexp(dim=-1), torch.zeros(2), atol=1e-5
    )


def test_soft_alpha_floor_bounds_the_weight_ratio() -> None:
    # With the uniform floor no weight can shrink below (1 - alpha) / K after
    # one step of an all-equal compatibility, preventing deterministic collapse.
    torch.manual_seed(0)
    model = _make_model(soft_alpha=0.5)
    observations, episode_starts = _sequence(8, 1, 3)
    hidden = model.initial_hidden(1)
    _, final_hidden = model.encode_sequence(observations, hidden, episode_starts)
    _, log_weights = model._unpack(final_hidden)
    compatibility_span = 0.0
    with torch.no_grad():
        particles, _ = model._unpack(final_hidden)
        tiled = observations[-1].unsqueeze(1).expand(1, model.num_particles, 3)
        scores = model.weight_net(torch.cat((particles, tiled), dim=-1)).squeeze(-1)
        compatibility_span = float(scores.max() - scores.min())
    floor = (1.0 - model.soft_alpha) / model.num_particles
    minimum_weight = float(log_weights.exp().min())
    assert minimum_weight > floor * torch.exp(torch.tensor(-compatibility_span)).item() / 4


def test_constructor_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError):
        _make_model(num_particles=1)
    with pytest.raises(ValueError):
        _make_model(particle_dim=0)
    with pytest.raises(ValueError):
        _make_model(soft_alpha=0.0)
    with pytest.raises(ValueError):
        _make_model(soft_alpha=1.5)


def test_config_schema_validates_pf_keys() -> None:
    config = load_config("config/local.json")
    resolved = config.to_dict()
    assert resolved["comparison"]["pf_hidden"] == [64]
    assert resolved["comparison"]["pf_particle_dim"] == 8
    assert resolved["comparison"]["pf_soft_alpha"] == 0.9
    with pytest.raises(ConfigError):
        config.with_overrides({"comparison.pf_soft_alpha": 0.0})
    with pytest.raises(ConfigError):
        config.with_overrides({"comparison.pf_particle_dim": "wide"})
    accepted = config.with_overrides(
        {"comparison.methods": ["particle_filter"], "comparison.pf_hidden": [52, 124]}
    )
    assert accepted["comparison"]["methods"] == ["particle_filter"]
