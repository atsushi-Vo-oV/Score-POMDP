"""Invariants of the alpha-vector-style pooled encoder and log-sum-exp heads."""

from __future__ import annotations

import pytest
import torch

from sb_pomdp.compare import _method_config
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.networks import AlphaLSEHead, AlphaPoolBeliefEncoder
from sb_pomdp.policies import ScoreBeliefActorCritic


def _model_config(**overrides: object) -> dict:
    config: dict = {
        "num_particles": 4,
        "langevin_steps": 3,
        "langevin_step_size": 0.05,
        "particle_clip": 0.0,
        "score_clip": 0.0,
        "energy_hidden": [8],
        "d_model": 8,
        "num_heads": 2,
        "num_transformer_layers": 1,
        "transformer_ff_dim": 16,
        "dropout": 0.0,
        "policy_hidden": [8],
        "encoder_kind": "alpha_pool",
        "policy_head_kind": "alpha_lse",
        "alpha_pieces": 5,
        "alpha_temperature": 0.5,
        "alpha_use_scores": False,
    }
    config.update(overrides)
    return config


def test_alpha_pool_encoder_is_permutation_invariant_and_ignores_scores() -> None:
    torch.manual_seed(0)
    encoder = AlphaPoolBeliefEncoder(3, 8, 16, use_scores=False)
    particles = torch.randn(2, 6, 3)
    scores = torch.randn(2, 6, 3)
    baseline = encoder(particles, scores)
    permutation = torch.randperm(6)
    permuted = encoder(particles[:, permutation], scores[:, permutation])
    assert torch.allclose(baseline, permuted, atol=1e-6)
    assert torch.allclose(baseline, encoder(particles, torch.randn(2, 6, 3)), atol=1e-6)

    with_scores = AlphaPoolBeliefEncoder(3, 8, 16, use_scores=True)
    a = with_scores(particles, scores)
    b = with_scores(particles, torch.randn(2, 6, 3))
    assert not torch.allclose(a, b)


def test_alpha_pool_encoder_is_linear_in_the_belief() -> None:
    # Mean pooling with no post-pooling network: the encoding of a union of two
    # equally sized clouds is the average of their encodings.
    torch.manual_seed(1)
    encoder = AlphaPoolBeliefEncoder(3, 8, 16, use_scores=False)
    cloud_a = torch.randn(1, 5, 3)
    cloud_b = torch.randn(1, 5, 3)
    scores = torch.zeros(1, 5, 3)
    union = encoder(
        torch.cat((cloud_a, cloud_b), dim=1), torch.cat((scores, scores), dim=1)
    )
    averaged = 0.5 * (encoder(cloud_a, scores) + encoder(cloud_b, scores))
    assert torch.allclose(union, averaged, atol=1e-6)


def test_alpha_lse_head_approaches_the_hard_maximum_as_temperature_vanishes() -> None:
    torch.manual_seed(2)
    condition = torch.randn(4, 8)
    cold = AlphaLSEHead(8, 3, 6, 1e-4)
    with torch.no_grad():
        values = cold.linear(condition).view(4, 3, 6)
        hard_max = values.max(dim=-1).values
        smooth = cold(condition)
    assert torch.allclose(smooth, hard_max, atol=1e-3)
    assert smooth.shape == (4, 3)

    warm = AlphaLSEHead(8, 1, 6, 1.0)
    assert warm(condition).shape == (4, 1)
    with pytest.raises(ValueError):
        AlphaLSEHead(8, 3, 6, 0.0)
    with pytest.raises(ValueError):
        AlphaLSEHead(8, 3, 0, 1.0)


def test_actor_critic_builds_the_alpha_variant_and_produces_finite_outputs() -> None:
    torch.manual_seed(3)
    model = ScoreBeliefActorCritic(
        observation_dim=3,
        state_dim=2,
        action_kind="discrete",
        action_feature_dim=2,
        discrete_actions=2,
        action_low=None,
        action_high=None,
        model_config=_model_config(),
    )
    assert isinstance(model.encoder, AlphaPoolBeliefEncoder)
    assert isinstance(model.categorical_head, AlphaLSEHead)
    assert isinstance(model.value_head, AlphaLSEHead)

    particles = torch.randn(2, 4, 2)
    scores = torch.randn(2, 4, 2)
    condition = model.encode(particles, scores)
    sample = model.sample_policy(condition, deterministic=True)
    value = model.value(condition)
    assert sample.action.shape == (2,)
    assert value.shape == (2,)
    assert torch.isfinite(sample.log_prob).all()
    assert torch.isfinite(value).all()

    with pytest.raises(ValueError, match="policy head"):
        ScoreBeliefActorCritic(
            observation_dim=3,
            state_dim=2,
            action_kind="discrete",
            action_feature_dim=2,
            discrete_actions=2,
            action_low=None,
            action_high=None,
            model_config=_model_config(policy_head_kind="maxout"),
        )


def test_continuous_alpha_variant_keeps_the_diffusion_actor() -> None:
    torch.manual_seed(4)
    model = ScoreBeliefActorCritic(
        observation_dim=3,
        state_dim=2,
        action_kind="continuous",
        action_feature_dim=1,
        discrete_actions=None,
        action_low=[-1.0],
        action_high=[1.0],
        model_config=_model_config(
            diffusion_steps=2,
            diffusion_beta_start=0.01,
            diffusion_beta_end=0.2,
            diffusion_min_std=0.05,
            continuous_policy_kind="diffusion",
        ),
    )
    assert isinstance(model.value_head, AlphaLSEHead)
    assert model.diffusion_policy is not None


def test_method_config_and_schema_accept_score_alpha() -> None:
    config = load_config("config/local.json")
    resolved = config.to_dict()
    assert resolved["model"]["policy_head_kind"] == "mlp"
    assert resolved["model"]["alpha_pieces"] == 16
    assert resolved["model"]["alpha_temperature"] == 1.0
    assert resolved["model"]["alpha_use_scores"] is False

    shard = _method_config(config, "score_alpha", "masked_cartpole", 3, "full")
    assert shard["model"]["encoder_kind"] == "alpha_pool"
    assert shard["model"]["policy_head_kind"] == "alpha_lse"
    assert shard["model"]["continuous_policy_kind"] == "diffusion"
    assert shard["comparison"]["methods"] == ["score_alpha"]

    with pytest.raises(ConfigError):
        config.with_overrides({"model.policy_head_kind": "maxout"})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.alpha_temperature": 0.0})
    with pytest.raises(ConfigError):
        config.with_overrides({"model.alpha_use_scores": "yes"})
    accepted = config.with_overrides(
        {
            "model.encoder_kind": "alpha_pool",
            "model.policy_head_kind": "alpha_lse",
            "model.alpha_pieces": 8,
            "model.alpha_temperature": 0.25,
        }
    )
    assert accepted["model"]["alpha_pieces"] == 8
