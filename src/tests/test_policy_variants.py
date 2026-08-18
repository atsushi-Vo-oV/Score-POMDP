from __future__ import annotations

import pytest
import torch
from torch.distributions import Categorical, Normal

from sb_pomdp.networks import BeliefSetEncoder, DeepSetsBeliefEncoder
from sb_pomdp.policies import (
    DiffusionPolicy,
    ScoreBeliefActorCritic,
    TanhGaussianPolicy,
)


def _model_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "energy_hidden": [8],
        "score_clip": 10.0,
        "particle_clip": 10.0,
        "d_model": 8,
        "num_heads": 2,
        "num_transformer_layers": 1,
        "transformer_ff_dim": 16,
        "dropout": 0.0,
        "policy_hidden": [8],
        "diffusion_steps": 3,
        "diffusion_beta_start": 0.05,
        "diffusion_beta_end": 0.2,
        "diffusion_min_std": 0.05,
    }
    config.update(overrides)
    return config


def _continuous_actor_critic(**overrides: object) -> ScoreBeliefActorCritic:
    return ScoreBeliefActorCritic(
        observation_dim=2,
        state_dim=2,
        action_kind="continuous",
        action_feature_dim=2,
        discrete_actions=None,
        action_low=[-2.0, -1.0],
        action_high=[2.0, 3.0],
        model_config=_model_config(**overrides),
    )


def _discrete_actor_critic(**overrides: object) -> ScoreBeliefActorCritic:
    return ScoreBeliefActorCritic(
        observation_dim=2,
        state_dim=2,
        action_kind="discrete",
        action_feature_dim=1,
        discrete_actions=3,
        action_low=None,
        action_high=None,
        model_config=_model_config(**overrides),
    )


def test_deep_sets_encoder_is_permutation_invariant() -> None:
    torch.manual_seed(101)
    encoder = DeepSetsBeliefEncoder(
        state_dim=3,
        d_model=12,
        feedforward_dim=24,
        dropout=0.0,
    )
    particles = torch.randn(4, 7, 3)
    scores = torch.randn(4, 7, 3)
    permutation = torch.tensor([5, 2, 0, 6, 1, 3, 4])

    expected = encoder(particles, scores)
    actual = encoder(particles[:, permutation], scores[:, permutation])

    assert expected.shape == (4, 12)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_tanh_gaussian_replays_log_prob_from_latent_and_action() -> None:
    torch.manual_seed(103)
    policy = TanhGaussianPolicy(
        condition_dim=5,
        action_low=[-2.0, -1.0],
        action_high=[2.0, 3.0],
        hidden_dims=[12],
        log_std_min=-4.0,
        log_std_max=1.0,
    )
    condition = torch.randn(8, 5)

    action, pre_tanh_action, sampled_log_prob = policy.sample(
        condition,
        deterministic=False,
    )
    latent_log_prob, entropy = policy.log_prob(
        condition,
        pre_tanh_action=pre_tanh_action,
    )
    action_log_prob, _ = policy.log_prob(condition, action=action)

    torch.testing.assert_close(latent_log_prob, sampled_log_prob)
    torch.testing.assert_close(action_log_prob, sampled_log_prob, atol=2e-5, rtol=2e-5)
    assert entropy.shape == (8,)
    assert torch.all(action[:, 0] >= -2.0) and torch.all(action[:, 0] <= 2.0)
    assert torch.all(action[:, 1] >= -1.0) and torch.all(action[:, 1] <= 3.0)


def test_tanh_gaussian_log_prob_includes_tanh_and_affine_jacobian() -> None:
    policy = TanhGaussianPolicy(
        condition_dim=2,
        action_low=[-2.0, 1.0],
        action_high=[4.0, 5.0],
        hidden_dims=[],
    )
    with torch.no_grad():
        for parameter in policy.parameter_network.parameters():
            parameter.zero_()

    condition = torch.zeros(2, 2)
    latent = torch.tensor([[0.2, -0.3], [0.7, 0.1]])
    actual, _ = policy.log_prob(condition, pre_tanh_action=latent)
    normal_log_prob = Normal(torch.zeros_like(latent), torch.ones_like(latent)).log_prob(latent)
    jacobian = torch.log(policy.action_scale * (1.0 - torch.tanh(latent).square())).sum(dim=-1)
    expected = normal_log_prob.sum(dim=-1) - jacobian

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_tanh_gaussian_deterministic_action_is_squashed_mean() -> None:
    torch.manual_seed(107)
    policy = TanhGaussianPolicy(
        condition_dim=3,
        action_low=[-3.0],
        action_high=[1.0],
        hidden_dims=[7],
    )
    condition = torch.randn(4, 3)
    action, latent, _ = policy.sample(condition, deterministic=True)
    distribution = policy.distribution(condition)

    torch.testing.assert_close(latent, distribution.base_dist.loc)
    torch.testing.assert_close(action, policy.action_from_pre_tanh(distribution.base_dist.loc))


def test_actor_critic_defaults_remain_transformer_and_diffusion() -> None:
    model = _continuous_actor_critic()

    assert model.encoder_kind == "transformer"
    assert isinstance(model.encoder, BeliefSetEncoder)
    assert model.continuous_policy_kind == "diffusion"
    assert isinstance(model.diffusion_policy, DiffusionPolicy)
    assert model.gaussian_policy is None


def test_actor_critic_selects_deep_sets_and_gaussian() -> None:
    model = _continuous_actor_critic(
        encoder_kind="deep_sets",
        continuous_policy_kind="gaussian",
        gaussian_log_std_min=-3.0,
        gaussian_log_std_max=0.5,
    )

    assert isinstance(model.encoder, DeepSetsBeliefEncoder)
    assert model.diffusion_policy is None
    assert isinstance(model.gaussian_policy, TanhGaussianPolicy)
    assert model.gaussian_policy.log_std_min == -3.0
    assert model.gaussian_policy.log_std_max == 0.5


def test_gaussian_rollout_log_prob_equals_first_ppo_recalculation() -> None:
    torch.manual_seed(109)
    model = _continuous_actor_critic(
        encoder_kind="deep_sets",
        continuous_policy_kind="gaussian",
    )
    batch_size, num_particles = 5, 6
    observations = torch.randn(batch_size, 2)
    previous_particles = torch.randn(batch_size, num_particles, 2)
    previous_actions = torch.randn(batch_size, 2)
    initial = torch.tensor([True, False, False, True, False])
    particles = torch.randn(batch_size, num_particles, 2)

    model.eval()
    with torch.no_grad():
        rollout_scores = model.belief.score_at(
            particles,
            observations,
            previous_particles,
            previous_actions,
            initial,
            create_graph=False,
        )
        rollout_condition = model.encode(particles, rollout_scores)
        sample = model.sample_policy(rollout_condition, deterministic=False)
    assert sample.pre_tanh_action is not None
    old_log_prob = sample.log_prob.detach()

    model.train()
    update_scores = model.belief.score_at(
        particles,
        observations,
        previous_particles,
        previous_actions,
        initial,
        create_graph=True,
    )
    update_condition = model.encode(particles, update_scores)
    assert model.gaussian_policy is not None
    current_log_prob, _ = model.gaussian_policy.log_prob(
        update_condition,
        pre_tanh_action=sample.pre_tanh_action,
    )

    torch.testing.assert_close(current_log_prob, old_log_prob, atol=1e-6, rtol=1e-6)
    ratio = (current_log_prob - old_log_prob).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"encoder_kind": "attention"}, "encoder"),
        ({"continuous_policy_kind": "beta"}, "continuous policy"),
    ],
)
def test_actor_critic_rejects_unknown_variant(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _continuous_actor_critic(**override)


def test_discrete_sampling_honors_an_explicit_generator() -> None:
    """A standalone re-evaluation must reproduce its actions from its own stream."""

    torch.manual_seed(211)
    model = _discrete_actor_critic()
    condition = torch.randn(64, 8)

    torch.manual_seed(0)
    first = model.sample_policy(
        condition,
        deterministic=False,
        generator=torch.Generator().manual_seed(17),
    )
    torch.manual_seed(1)
    second = model.sample_policy(
        condition,
        deterministic=False,
        generator=torch.Generator().manual_seed(17),
    )
    other = model.sample_policy(
        condition,
        deterministic=False,
        generator=torch.Generator().manual_seed(18),
    )

    with torch.no_grad():
        probs = Categorical(logits=model.categorical_head(condition)).probs
    expected = torch.multinomial(probs, 1, generator=torch.Generator().manual_seed(17)).reshape(
        probs.shape[:-1]
    )

    torch.testing.assert_close(first.action, second.action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.action, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(first.action, other.action)
    assert first.action.shape == (64,)
    assert first.log_prob.shape == (64,)


def test_discrete_sampling_without_a_generator_keeps_the_global_rng_draw() -> None:
    """The trainer passes no generator, so its default draw sequence is unchanged."""

    torch.manual_seed(211)
    model = _discrete_actor_critic()
    condition = torch.randn(64, 8)
    with torch.no_grad():
        distribution = Categorical(logits=model.categorical_head(condition))

    torch.manual_seed(23)
    expected = [distribution.sample() for _ in range(3)]
    torch.manual_seed(23)
    actual = [
        model.sample_policy(condition, deterministic=False).action for _ in range(3)
    ]

    for expected_action, actual_action in zip(expected, actual):
        torch.testing.assert_close(actual_action, expected_action, rtol=0.0, atol=0.0)


def test_discrete_deterministic_readout_ignores_the_generator() -> None:
    torch.manual_seed(211)
    model = _discrete_actor_critic()
    condition = torch.randn(8, 8)

    greedy = model.sample_policy(condition, deterministic=True)
    seeded_greedy = model.sample_policy(
        condition,
        deterministic=True,
        generator=torch.Generator().manual_seed(17),
    )
    with torch.no_grad():
        expected = Categorical(logits=model.categorical_head(condition)).probs.argmax(dim=-1)

    torch.testing.assert_close(greedy.action, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(seeded_greedy.action, expected, rtol=0.0, atol=0.0)


def test_tanh_gaussian_requires_exactly_one_replay_representation() -> None:
    policy = TanhGaussianPolicy(2, [-1.0], [1.0], [4])
    condition = torch.zeros(1, 2)
    action = torch.zeros(1, 1)

    with pytest.raises(ValueError, match="exactly one"):
        policy.log_prob(condition)
    with pytest.raises(ValueError, match="exactly one"):
        policy.log_prob(condition, action=action, pre_tanh_action=action)
    with pytest.raises(ValueError, match="outside"):
        policy.log_prob(condition, action=torch.tensor([[1.1]]))
