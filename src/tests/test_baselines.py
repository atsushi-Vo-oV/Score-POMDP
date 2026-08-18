from __future__ import annotations

import pytest
import torch

from sb_pomdp.baselines import (
    GRUActorCritic,
    ObservationActorCritic,
    OracleStateActorCritic,
    SquashedDiagonalGaussian,
    make_baseline_actor_critic,
)
from sb_pomdp.policies import ScoreBeliefActorCritic


def test_observation_categorical_policy_replays_rollout_likelihood() -> None:
    torch.manual_seed(4)
    model = ObservationActorCritic(
        3,
        action_kind="discrete",
        hidden_dims=[8, 6],
        discrete_actions=4,
    )
    observations = torch.randn(7, 3)
    generator = torch.Generator().manual_seed(12)

    sample, values = model.act(
        observations,
        deterministic=False,
        generator=generator,
    )
    replay = model.evaluate_actions(observations, sample.action)

    assert sample.action.shape == (7,)
    assert values.shape == (7,)
    assert torch.equal(sample.log_prob, replay.log_prob)
    assert torch.equal(values, replay.value)
    assert torch.isfinite(replay.entropy).all()
    assert model.input_source == "observation"


def test_oracle_continuous_policy_is_bounded_and_replays_likelihood() -> None:
    torch.manual_seed(5)
    model = OracleStateActorCritic(
        4,
        action_kind="continuous",
        hidden_dims=[12],
        action_low=[-2.0, 1.0],
        action_high=[0.5, 3.0],
    )
    states = torch.randn(9, 4)
    sample, values = model.act(states, deterministic=False)
    assert sample.pre_tanh_action is not None
    replay = model.evaluate_actions(
        states,
        sample.action,
        pre_tanh_actions=sample.pre_tanh_action,
    )

    assert sample.action.shape == (9, 2)
    assert values.shape == (9,)
    assert torch.all(sample.action[:, 0] >= -2.0)
    assert torch.all(sample.action[:, 0] <= 0.5)
    assert torch.all(sample.action[:, 1] >= 1.0)
    assert torch.all(sample.action[:, 1] <= 3.0)
    torch.testing.assert_close(sample.log_prob, replay.log_prob, rtol=0.0, atol=0.0)
    loss = -(replay.log_prob.mean() + 0.1 * replay.entropy.mean()) + replay.value.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.input_source == "oracle_state"


def test_squashed_gaussian_deterministic_action_uses_transformed_mean() -> None:
    policy = SquashedDiagonalGaussian(3, [-3.0], [1.0])
    with torch.no_grad():
        policy.mean_head.weight.zero_()
        policy.mean_head.bias.fill_(torch.atanh(torch.tensor(0.5)))
    features = torch.zeros(2, 3)

    sample = policy.sample(features, deterministic=True)

    torch.testing.assert_close(sample.action, torch.zeros(2, 1), atol=1e-6, rtol=0.0)
    assert torch.isfinite(sample.log_prob).all()


def test_gru_step_matches_sequence_path_without_resets() -> None:
    torch.manual_seed(6)
    model = GRUActorCritic(
        2,
        action_kind="discrete",
        hidden_dims=[7],
        recurrent_hidden_dim=5,
        recurrent_layers=2,
        discrete_actions=3,
    )
    observations = torch.randn(4, 3, 2)
    initial_hidden = model.initial_hidden(3)
    episode_starts = torch.zeros(4, 3, dtype=torch.bool)

    sequence_features, sequence_hidden = model.encode_sequence(
        observations,
        initial_hidden,
        episode_starts,
    )
    hidden = initial_hidden
    step_features = []
    for time_index in range(observations.shape[0]):
        features, hidden = model.step(
            observations[time_index],
            hidden,
            episode_starts[time_index],
        )
        step_features.append(features)

    torch.testing.assert_close(sequence_features, torch.stack(step_features))
    torch.testing.assert_close(sequence_hidden, hidden)


def test_gru_episode_start_cuts_history_and_preserves_later_bptt() -> None:
    torch.manual_seed(7)
    model = GRUActorCritic(
        2,
        action_kind="discrete",
        hidden_dims=[6],
        recurrent_hidden_dim=5,
        discrete_actions=2,
    )
    observations = torch.randn(5, 1, 2, requires_grad=True)
    episode_starts = torch.tensor([[True], [False], [False], [True], [False]])
    hidden = model.initial_hidden(1)

    features, _ = model.encode_sequence(observations, hidden, episode_starts)
    features[-1].sum().backward()

    assert observations.grad is not None
    assert torch.count_nonzero(observations.grad[:3]) == 0
    assert torch.count_nonzero(observations.grad[3:]) > 0


def test_gru_full_and_one_step_modes_match_forward_but_cut_different_gradients() -> None:
    torch.manual_seed(17)
    model = GRUActorCritic(
        2,
        action_kind="discrete",
        hidden_dims=[6],
        recurrent_hidden_dim=5,
        discrete_actions=2,
    )
    observations = torch.randn(4, 1, 2)
    episode_starts = torch.tensor([[True], [False], [False], [False]])

    full_inputs = observations.clone().requires_grad_()
    full_features, _ = model.encode_sequence(
        full_inputs,
        model.initial_hidden(1),
        episode_starts,
        temporal_gradient_mode="full",
    )
    full_features[-1].sum().backward()
    assert full_inputs.grad is not None
    assert torch.count_nonzero(full_inputs.grad[:-1]) > 0

    one_step_inputs = observations.clone().requires_grad_()
    one_step_features, _ = model.encode_sequence(
        one_step_inputs,
        model.initial_hidden(1),
        episode_starts,
        temporal_gradient_mode="tbptt_1",
    )
    one_step_features[-1].sum().backward()
    assert one_step_inputs.grad is not None
    torch.testing.assert_close(one_step_inputs.grad[:-1], torch.zeros_like(one_step_inputs[:-1]))
    assert torch.count_nonzero(one_step_inputs.grad[-1]) > 0
    torch.testing.assert_close(one_step_features, full_features)


def test_gru_diffusion_policy_replays_the_saved_reverse_chain() -> None:
    torch.manual_seed(23)
    model = GRUActorCritic(
        3,
        action_kind="continuous",
        hidden_dims=[7, 5],
        recurrent_hidden_dim=6,
        policy_condition_dim=8,
        head_hidden_dims=[9],
        continuous_policy_kind="diffusion",
        action_low=[-2.0, -1.0],
        action_high=[1.0, 3.0],
        diffusion_steps=3,
        diffusion_beta_start=0.01,
        diffusion_beta_end=0.1,
        diffusion_min_std=0.05,
    )
    observations = torch.randn(4, 2, 3)
    starts = torch.zeros(4, 2, dtype=torch.bool)
    starts[0] = True
    hidden = model.initial_hidden(2)
    features, _ = model.encode_sequence(observations, hidden, starts)
    sample = model.heads.sample_policy(features, deterministic=False)

    assert sample.diffusion_chain is not None
    assert sample.pre_tanh_action is None
    assert sample.action.shape == (4, 2, 2)
    assert sample.log_prob.shape == (4, 2, 3)
    assert torch.all(sample.action >= torch.tensor([-2.0, -1.0]))
    assert torch.all(sample.action <= torch.tensor([1.0, 3.0]))

    replay = model.evaluate_sequence(
        observations,
        sample.action,
        hidden,
        starts,
        diffusion_chains=sample.diffusion_chain,
    )
    torch.testing.assert_close(sample.log_prob, replay.log_prob, rtol=0.0, atol=0.0)
    assert replay.entropy.shape == (4, 2, 3)
    loss = -replay.log_prob.mean() + replay.value.square().mean()
    loss.backward()
    assert model.gru.weight_hh_l0.grad is not None
    assert model.heads.diffusion_policy is not None
    assert next(model.heads.diffusion_policy.parameters()).grad is not None


def test_score_and_gru_continuous_heads_have_identical_diffusion_behavior() -> None:
    """Guard the policy-head part of the scientific comparison against drift."""

    model_config = {
        "num_particles": 3,
        "langevin_steps": 1,
        "langevin_step_size": 0.02,
        "particle_clip": 0.0,
        "score_clip": 0.0,
        "energy_hidden": [5],
        "d_model": 8,
        "num_heads": 2,
        "num_transformer_layers": 1,
        "transformer_ff_dim": 12,
        "dropout": 0.0,
        "policy_hidden": [9],
        "diffusion_steps": 3,
        "diffusion_beta_start": 0.01,
        "diffusion_beta_end": 0.1,
        "diffusion_min_std": 0.05,
        "diffusion_advantage_discount": 0.95,
        "encoder_kind": "transformer",
        "continuous_policy_kind": "diffusion",
        "belief_gradient_mode": "full",
    }
    score = ScoreBeliefActorCritic(
        observation_dim=2,
        state_dim=3,
        action_kind="continuous",
        action_feature_dim=2,
        discrete_actions=None,
        action_low=[-2.0, -1.0],
        action_high=[1.0, 3.0],
        model_config=model_config,
    )
    gru = GRUActorCritic(
        4,
        action_kind="continuous",
        hidden_dims=[7],
        recurrent_hidden_dim=6,
        policy_condition_dim=8,
        head_hidden_dims=[9],
        continuous_policy_kind="diffusion",
        action_low=[-2.0, -1.0],
        action_high=[1.0, 3.0],
        diffusion_steps=3,
        diffusion_beta_start=0.01,
        diffusion_beta_end=0.1,
        diffusion_min_std=0.05,
    )
    assert score.diffusion_policy is not None
    assert gru.heads.diffusion_policy is not None
    gru.heads.diffusion_policy.load_state_dict(score.diffusion_policy.state_dict())
    gru.heads.value_head.load_state_dict(score.value_head.state_dict())

    condition = torch.randn(5, 8)
    score_sample = score.sample_policy(
        condition,
        deterministic=False,
        generator=torch.Generator().manual_seed(31),
    )
    gru_sample = gru.heads.sample_policy(
        condition,
        deterministic=False,
        generator=torch.Generator().manual_seed(31),
    )
    assert score_sample.diffusion_chain is not None
    assert gru_sample.diffusion_chain is not None
    torch.testing.assert_close(score_sample.action, gru_sample.action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        score_sample.diffusion_chain,
        gru_sample.diffusion_chain,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(score_sample.log_prob, gru_sample.log_prob, rtol=0.0, atol=0.0)
    torch.testing.assert_close(score_sample.entropy, gru_sample.entropy, rtol=0.0, atol=0.0)

    replay = gru.heads.evaluate_actions(
        condition,
        score_sample.action,
        diffusion_chains=score_sample.diffusion_chain,
    )
    torch.testing.assert_close(score_sample.log_prob, replay.log_prob, rtol=0.0, atol=0.0)
    torch.testing.assert_close(score.value(condition), replay.value, rtol=0.0, atol=0.0)


def test_score_and_gru_discrete_heads_sample_identically_from_one_generator() -> None:
    """Both arms must draw the same categorical action from the same stream."""

    torch.manual_seed(41)
    score = ScoreBeliefActorCritic(
        observation_dim=2,
        state_dim=3,
        action_kind="discrete",
        action_feature_dim=1,
        discrete_actions=3,
        action_low=None,
        action_high=None,
        model_config={
            "num_particles": 3,
            "langevin_steps": 1,
            "langevin_step_size": 0.02,
            "particle_clip": 0.0,
            "score_clip": 0.0,
            "energy_hidden": [5],
            "d_model": 8,
            "num_heads": 2,
            "num_transformer_layers": 1,
            "transformer_ff_dim": 12,
            "dropout": 0.0,
            "policy_hidden": [9],
            "diffusion_steps": 3,
            "diffusion_beta_start": 0.01,
            "diffusion_beta_end": 0.1,
            "diffusion_min_std": 0.05,
            "encoder_kind": "transformer",
            "continuous_policy_kind": "diffusion",
            "belief_gradient_mode": "full",
        },
    )
    gru = GRUActorCritic(
        4,
        action_kind="discrete",
        hidden_dims=[7],
        recurrent_hidden_dim=6,
        policy_condition_dim=8,
        head_hidden_dims=[9],
        discrete_actions=3,
    )
    assert score.categorical_head is not None
    assert gru.heads.categorical_head is not None
    gru.heads.categorical_head.load_state_dict(score.categorical_head.state_dict())

    condition = torch.randn(64, 8)
    score_sample = score.sample_policy(
        condition,
        deterministic=False,
        generator=torch.Generator().manual_seed(31),
    )
    gru_sample = gru.heads.sample_policy(
        condition,
        deterministic=False,
        generator=torch.Generator().manual_seed(31),
    )

    torch.testing.assert_close(score_sample.action, gru_sample.action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(score_sample.log_prob, gru_sample.log_prob, rtol=0.0, atol=0.0)
    torch.testing.assert_close(score_sample.entropy, gru_sample.entropy, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("action_kind", ["discrete", "continuous"])
def test_gru_sequence_evaluation_shapes_and_gradients(action_kind: str) -> None:
    torch.manual_seed(8)
    kwargs: dict[str, object]
    if action_kind == "discrete":
        kwargs = {"discrete_actions": 3}
    else:
        kwargs = {"action_low": [-1.0], "action_high": [2.0]}
    model = GRUActorCritic(
        2,
        action_kind=action_kind,
        hidden_dims=[8],
        recurrent_hidden_dim=6,
        **kwargs,
    )
    observations = torch.randn(4, 3, 2)
    starts = torch.zeros(4, 3, dtype=torch.bool)
    starts[0] = True
    hidden = model.initial_hidden(3)
    features, _ = model.encode_sequence(observations, hidden, starts)
    sample = model.heads.sample_policy(features, deterministic=False)

    evaluation = model.evaluate_sequence(observations, sample.action, hidden, starts)

    assert evaluation.log_prob.shape == (4, 3)
    assert evaluation.entropy.shape == (4, 3)
    assert evaluation.value.shape == (4, 3)
    assert evaluation.final_hidden.shape == (1, 3, 6)
    torch.testing.assert_close(sample.log_prob, evaluation.log_prob)
    loss = -evaluation.log_prob.mean() + evaluation.value.square().mean()
    loss.backward()
    assert model.gru.weight_hh_l0.grad is not None
    assert torch.isfinite(model.gru.weight_hh_l0.grad).all()


@pytest.mark.parametrize(
    ("kind", "expected_type", "input_source"),
    [
        ("observation", ObservationActorCritic, "observation"),
        ("oracle_state", OracleStateActorCritic, "oracle_state"),
        ("gru", GRUActorCritic, "observation"),
    ],
)
def test_baseline_factory(kind: str, expected_type: type, input_source: str) -> None:
    model = make_baseline_actor_critic(
        kind,
        observation_dim=2,
        state_dim=4,
        action_kind="discrete",
        hidden_dims=[8],
        discrete_actions=2,
        recurrent_hidden_dim=7,
    )

    assert isinstance(model, expected_type)
    assert model.input_source == input_source


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SquashedDiagonalGaussian(3, [-1.0], [float("inf")]),
        lambda: ObservationActorCritic(
            2,
            action_kind="discrete",
            hidden_dims=[],
            discrete_actions=2,
        ),
        lambda: make_baseline_actor_critic(
            "unknown",  # type: ignore[arg-type]
            observation_dim=2,
            state_dim=4,
            action_kind="discrete",
            hidden_dims=[8],
            discrete_actions=2,
        ),
    ],
)
def test_invalid_baseline_configuration_fails_fast(factory) -> None:
    with pytest.raises(ValueError):
        factory()
