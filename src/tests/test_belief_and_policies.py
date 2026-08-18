from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch.distributions import Categorical

from sb_pomdp.belief import EnergyBelief, PotentialBranch, score_from_potential
from sb_pomdp.networks import BeliefSetEncoder
from sb_pomdp.policies import DiffusionPolicy, ScoreBeliefActorCritic
from sb_pomdp.ppo import _replay_belief_step


def _dense_reference_potential(
    belief: EnergyBelief,
    query: torch.Tensor,
    observation: torch.Tensor,
    previous_particles: torch.Tensor,
    previous_action: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """The pre-fast-path expression that materialises all three networks."""

    num_query = query.shape[1]
    num_previous = previous_particles.shape[1]
    obs = observation[:, None, :].expand(-1, num_query, -1)
    action = previous_action[:, None, :].expand(-1, num_query, -1)
    initial_value = belief.initial_energy(torch.cat((obs, query), dim=-1)).squeeze(-1)
    f_value = belief.observation_energy(
        torch.cat((obs, query, action), dim=-1)
    ).squeeze(-1)
    query_pairs = query[:, :, None, :].expand(-1, -1, num_previous, -1)
    previous_pairs = previous_particles[:, None, :, :].expand(
        -1, num_query, -1, -1
    )
    action_pairs = previous_action[:, None, None, :].expand(
        -1, num_query, num_previous, -1
    )
    u_value = belief.transition_energy(
        torch.cat((query_pairs, previous_pairs, action_pairs), dim=-1)
    ).squeeze(-1)
    transition_term = torch.logsumexp(u_value, dim=-1) - math.log(num_previous)
    initial_potential = initial_value - 0.5 * query.square().sum(dim=-1)
    return torch.where(
        initial[:, None].bool(),
        initial_potential,
        f_value + transition_term,
    )


@pytest.mark.parametrize(
    "initial",
    [
        torch.tensor([True, True, True, True]),
        torch.tensor([False, False, False, False]),
        torch.tensor([True, False, False, True]),
    ],
    ids=["all-initial", "all-recursive", "mixed"],
)
def test_energy_fast_path_matches_dense_forward_score_and_gradients(
    initial: torch.Tensor,
) -> None:
    torch.manual_seed(101)
    actual_belief = EnergyBelief(2, 2, 1, [7, 5], score_clip=0.0).double()
    reference_belief = copy.deepcopy(actual_belief)

    base_inputs = (
        torch.randn(4, 3, 2, dtype=torch.float64),
        torch.randn(4, 2, dtype=torch.float64),
        torch.randn(4, 5, 2, dtype=torch.float64),
        torch.randn(4, 1, dtype=torch.float64),
    )
    actual_inputs = tuple(value.clone().requires_grad_(True) for value in base_inputs)
    reference_inputs = tuple(value.clone().requires_grad_(True) for value in base_inputs)

    actual_potential = actual_belief.log_potential(
        *actual_inputs,
        initial,
    )
    reference_potential = _dense_reference_potential(
        reference_belief,
        *reference_inputs,
        initial,
    )
    torch.testing.assert_close(actual_potential, reference_potential)

    (actual_score,) = torch.autograd.grad(
        actual_potential.sum(),
        actual_inputs[0],
        create_graph=True,
        retain_graph=True,
    )
    (reference_score,) = torch.autograd.grad(
        reference_potential.sum(),
        reference_inputs[0],
        create_graph=True,
        retain_graph=True,
    )
    torch.testing.assert_close(actual_score, reference_score)
    (actual_potential.square().mean() + actual_score.square().mean()).backward()
    (reference_potential.square().mean() + reference_score.square().mean()).backward()

    for actual_input, reference_input in zip(
        actual_inputs,
        reference_inputs,
        strict=True,
    ):
        actual_gradient = (
            torch.zeros_like(actual_input)
            if actual_input.grad is None
            else actual_input.grad
        )
        reference_gradient = (
            torch.zeros_like(reference_input)
            if reference_input.grad is None
            else reference_input.grad
        )
        torch.testing.assert_close(actual_gradient, reference_gradient)
    for (actual_name, actual_parameter), (reference_name, reference_parameter) in zip(
        actual_belief.named_parameters(),
        reference_belief.named_parameters(),
        strict=True,
    ):
        assert actual_name == reference_name
        actual_gradient = (
            torch.zeros_like(actual_parameter)
            if actual_parameter.grad is None
            else actual_parameter.grad
        )
        reference_gradient = (
            torch.zeros_like(reference_parameter)
            if reference_parameter.grad is None
            else reference_parameter.grad
        )
        torch.testing.assert_close(actual_gradient, reference_gradient)


@pytest.mark.parametrize(
    ("initial", "expected_calls"),
    [
        (torch.ones(3, dtype=torch.bool), (1, 0, 0)),
        (torch.zeros(3, dtype=torch.bool), (0, 1, 1)),
        (torch.tensor([True, False, True]), (1, 1, 1)),
    ],
    ids=["all-initial", "all-recursive", "mixed"],
)
def test_energy_fast_path_only_executes_required_networks(
    initial: torch.Tensor,
    expected_calls: tuple[int, int, int],
) -> None:
    belief = EnergyBelief(2, 2, 1, [8])
    calls = [0, 0, 0]
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, module in enumerate(
            (belief.initial_energy, belief.observation_energy, belief.transition_energy)
        )
    ]
    try:
        belief.log_potential(
            torch.randn(3, 4, 2),
            torch.randn(3, 2),
            torch.randn(3, 5, 2),
            torch.randn(3, 1),
            initial,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert tuple(calls) == expected_calls


@pytest.mark.parametrize(
    ("initial", "expected_calls"),
    [
        (torch.ones(3, dtype=torch.bool), (9, 0, 0)),
        (torch.zeros(3, dtype=torch.bool), (0, 9, 9)),
        (torch.tensor([True, False, True]), (9, 9, 9)),
    ],
    ids=["all-initial", "all-recursive", "mixed"],
)
def test_particle_replay_reuses_one_branch_for_all_nine_score_calls(
    initial: torch.Tensor,
    expected_calls: tuple[int, int, int],
) -> None:
    torch.manual_seed(102)
    belief = EnergyBelief(2, 2, 1, [8], score_clip=0.0)
    calls = [0, 0, 0]
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, module in enumerate(
            (belief.initial_energy, belief.observation_energy, belief.transition_energy)
        )
    ]
    try:
        belief.particles_from_noise(
            torch.randn(3, 2),
            torch.randn(3, 4, 2),
            torch.randn(3, 1),
            initial,
            initial_noise=torch.randn(3, 4, 2),
            langevin_noise=torch.randn(3, 8, 4, 2),
            step_size=0.02,
            track_grad=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert tuple(calls) == expected_calls


@pytest.mark.parametrize(
    ("initial_value", "inactive_networks"),
    [(True, (1, 2)), (False, (0,))],
    ids=["all-initial", "all-recursive"],
)
def test_checkpoint_replay_classifies_once_and_never_runs_inactive_networks(
    monkeypatch: pytest.MonkeyPatch,
    initial_value: bool,
    inactive_networks: tuple[int, ...],
) -> None:
    torch.manual_seed(103)
    belief = EnergyBelief(2, 2, 1, [8], score_clip=0.0)
    classifications = 0
    network_calls = [0, 0, 0]
    original_classifier = EnergyBelief.classify_potential_branch

    def counted_classifier(
        self: EnergyBelief,
        initial: torch.Tensor,
    ) -> PotentialBranch:
        nonlocal classifications
        classifications += 1
        return original_classifier(self, initial)

    monkeypatch.setattr(EnergyBelief, "classify_potential_branch", counted_classifier)
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, index=index: network_calls.__setitem__(
                index, network_calls[index] + 1
            )
        )
        for index, module in enumerate(
            (belief.initial_energy, belief.observation_energy, belief.transition_energy)
        )
    ]
    try:
        particles, scores = _replay_belief_step(
            SimpleNamespace(belief=belief),  # type: ignore[arg-type]
            torch.randn(2, 2),
            torch.randn(2, 3, 2),
            torch.randn(2, 1),
            torch.full((2,), initial_value, dtype=torch.bool),
            torch.randn(2, 3, 2),
            torch.randn(2, 8, 3, 2),
            step_size=0.05,
            gradient_mode="full",
            track_grad=True,
        )
        (particles.square().mean() + scores.square().mean()).backward()
    finally:
        for handle in handles:
            handle.remove()

    assert classifications == 1
    assert all(network_calls[index] == 0 for index in inactive_networks)
    assert all(
        call_count > 0
        for index, call_count in enumerate(network_calls)
        if index not in inactive_networks
    )


def test_score_from_quadratic_potential_matches_analytic_gradient() -> None:
    query = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]], requires_grad=True)
    potential = -0.5 * query.square().sum(dim=-1)
    score = score_from_potential(potential, query, create_graph=False)
    torch.testing.assert_close(score, -query)


def test_recursive_score_is_invariant_to_previous_particle_order() -> None:
    torch.manual_seed(3)
    belief = EnergyBelief(2, 2, 1, [8], score_clip=100.0)
    query = torch.randn(2, 4, 2)
    observation = torch.randn(2, 2)
    previous = torch.randn(2, 5, 2)
    action = torch.randn(2, 1)
    initial = torch.zeros(2, dtype=torch.bool)

    first = belief.score_at(query, observation, previous, action, initial, create_graph=False)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    second = belief.score_at(
        query, observation, previous[:, permutation], action, initial, create_graph=False
    )
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)


def test_initial_potential_uses_a_dedicated_network_and_standard_normal_base() -> None:
    torch.manual_seed(5)
    belief = EnergyBelief(2, 2, 1, [8], score_clip=100.0)
    query = torch.randn(3, 4, 2)
    observation = torch.randn(3, 2)
    previous = torch.randn(3, 5, 2)
    action = torch.randn(3, 1)
    initial = torch.ones(3, dtype=torch.bool)

    potential = belief.log_potential(query, observation, previous, action, initial)
    expanded_observation = observation[:, None, :].expand(-1, query.shape[1], -1)
    expected = belief.initial_energy(
        torch.cat((expanded_observation, query), dim=-1)
    ).squeeze(-1) - 0.5 * query.square().sum(dim=-1)
    torch.testing.assert_close(potential, expected)

    score = belief.score_at(
        query,
        observation,
        previous,
        action,
        initial,
        create_graph=True,
    )
    score.square().mean().backward()
    initial_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in belief.initial_energy.parameters()
        if parameter.grad is not None
    )
    recursive_gradient = sum(
        float(parameter.grad.abs().sum())
        for network in (belief.observation_energy, belief.transition_energy)
        for parameter in network.parameters()
        if parameter.grad is not None
    )
    assert initial_gradient > 0
    assert recursive_gradient == 0


def test_score_loss_reaches_both_energy_networks() -> None:
    torch.manual_seed(9)
    belief = EnergyBelief(2, 2, 1, [8, 8], score_clip=100.0)
    particles = torch.randn(3, 4, 2)
    observation = torch.randn(3, 2)
    previous = torch.randn(3, 4, 2)
    action = torch.randn(3, 1)
    initial = torch.zeros(3, dtype=torch.bool)
    score = belief.score_at(particles, observation, previous, action, initial, create_graph=True)
    score.square().mean().backward()
    f_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in belief.observation_energy.parameters()
        if parameter.grad is not None
    )
    u_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in belief.transition_energy.parameters()
        if parameter.grad is not None
    )
    assert f_gradient > 0
    assert u_gradient > 0


def test_set_encoder_is_permutation_invariant_in_eval_mode() -> None:
    torch.manual_seed(7)
    encoder = BeliefSetEncoder(
        3, d_model=12, num_heads=3, num_layers=2, feedforward_dim=24, dropout=0.0
    )
    encoder.eval()
    particles = torch.randn(2, 6, 3)
    scores = torch.randn(2, 6, 3)
    expected = encoder(particles, scores)
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    actual = encoder(particles[:, permutation], scores[:, permutation])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_diffusion_chain_log_prob_replays_and_action_is_bounded() -> None:
    torch.manual_seed(11)
    policy = DiffusionPolicy(
        condition_dim=8,
        action_low=[-2.0, -1.0],
        action_high=[2.0, 3.0],
        hidden_dims=[16],
        num_steps=3,
        beta_start=0.05,
        beta_end=0.2,
        min_std=0.05,
    )
    condition = torch.randn(5, 8)
    action, chain, sampled_log_prob = policy.sample(condition, deterministic=False)
    replayed_log_prob, entropy = policy.log_prob_chain(condition, chain)
    torch.testing.assert_close(sampled_log_prob, replayed_log_prob)
    assert chain.shape == (5, 4, 2)
    assert entropy.shape == (5, 3)
    assert torch.all(action[:, 0] >= -2.0) and torch.all(action[:, 0] <= 2.0)
    assert torch.all(action[:, 1] >= -1.0) and torch.all(action[:, 1] <= 3.0)


def test_diffusion_chain_log_prob_matches_independent_gaussian_formula() -> None:
    policy = DiffusionPolicy(
        condition_dim=2,
        action_low=[-2.0, -1.0],
        action_high=[2.0, 3.0],
        hidden_dims=[8],
        num_steps=3,
        beta_start=0.05,
        beta_end=0.2,
        min_std=0.05,
    )
    with torch.no_grad():
        for parameter in policy.denoiser.parameters():
            parameter.zero_()

    condition = torch.tensor([[0.3, -0.2], [0.1, 0.4]])
    chain = torch.tensor(
        [
            [[0.7, -0.1], [0.2, 0.4], [-0.3, 0.8], [0.5, -0.6]],
            [[-0.2, 0.9], [0.7, -0.4], [0.1, 0.2], [-0.8, 0.3]],
        ]
    )

    actual, _ = policy.log_prob_chain(condition, chain)
    betas = torch.linspace(0.05, 0.2, 3)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    previous_alpha_bars = torch.cat((torch.ones(1), alpha_bars[:-1]))
    posterior_variance = betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
    expected_steps: list[torch.Tensor] = []
    for chain_index, step_index in enumerate(reversed(range(policy.num_steps))):
        mean = chain[:, chain_index] / torch.sqrt(alphas[step_index])
        std = torch.sqrt(posterior_variance[step_index].clamp_min(0.05**2))
        standardized = (chain[:, chain_index + 1] - mean) / std
        expected_steps.append(
            -0.5
            * (standardized.square() + 2.0 * torch.log(std) + math.log(2.0 * math.pi)).sum(dim=-1)
        )
    expected = torch.stack(expected_steps, dim=1)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("action_kind", ["discrete", "continuous"])
def test_rollout_log_prob_equals_first_ppo_recalculation(action_kind: str) -> None:
    torch.manual_seed(23)
    is_discrete = action_kind == "discrete"
    action_feature_dim = 2 if is_discrete else 1
    model = ScoreBeliefActorCritic(
        observation_dim=2,
        state_dim=2,
        action_kind=action_kind,
        action_feature_dim=action_feature_dim,
        discrete_actions=2 if is_discrete else None,
        action_low=None if is_discrete else [-1.0],
        action_high=None if is_discrete else [1.0],
        model_config={
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
        },
    )
    batch_size, num_particles = 4, 5
    observations = torch.randn(batch_size, 2)
    previous_particles = torch.randn(batch_size, num_particles, 2)
    previous_actions = torch.randn(batch_size, action_feature_dim)
    initial = torch.tensor([True, False, False, True])
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
    if is_discrete:
        assert model.categorical_head is not None
        current_log_prob = Categorical(logits=model.categorical_head(update_condition)).log_prob(
            sample.action
        )
    else:
        assert model.diffusion_policy is not None
        assert sample.diffusion_chain is not None
        current_log_prob, _ = model.diffusion_policy.log_prob_chain(
            update_condition, sample.diffusion_chain
        )

    torch.testing.assert_close(current_log_prob, old_log_prob, atol=1e-6, rtol=1e-6)
    log_ratio = current_log_prob - old_log_prob
    ratio = log_ratio.exp()
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        approximate_kl,
        torch.zeros_like(approximate_kl),
        atol=1e-7,
        rtol=0,
    )


def test_diffusion_policy_rejects_nonfinite_action_bounds() -> None:
    with pytest.raises(ValueError, match="finite"):
        DiffusionPolicy(
            condition_dim=4,
            action_low=[-float("inf")],
            action_high=[1.0],
            hidden_dims=[8],
            num_steps=2,
            beta_start=0.05,
            beta_end=0.1,
            min_std=0.05,
        )
