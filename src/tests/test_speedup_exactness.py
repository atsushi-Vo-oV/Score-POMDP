from __future__ import annotations

import copy
from dataclasses import fields, replace
from unittest.mock import patch

import pytest
import torch

from sb_pomdp import ppo
from sb_pomdp.belief import EnergyBelief, PotentialBranch
from sb_pomdp.buffer import BeliefReplayPrefix, RolloutBatch
from sb_pomdp.policies import DiffusionPolicy, ScoreBeliefActorCritic
from sb_pomdp.ppo import _replay_belief_step


def _old_replay(
    model: ScoreBeliefActorCritic,
    rollout: RolloutBatch,
    environment_indices: torch.Tensor,
    model_config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay one rollout-long environment group with saved ULA randomness.

    An active-episode prefix is replayed across rollout boundaries.  ``full``
    therefore retains the graph from the episode start, while ``tbptt_1``
    detaches at every environment-time boundary.  Realised observations and
    actions remain fixed PPO replay data; no environment model is used or
    differentiated.
    """

    gradient_mode = str(model_config["belief_gradient_mode"])
    if gradient_mode not in {"full", "tbptt_1"}:
        raise ValueError("model.belief_gradient_mode must be full or tbptt_1")

    observations = rollout.observations[:, environment_indices]
    previous_actions = rollout.previous_actions[:, environment_indices]
    episode_starts = rollout.initial[:, environment_indices]
    initial_noises = rollout.ula_initial_noise[:, environment_indices]
    step_noises = rollout.ula_step_noises[:, environment_indices]
    energy_belief = isinstance(model.belief, EnergyBelief)
    rollout_branches = (
        model.belief.classify_potential_branches(episode_starts)
        if energy_belief
        else (None,) * rollout.rollout_shape[0]
    )
    if len(rollout.belief_prefixes) != rollout.rollout_shape[1]:
        raise ValueError("one belief replay prefix is required per rollout environment")
    activation_checkpoint = bool(model_config.get("belief_activation_checkpoint", True))
    selected_environment_indices = environment_indices.detach().cpu().tolist()
    selected_prefixes = tuple(
        rollout.belief_prefixes[environment_index]
        for environment_index in selected_environment_indices
    )
    prefix_initial_values = (
        torch.cat([prefix.initial for prefix in selected_prefixes if prefix.length])
        .detach()
        .cpu()
        .tolist()
        if energy_belief and any(prefix.length for prefix in selected_prefixes)
        else []
    )
    prefix_initial_offset = 0
    prefix_particles: list[torch.Tensor] = []
    for environment_index, prefix in zip(
        selected_environment_indices,
        selected_prefixes,
        strict=True,
    ):
        if not prefix.length:
            prefix_particles.append(
                rollout.previous_particles[0, environment_index : environment_index + 1].detach()
            )
            continue
        prefix_branches: tuple[PotentialBranch | None, ...]
        if energy_belief:
            prefix_branches = tuple(
                "initial" if bool(value) else "recursive"
                for value in prefix_initial_values[
                    prefix_initial_offset : prefix_initial_offset + prefix.length
                ]
            )
            prefix_initial_offset += prefix.length
        else:
            prefix_branches = (None,) * prefix.length
        previous = torch.zeros_like(prefix.ula_initial_noise[:1])
        track_prefix_grad = gradient_mode == "full"
        for prefix_index in range(prefix.length):
            previous, _ = _replay_belief_step(
                model,
                prefix.observations[prefix_index : prefix_index + 1],
                previous,
                prefix.previous_actions[prefix_index : prefix_index + 1],
                prefix.initial[prefix_index : prefix_index + 1],
                prefix.ula_initial_noise[prefix_index : prefix_index + 1],
                prefix.ula_step_noises[prefix_index : prefix_index + 1],
                step_size=float(model_config["langevin_step_size"]),
                gradient_mode=gradient_mode,
                track_grad=track_prefix_grad,
                potential_branch=prefix_branches[prefix_index],
                activation_checkpoint=activation_checkpoint,
            )
        prefix_particles.append(previous)
    previous_particles = torch.cat(prefix_particles, dim=0)
    replayed_particles: list[torch.Tensor] = []
    replayed_scores: list[torch.Tensor] = []

    for time_index in range(rollout.rollout_shape[0]):
        if time_index:
            reset = episode_starts[time_index, :, None, None]
            previous_particles = torch.where(
                reset,
                torch.zeros_like(previous_particles),
                previous_particles,
            )
        particles, scores = _replay_belief_step(
            model,
            observations[time_index],
            previous_particles,
            previous_actions[time_index],
            episode_starts[time_index],
            initial_noises[time_index],
            step_noises[time_index],
            step_size=float(model_config["langevin_step_size"]),
            gradient_mode=gradient_mode,
            track_grad=True,
            potential_branch=rollout_branches[time_index],
            activation_checkpoint=activation_checkpoint,
        )
        replayed_particles.append(particles)
        replayed_scores.append(scores)
        previous_particles = particles

    return torch.stack(replayed_particles), torch.stack(replayed_scores)


def _old_log_prob_chain(
    self,
    condition: torch.Tensor,
    chain: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = (condition.shape[0], self.num_steps + 1, self.action_dim)
    if tuple(chain.shape) != expected:
        raise ValueError(f"expected diffusion chain shape {expected}, got {tuple(chain.shape)}")
    log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for chain_index, step_index in enumerate(reversed(range(self.num_steps))):
        distribution = self.transition_distribution(
            condition,
            chain[:, chain_index],
            step_index,
        )
        log_probabilities.append(distribution.log_prob(chain[:, chain_index + 1]))
        entropies.append(distribution.entropy())
    return torch.stack(log_probabilities, dim=1), torch.stack(entropies, dim=1)


def _fixture(parameterization="energy", sampler="ula", encoder="transformer", continuous=False):
    torch.manual_seed(104)
    config = {
        "energy_hidden": [8],
        "score_clip": 10.0,
        "particle_clip": 10.0,
        "d_model": 8,
        "num_heads": 2,
        "num_transformer_layers": 1,
        "transformer_ff_dim": 16,
        "dropout": 0.0,
        "policy_hidden": [8],
        "score_parameterization": parameterization,
        "particle_sampler": sampler,
        "encoder_kind": encoder,
        "belief_gradient_mode": "full",
        "belief_activation_checkpoint": False,
        "langevin_step_size": 0.02,
        "diffusion_steps": 2,
        "diffusion_beta_start": 0.05,
        "diffusion_beta_end": 0.15,
        "diffusion_min_std": 0.05,
        "diffusion_advantage_discount": 0.95,
    }
    model = ScoreBeliefActorCritic(
        observation_dim=2,
        state_dim=2,
        action_kind="continuous" if continuous else "discrete",
        action_feature_dim=2,
        discrete_actions=None if continuous else 2,
        action_low=[-1.0, -1.0] if continuous else None,
        action_high=[1.0, 1.0] if continuous else None,
        model_config=config,
    )
    g = torch.Generator().manual_seed(203)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    prefixes = tuple(
        BeliefReplayPrefix(
            observations=rand(n, 2),
            previous_actions=rand(n, 2),
            initial=torch.tensor([i == 0 for i in range(n)], dtype=torch.bool),
            ula_initial_noise=rand(n, 2, 2),
            ula_step_noises=rand(n, 2, 2, 2),
        )
        for n in (0, 1, 3)
    )
    shape = (3, 3)
    rollout = RolloutBatch(
        observations=rand(*shape, 2),
        previous_particles=rand(*shape, 2, 2),
        previous_actions=rand(*shape, 2),
        initial=torch.tensor([[False, False, False], [False, True, False], [False, False, False]]),
        particles=rand(*shape, 2, 2),
        ula_initial_noise=rand(*shape, 2, 2),
        ula_step_noises=rand(*shape, 2, 2, 2),
        actions=torch.tensor([[0, 1, 0], [1, 0, 1], [0, 0, 1]]),
        old_log_prob=rand(*shape),
        old_values=rand(*shape),
        rewards=rand(*shape),
        dones=torch.zeros(*shape, dtype=torch.bool),
        advantages=rand(*shape),
        returns=rand(*shape),
        diffusion_chains=None,
        pre_tanh_actions=None,
        belief_prefixes=prefixes,
    )
    if continuous:
        rollout = replace(
            rollout,
            actions=rand(*shape, 2),
            old_log_prob=torch.zeros(*shape, 2),
            diffusion_chains=rand(*shape, 3, 2) * 0.1,
        )
    return model, config, rollout


def _grads(model):
    return {
        n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()
    }


def _compare_grads(a, b):
    assert a.keys() == b.keys()
    for name in a:
        assert (a[name] is None) == (b[name] is None), name
        if a[name] is not None:
            torch.testing.assert_close(
                a[name], b[name], atol=1e-5, rtol=1e-5, msg=lambda m, name=name: f"{name}: {m}"
            )


@pytest.mark.parametrize("mode", ["full", "tbptt_1"])
@pytest.mark.parametrize("parameterization", ["energy", "direct", "direct_joint"])
@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
def test_prefix_replay_matches_previous_implementation(mode, parameterization, sampler):
    model, config, rollout = _fixture(parameterization, sampler)
    config["belief_gradient_mode"] = mode
    results = []
    for replay in (_old_replay, ppo.replay_belief_sequence):
        model.zero_grad(set_to_none=True)
        particles, scores = replay(model, rollout, torch.tensor([2, 0, 1]), config)
        (particles.square().sum() + scores.square().sum()).backward()
        results.append((particles.detach(), scores.detach(), _grads(model)))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(results[0][1], results[1][1], atol=1e-6, rtol=1e-6)
    _compare_grads(results[0][2], results[1][2])


@pytest.mark.parametrize("mode", ["full", "tbptt_1"])
@pytest.mark.parametrize("parameterization", ["energy", "direct", "direct_joint"])
@pytest.mark.parametrize("sampler", ["ula", "svgd"])
def test_alpha_pool_prefix_replay_matches_previous_implementation(mode, parameterization, sampler):
    model, config, rollout = _fixture(parameterization, sampler, "alpha_pool")
    config["belief_gradient_mode"] = mode
    results = []
    for replay in (_old_replay, ppo.replay_belief_sequence):
        model.zero_grad(set_to_none=True)
        particles, scores = replay(model, rollout, torch.tensor([2, 0, 1]), config)
        condition = model.encode(particles.flatten(0, 1), scores.flatten(0, 1))
        (condition.square().sum() + particles.square().sum()).backward()
        results.append((particles.detach(), scores.detach(), condition.detach(), _grads(model)))
    for old, new in zip(results[0][:3], results[1][:3], strict=True):
        torch.testing.assert_close(old, new, atol=1e-6, rtol=1e-6)
    _compare_grads(results[0][3], results[1][3])


@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
@pytest.mark.parametrize("track_grad", [True, False])
@pytest.mark.parametrize("parameterization", ["energy", "direct", "direct_joint"])
def test_optional_final_score_preserves_particles_and_rng(sampler, track_grad, parameterization):
    model, config, rollout = _fixture(parameterization, sampler)
    belief = model.belief
    args = (
        rollout.observations[0],
        rollout.previous_particles[0],
        rollout.previous_actions[0],
        rollout.initial[0],
    )
    kwargs = {
        "initial_noise": rollout.ula_initial_noise[0],
        "langevin_noise": rollout.ula_step_noises[0],
        "step_size": config["langevin_step_size"],
        "track_grad": track_grad,
    }
    results = []
    for need_scores in (True, False):
        model.zero_grad(set_to_none=True)
        rng = torch.random.get_rng_state().clone()
        with patch.object(
            belief, "_score_at_for_branch", wraps=belief._score_at_for_branch
        ) as score:
            particles, scores = belief.particles_from_noise(
                *args, **kwargs, need_scores=need_scores
            )
        assert torch.equal(rng, torch.random.get_rng_state())
        if track_grad:
            particles.square().sum().backward()
        results.append((particles.detach(), scores, score.call_count, _grads(model)))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    assert results[0][1] is not None and results[1][1] is None
    assert results[0][2] == results[1][2] + 1
    if track_grad:
        _compare_grads(results[0][3], results[1][3])


@pytest.mark.parametrize("mode", ["full", "tbptt_1"])
@pytest.mark.parametrize("parameterization", ["energy", "direct", "direct_joint"])
@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
@pytest.mark.parametrize("encoder", ["transformer", "alpha_pool"])
def test_update_ppo_matches_previous_replay(mode, parameterization, sampler, encoder):
    model, config, rollout = _fixture(parameterization, sampler, encoder, continuous=True)
    config["belief_gradient_mode"] = mode
    other = copy.deepcopy(model)
    opt1 = torch.optim.Adam(model.parameters(), lr=0.001)
    opt2 = torch.optim.Adam(other.parameters(), lr=0.001)
    ppo_config = {
        "minibatch_size": 9,
        "sequence_microbatch_size": 9,
        "normalize_advantage": False,
        "epochs": 2,
        "clip_coef": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 100.0,
    }
    rng = torch.random.get_rng_state().clone()
    gen1 = torch.Generator().manual_seed(31)
    with (
        patch.object(ppo, "replay_belief_sequence", _old_replay),
        patch.object(DiffusionPolicy, "log_prob_chain", _old_log_prob_chain),
    ):
        metrics1 = ppo.update_ppo(model, opt1, rollout, config, ppo_config, generator=gen1)
    rng1 = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng)
    gen2 = torch.Generator().manual_seed(31)
    metrics2 = ppo.update_ppo(other, opt2, rollout, config, ppo_config, generator=gen2)
    for field in fields(metrics1):
        torch.testing.assert_close(
            torch.tensor(getattr(metrics1, field.name)),
            torch.tensor(getattr(metrics2, field.name)),
            atol=1e-6,
            rtol=1e-5,
        )
    _compare_grads(_grads(model), _grads(other))
    assert torch.equal(rng1, torch.random.get_rng_state())
    assert torch.equal(gen1.get_state(), gen2.get_state())
    # Losses and gradients agree to 1e-6 / 1e-5 above; the batched summation order still moves a
    # gradient element that is itself ~1e-7 by that much, and Adam normalises every element by its
    # running magnitude, so such an element's update can differ by up to the learning rate (1e-3)
    # times its sign flip. Observed 2.7e-5 on Linux MKL (transformer / smi / direct_joint / full).
    for (name, p1), (_, p2) in zip(model.named_parameters(), other.named_parameters(), strict=True):
        torch.testing.assert_close(
            p1, p2, atol=1e-4, rtol=1e-5, msg=lambda m, name=name: f"{name}: {m}"
        )
        for key in opt1.state[p1]:
            torch.testing.assert_close(
                opt1.state[p1][key], opt2.state[p2][key], atol=1e-6, rtol=1e-5
            )


@pytest.mark.parametrize("mode", ["full", "tbptt_1"])
def test_checkpointed_prefix_replay_matches_old(mode):
    model, config, rollout = _fixture()
    config.update(belief_gradient_mode=mode, belief_activation_checkpoint=True)
    results = []
    for replay in (_old_replay, ppo.replay_belief_sequence):
        model.zero_grad(set_to_none=True)
        particles, scores = replay(model, rollout, torch.tensor([2, 0, 1]), config)
        (particles.square().sum() + scores.square().sum()).backward()
        results.append((particles.detach(), scores.detach(), _grads(model)))
    for a, b in zip(results[0][:2], results[1][:2], strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    _compare_grads(results[0][2], results[1][2])


def test_prefix_batch_has_no_cross_environment_gradient():
    model, config, rollout = _fixture("direct", "ula")
    # Perturb only environment 1's prefix; environment 2's output and parameter
    # gradients must not depend on another row sharing the prefix batch.
    altered_prefix = replace(
        rollout.belief_prefixes[1], observations=rollout.belief_prefixes[1].observations + 10
    )
    altered = replace(
        rollout,
        belief_prefixes=(rollout.belief_prefixes[0], altered_prefix, rollout.belief_prefixes[2]),
    )
    results = []
    for batch in (rollout, altered):
        model.zero_grad(set_to_none=True)
        particles, scores = ppo.replay_belief_sequence(model, batch, torch.tensor([1, 2]), config)
        (particles[:, 1].square().sum() + scores[:, 1].square().sum()).backward()
        results.append((particles[:, 1].detach(), scores[:, 1].detach(), _grads(model)))
    for a, b in zip(results[0][:2], results[1][:2], strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    _compare_grads(results[0][2], results[1][2])


@pytest.mark.parametrize("mixed", [False, True])
def test_prefix_only_replays_active_rows_on_their_saved_branch(mixed):
    model, config, rollout = _fixture()
    if mixed:
        rollout.belief_prefixes[1].initial[0] = False
    expected = _old_replay(model, rollout, torch.tensor([0, 1, 2]), config)
    with patch.object(
        model.belief,
        "particles_from_noise",
        wraps=model.belief.particles_from_noise,
    ) as replay:
        actual = ppo.replay_belief_sequence(model, rollout, torch.tensor([0, 1, 2]), config)
    prefix_calls = [
        call for call in replay.call_args_list if not call.kwargs.get("need_scores", True)
    ]
    assert [call.args[0].shape[0] for call in prefix_calls] == (
        [1, 1, 1, 1] if mixed else [2, 1, 1]
    )
    for call in prefix_calls:
        initial = call.args[3]
        branch = call.kwargs["potential_branch"]
        assert torch.all(initial if branch == "initial" else ~initial)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("sampler", ["ula", "smi"])
@pytest.mark.parametrize("encoder", ["transformer", "alpha_pool"])
def test_update_ppo_is_invariant_to_sequence_microbatch(sampler, encoder):
    """``ppo.sequence_microbatch_size`` is execution-only: replaying one environment per
    microbatch and the whole minibatch at once give the same metrics, gradients and optimizer
    steps (episodes end at different times so the per-sequence mean matters)."""

    model, config, rollout = _fixture("energy", sampler, encoder, continuous=True)
    dones = torch.zeros_like(rollout.dones)
    dones[1, 0] = True
    dones[2, 2] = True
    rollout = replace(rollout, dones=dones)
    other = copy.deepcopy(model)
    opt1 = torch.optim.Adam(model.parameters(), lr=0.001)
    opt2 = torch.optim.Adam(other.parameters(), lr=0.001)
    base = {
        "minibatch_size": 9,
        "normalize_advantage": True,
        "epochs": 2,
        "clip_coef": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 100.0,
    }
    rng = torch.random.get_rng_state().clone()
    gen1 = torch.Generator().manual_seed(31)
    metrics1 = ppo.update_ppo(
        model, opt1, rollout, config, {**base, "sequence_microbatch_size": 9}, generator=gen1
    )
    rng1 = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng)
    gen2 = torch.Generator().manual_seed(31)
    metrics2 = ppo.update_ppo(
        other, opt2, rollout, config, {**base, "sequence_microbatch_size": 3}, generator=gen2
    )
    for field in fields(metrics1):
        torch.testing.assert_close(
            torch.tensor(getattr(metrics1, field.name)),
            torch.tensor(getattr(metrics2, field.name)),
            atol=1e-6,
            rtol=1e-5,
            msg=lambda m, name=field.name: f"{name}: {m}",
        )
    _compare_grads(_grads(model), _grads(other))
    assert torch.equal(rng1, torch.random.get_rng_state())
    assert torch.equal(gen1.get_state(), gen2.get_state())
    # Gradients agree to 1e-5 above.  Adam divides every element by its running magnitude, so an
    # element whose gradient is itself ~1e-7 (a transformer bias here) can move by up to the
    # learning rate when the summation order flips its sign; observed 2.0e-4 on Windows MKL.
    for (name, p1), (_, p2) in zip(model.named_parameters(), other.named_parameters(), strict=True):
        torch.testing.assert_close(
            p1, p2, atol=1e-3, rtol=1e-5, msg=lambda m, name=name: f"{name}: {m}"
        )
