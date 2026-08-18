from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional

from sb_pomdp.buffer import BeliefReplayPrefix, RolloutBatch
from sb_pomdp.policies import ScoreBeliefActorCritic
from sb_pomdp.ppo import replay_belief_sequence


class _TwoPhaseTemporalBelief(nn.Module):
    """Minimal recurrence that exposes gradients before and after a reset."""

    def __init__(self) -> None:
        super().__init__()
        self.phase_weights = nn.Parameter(torch.tensor([1.0, 1.0]))

    def particles_from_noise(
        self,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        *_: torch.Tensor,
        initial_noise: torch.Tensor,
        track_grad: bool,
        temporal_gradient_mode: str,
        **__: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        incoming = (
            previous_particles.detach()
            if temporal_gradient_mode == "tbptt_1"
            else previous_particles
        )
        with torch.set_grad_enabled(track_grad):
            increment = (observation * self.phase_weights).sum(dim=-1)
            particles = incoming + initial_noise + increment[:, None, None]
        if not track_grad:
            particles = particles.detach()
        return particles, particles


class _TwoPhaseTemporalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.belief = _TwoPhaseTemporalBelief()


def _empty_prefix(*, observation_dim: int, action_dim: int) -> BeliefReplayPrefix:
    return BeliefReplayPrefix(
        observations=torch.empty(0, observation_dim),
        previous_actions=torch.empty(0, action_dim),
        initial=torch.empty(0, dtype=torch.bool),
        ula_initial_noise=torch.empty(0, 1, 1),
        ula_step_noises=torch.empty(0, 1, 1, 1),
    )


def _reset_rollout() -> RolloutBatch:
    time_steps = 3
    particles = torch.zeros(time_steps, 1, 1, 1)
    return RolloutBatch(
        observations=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]]]),
        previous_particles=particles.clone(),
        previous_actions=torch.zeros(time_steps, 1, 1),
        initial=torch.tensor([[False], [True], [False]]),
        particles=particles,
        ula_initial_noise=torch.zeros(time_steps, 1, 1, 1),
        ula_step_noises=torch.zeros(time_steps, 1, 1, 1, 1),
        actions=torch.zeros(time_steps, 1, dtype=torch.long),
        old_log_prob=torch.zeros(time_steps, 1),
        old_values=torch.zeros(time_steps, 1),
        rewards=torch.zeros(time_steps, 1),
        dones=torch.zeros(time_steps, 1, dtype=torch.bool),
        advantages=torch.zeros(time_steps, 1),
        returns=torch.zeros(time_steps, 1),
        diffusion_chains=None,
        pre_tanh_actions=None,
        belief_prefixes=(_empty_prefix(observation_dim=2, action_dim=1),),
    )


@pytest.mark.parametrize(
    ("gradient_mode", "expected_post_reset_gradient"),
    [("full", 2.0), ("tbptt_1", 1.0)],
)
def test_episode_reset_blocks_pre_reset_history_gradient(
    gradient_mode: str,
    expected_post_reset_gradient: float,
) -> None:
    model = _TwoPhaseTemporalModel()
    particles, _ = replay_belief_sequence(
        model,  # type: ignore[arg-type]
        _reset_rollout(),
        torch.tensor([0]),
        {
            "belief_gradient_mode": gradient_mode,
            "langevin_step_size": 0.1,
        },
    )

    (gradient,) = torch.autograd.grad(particles[-1].sum(), model.belief.phase_weights)

    # The first coefficient occurs only before the reset.  The second remains
    # live after it, so a zero first component cannot be explained by a dead graph.
    torch.testing.assert_close(gradient[0], torch.tensor(0.0), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        gradient[1],
        torch.tensor(expected_post_reset_gradient),
        atol=0.0,
        rtol=0.0,
    )


def _model_config(policy_kind: str, gradient_mode: str) -> dict[str, Any]:
    return {
        "energy_hidden": [8],
        "score_clip": 10.0,
        "particle_clip": 10.0,
        "d_model": 8,
        "num_heads": 2,
        "num_transformer_layers": 1,
        "transformer_ff_dim": 16,
        "dropout": 0.0,
        "policy_hidden": [8],
        "diffusion_steps": 2,
        "diffusion_beta_start": 0.05,
        "diffusion_beta_end": 0.15,
        "diffusion_min_std": 0.05,
        "diffusion_advantage_discount": 0.95,
        "continuous_policy_kind": (
            policy_kind if policy_kind in {"diffusion", "gaussian"} else "diffusion"
        ),
        "belief_gradient_mode": gradient_mode,
        "langevin_step_size": 0.02,
    }


def _actor_critic(policy_kind: str, gradient_mode: str) -> ScoreBeliefActorCritic:
    discrete = policy_kind == "categorical"
    return ScoreBeliefActorCritic(
        observation_dim=2,
        state_dim=2,
        action_kind="discrete" if discrete else "continuous",
        action_feature_dim=2 if discrete else 1,
        discrete_actions=2 if discrete else None,
        action_low=None if discrete else [-1.0],
        action_high=None if discrete else [1.0],
        model_config=_model_config(policy_kind, gradient_mode),
    )


def _next_action_features(
    model: ScoreBeliefActorCritic,
    action: torch.Tensor,
) -> torch.Tensor:
    if model.action_kind == "discrete":
        return functional.one_hot(action.long(), num_classes=2).float()
    return action


def _rollout_with_nonempty_prefix(
    model: ScoreBeliefActorCritic,
    model_config: dict[str, Any],
) -> RolloutBatch:
    generator = torch.Generator().manual_seed(719)
    num_particles = 2
    langevin_steps = 1
    previous_particles = torch.zeros(1, num_particles, 2)
    previous_action = torch.zeros(1, 2 if model.action_kind == "discrete" else 1)

    prefix_observations: list[torch.Tensor] = []
    prefix_actions: list[torch.Tensor] = []
    prefix_initial: list[torch.Tensor] = []
    prefix_initial_noises: list[torch.Tensor] = []
    prefix_step_noises: list[torch.Tensor] = []
    for prefix_index in range(2):
        observation = torch.randn(1, 2, generator=generator)
        initial = torch.tensor([prefix_index == 0])
        initial_noise, step_noise = model.belief.draw_particle_noise(
            observation,
            num_particles=num_particles,
            num_steps=langevin_steps,
            generator=generator,
        )
        particles, _ = model.belief.particles_from_noise(
            observation,
            previous_particles,
            previous_action,
            initial,
            initial_noise=initial_noise,
            langevin_noise=step_noise,
            step_size=float(model_config["langevin_step_size"]),
            track_grad=False,
        )
        prefix_observations.append(observation.squeeze(0))
        prefix_actions.append(previous_action.squeeze(0))
        prefix_initial.append(initial.squeeze(0))
        prefix_initial_noises.append(initial_noise.squeeze(0))
        prefix_step_noises.append(step_noise.squeeze(0))
        previous_particles = particles
        previous_action = torch.randn(
            previous_action.shape,
            generator=generator,
        )

    prefix = BeliefReplayPrefix(
        observations=torch.stack(prefix_observations),
        previous_actions=torch.stack(prefix_actions),
        initial=torch.stack(prefix_initial),
        ula_initial_noise=torch.stack(prefix_initial_noises),
        ula_step_noises=torch.stack(prefix_step_noises),
    )

    observations: list[torch.Tensor] = []
    previous_particle_steps: list[torch.Tensor] = []
    previous_action_steps: list[torch.Tensor] = []
    initials: list[torch.Tensor] = []
    particle_steps: list[torch.Tensor] = []
    initial_noises: list[torch.Tensor] = []
    step_noises: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    old_values: list[torch.Tensor] = []
    diffusion_chains: list[torch.Tensor] = []
    pre_tanh_actions: list[torch.Tensor] = []

    model.eval()
    for _ in range(2):
        observation = torch.randn(1, 2, generator=generator)
        initial = torch.zeros(1, dtype=torch.bool)
        initial_noise, step_noise = model.belief.draw_particle_noise(
            observation,
            num_particles=num_particles,
            num_steps=langevin_steps,
            generator=generator,
        )
        particles, scores = model.belief.particles_from_noise(
            observation,
            previous_particles,
            previous_action,
            initial,
            initial_noise=initial_noise,
            langevin_noise=step_noise,
            step_size=float(model_config["langevin_step_size"]),
            track_grad=False,
        )
        with torch.no_grad():
            condition = model.encode(particles, scores)
            sample = model.sample_policy(
                condition,
                deterministic=False,
                generator=generator,
            )
            value = model.value(condition)

        observations.append(observation)
        previous_particle_steps.append(previous_particles)
        previous_action_steps.append(previous_action)
        initials.append(initial)
        particle_steps.append(particles)
        initial_noises.append(initial_noise)
        step_noises.append(step_noise)
        actions.append(sample.action)
        old_log_probs.append(sample.log_prob)
        old_values.append(value)
        if sample.diffusion_chain is not None:
            diffusion_chains.append(sample.diffusion_chain)
        if sample.pre_tanh_action is not None:
            pre_tanh_actions.append(sample.pre_tanh_action)

        previous_particles = particles
        previous_action = _next_action_features(model, sample.action)

    time_steps = len(observations)
    return RolloutBatch(
        observations=torch.stack(observations),
        previous_particles=torch.stack(previous_particle_steps),
        previous_actions=torch.stack(previous_action_steps),
        initial=torch.stack(initials),
        particles=torch.stack(particle_steps),
        ula_initial_noise=torch.stack(initial_noises),
        ula_step_noises=torch.stack(step_noises),
        actions=torch.stack(actions),
        old_log_prob=torch.stack(old_log_probs),
        old_values=torch.stack(old_values),
        rewards=torch.zeros(time_steps, 1),
        dones=torch.zeros(time_steps, 1, dtype=torch.bool),
        advantages=torch.zeros(time_steps, 1),
        returns=torch.zeros(time_steps, 1),
        diffusion_chains=(
            torch.stack(diffusion_chains) if diffusion_chains else None
        ),
        pre_tanh_actions=(
            torch.stack(pre_tanh_actions) if pre_tanh_actions else None
        ),
        belief_prefixes=(prefix,),
    )


@pytest.mark.parametrize("gradient_mode", ["full", "tbptt_1"])
@pytest.mark.parametrize("policy_kind", ["categorical", "diffusion", "gaussian"])
def test_nonempty_prefix_replay_preserves_old_policy_log_probability(
    policy_kind: str,
    gradient_mode: str,
) -> None:
    torch.manual_seed(701)
    model_config = _model_config(policy_kind, gradient_mode)
    model = _actor_critic(policy_kind, gradient_mode)
    rollout = _rollout_with_nonempty_prefix(model, model_config)

    model.train()
    replayed_particles, replayed_scores = replay_belief_sequence(
        model,
        rollout,
        torch.tensor([0]),
        model_config,
    )
    torch.testing.assert_close(
        replayed_particles,
        rollout.particles,
        atol=1e-6,
        rtol=1e-6,
    )
    condition = model.encode(
        replayed_particles.flatten(0, 1),
        replayed_scores.flatten(0, 1),
    )

    if policy_kind == "categorical":
        assert model.categorical_head is not None
        new_log_prob = Categorical(logits=model.categorical_head(condition)).log_prob(
            rollout.actions.flatten(0, 1).long()
        )
    elif policy_kind == "diffusion":
        assert model.diffusion_policy is not None
        assert rollout.diffusion_chains is not None
        chains = rollout.diffusion_chains.flatten(0, 1)
        new_log_prob, _ = model.diffusion_policy.log_prob_chain(condition, chains)
    else:
        assert model.gaussian_policy is not None
        assert rollout.pre_tanh_actions is not None
        pre_tanh_actions = rollout.pre_tanh_actions.flatten(0, 1)
        new_log_prob, _ = model.gaussian_policy.log_prob(
            condition,
            pre_tanh_action=pre_tanh_actions,
        )

    old_log_prob = rollout.old_log_prob.flatten(0, 1)
    torch.testing.assert_close(new_log_prob, old_log_prob, atol=2e-6, rtol=2e-6)
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        approximate_kl,
        torch.zeros_like(approximate_kl),
        atol=1e-7,
        rtol=0.0,
    )
