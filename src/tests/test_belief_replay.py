from __future__ import annotations

import torch
from torch import nn

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.buffer import BeliefReplayPrefix, RolloutBatch
from sb_pomdp.ppo import replay_belief_sequence


def _context() -> tuple[
    EnergyBelief,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(101)
    belief = EnergyBelief(
        observation_dim=2,
        state_dim=2,
        action_feature_dim=1,
        hidden_dims=[8],
        score_clip=100.0,
        particle_clip=100.0,
    )
    observation = torch.randn(2, 2)
    previous_particles = torch.randn(2, 3, 2)
    previous_action = torch.randn(2, 1)
    initial = torch.zeros(2, dtype=torch.bool)
    generator = torch.Generator().manual_seed(202)
    initial_noise, langevin_noise = belief.draw_particle_noise(
        observation,
        num_particles=3,
        num_steps=3,
        generator=generator,
    )
    return (
        belief,
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise,
        langevin_noise,
    )


def _energy_gradient_norms(belief: EnergyBelief) -> tuple[float, float]:
    observation_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in belief.observation_energy.parameters()
        if parameter.grad is not None
    )
    transition_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in belief.transition_energy.parameters()
        if parameter.grad is not None
    )
    return observation_norm, transition_norm


def test_explicit_noise_rollout_and_differentiable_replay_match() -> None:
    (
        belief,
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise,
        langevin_noise,
    ) = _context()

    rollout_particles, rollout_scores = belief.particles_from_noise(
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=langevin_noise,
        step_size=[0.01, 0.02, 0.03],
        track_grad=False,
    )
    replay_particles, replay_scores = belief.particles_from_noise(
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=langevin_noise,
        step_size=[0.01, 0.02, 0.03],
        track_grad=True,
        temporal_gradient_mode="full",
    )

    torch.testing.assert_close(replay_particles, rollout_particles, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(replay_scores, rollout_scores, atol=1e-7, rtol=1e-6)
    assert replay_particles.grad_fn is not None
    assert replay_scores.grad_fn is not None
    assert rollout_particles.grad_fn is None
    assert rollout_scores.grad_fn is None


def test_full_replay_differentiates_through_incoming_particles_and_ula() -> None:
    (
        belief,
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise,
        langevin_noise,
    ) = _context()
    previous_particles.requires_grad_(True)

    particles, _ = belief.particles_from_noise(
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=langevin_noise,
        step_size=0.02,
        track_grad=True,
        temporal_gradient_mode="full",
    )
    particles.square().mean().backward()

    assert previous_particles.grad is not None
    assert torch.isfinite(previous_particles.grad).all()
    assert float(previous_particles.grad.abs().sum()) > 0.0
    observation_norm, transition_norm = _energy_gradient_norms(belief)
    assert observation_norm > 0.0
    assert transition_norm > 0.0


def test_one_step_replay_cuts_only_the_temporal_boundary() -> None:
    (
        belief,
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise,
        langevin_noise,
    ) = _context()
    previous_particles.requires_grad_(True)

    particles, _ = belief.particles_from_noise(
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=langevin_noise,
        step_size=0.02,
        track_grad=True,
        temporal_gradient_mode="tbptt_1",
    )
    particles.square().mean().backward()

    assert previous_particles.grad is None
    observation_norm, transition_norm = _energy_gradient_norms(belief)
    # The loss is on sampled particles, so these gradients can only arrive via
    # the differentiable ULA steps (not via the final returned score tensor).
    assert observation_norm > 0.0
    assert transition_norm > 0.0


def test_full_and_one_step_replay_have_identical_forward_values() -> None:
    (
        belief,
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise,
        langevin_noise,
    ) = _context()
    full_particles, full_scores = belief.particles_from_noise(
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=langevin_noise,
        step_size=0.02,
        track_grad=True,
        temporal_gradient_mode="full",
    )
    one_step_particles, one_step_scores = belief.particles_from_noise(
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=langevin_noise,
        step_size=0.02,
        track_grad=True,
        temporal_gradient_mode="tbptt_1",
    )

    torch.testing.assert_close(one_step_particles, full_particles)
    torch.testing.assert_close(one_step_scores, full_scores)


class _LinearTemporalBelief(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

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
            particles = incoming + initial_noise + observation[:, None, :] * self.weight
        if not track_grad:
            particles = particles.detach()
        return particles, particles


class _LinearTemporalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.belief = _LinearTemporalBelief()


def _prefix_rollout() -> RolloutBatch:
    prefix = BeliefReplayPrefix(
        observations=torch.ones(1, 1),
        previous_actions=torch.zeros(1, 1),
        initial=torch.ones(1, dtype=torch.bool),
        ula_initial_noise=torch.zeros(1, 1, 1),
        ula_step_noises=torch.zeros(1, 1, 1, 1),
    )
    return RolloutBatch(
        observations=torch.zeros(1, 1, 1),
        previous_particles=torch.ones(1, 1, 1, 1),
        previous_actions=torch.zeros(1, 1, 1),
        initial=torch.zeros(1, 1, dtype=torch.bool),
        particles=torch.ones(1, 1, 1, 1),
        ula_initial_noise=torch.zeros(1, 1, 1, 1),
        ula_step_noises=torch.zeros(1, 1, 1, 1, 1),
        actions=torch.zeros(1, 1, dtype=torch.long),
        old_log_prob=torch.zeros(1, 1),
        old_values=torch.zeros(1, 1),
        rewards=torch.zeros(1, 1),
        dones=torch.zeros(1, 1, dtype=torch.bool),
        advantages=torch.zeros(1, 1),
        returns=torch.zeros(1, 1),
        diffusion_chains=None,
        pre_tanh_actions=None,
        belief_prefixes=(prefix,),
    )


def test_sequence_replay_full_crosses_rollout_boundary_but_tbptt_one_does_not() -> None:
    model = _LinearTemporalModel()
    rollout = _prefix_rollout()
    environment_indices = torch.tensor([0])

    full_particles, _ = replay_belief_sequence(
        model,  # type: ignore[arg-type]
        rollout,
        environment_indices,
        {"belief_gradient_mode": "full", "langevin_step_size": 0.1},
    )
    (full_gradient,) = torch.autograd.grad(full_particles.sum(), model.belief.weight)
    one_step_particles, _ = replay_belief_sequence(
        model,  # type: ignore[arg-type]
        rollout,
        environment_indices,
        {"belief_gradient_mode": "tbptt_1", "langevin_step_size": 0.1},
    )
    (one_step_gradient,) = torch.autograd.grad(
        one_step_particles.sum(), model.belief.weight
    )

    torch.testing.assert_close(full_particles, one_step_particles)
    torch.testing.assert_close(full_gradient, torch.tensor(1.0))
    torch.testing.assert_close(one_step_gradient, torch.tensor(0.0))
