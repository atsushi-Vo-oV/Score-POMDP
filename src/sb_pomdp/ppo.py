"""PPO updates with differentiable replay of the recursive particle belief."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint

from .belief import EnergyBelief, PotentialBranch
from .buffer import RolloutBatch
from .policies import ScoreBeliefActorCritic
from .value_transform import ValueTransform


@dataclass(slots=True)
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    initial_energy_gradient_norm: float
    observation_energy_gradient_norm: float
    transition_energy_gradient_norm: float


def _module_gradient_norm(
    module: torch.nn.Module | None,
    *,
    reference: torch.nn.Module,
) -> torch.Tensor:
    """Return the pre-clipping L2 norm of all available module gradients."""

    squared_norms = [
        parameter.grad.detach().float().square().sum()
        for parameter in (() if module is None else module.parameters())
        if parameter.grad is not None
    ]
    if not squared_norms:
        parameter = next(reference.parameters())
        return torch.zeros((), device=parameter.device, dtype=torch.float32)
    return torch.stack(squared_norms).sum().sqrt()


def diffusion_advantage_weights(
    num_steps: int,
    discount: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Weights for a saved chain ordered ``x^N -> ... -> x^0``.

    The first transition is the noisiest and receives ``discount**(N-1)``;
    the final transition receives one.
    """

    if num_steps <= 0 or not 0.0 <= discount <= 1.0:
        raise ValueError("num_steps must be positive and discount must be in [0, 1]")
    exponents = torch.arange(num_steps - 1, -1, -1, device=device, dtype=dtype)
    return discount**exponents


def _clipped_policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_coefficient, 1.0 + clip_coefficient) * advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_coefficient).float().mean()
    return loss, approximate_kl, clip_fraction


def _time_environment_flatten(value: torch.Tensor) -> torch.Tensor:
    """Flatten a selected ``[time, environment, ...]`` sequence minibatch."""

    return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])


def _replay_belief_step(
    model: ScoreBeliefActorCritic,
    observation: torch.Tensor,
    previous_particles: torch.Tensor,
    previous_action: torch.Tensor,
    initial: torch.Tensor,
    initial_noise: torch.Tensor,
    step_noises: torch.Tensor,
    *,
    step_size: float,
    gradient_mode: str,
    track_grad: bool,
    potential_branch: PotentialBranch | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay one time step, checkpointing the long full-BPTT graph."""

    # Classify before entering the checkpointed function.  Backward may replay
    # ``particle_update``, but it must neither synchronise the CUDA stream again
    # nor build an inactive energy branch during that recomputation.
    energy_belief = isinstance(model.belief, EnergyBelief)
    if energy_belief and potential_branch is None:
        potential_branch = model.belief.classify_potential_branch(initial)

    def particle_update(
        step_observation: torch.Tensor,
        step_previous_particles: torch.Tensor,
        step_previous_action: torch.Tensor,
        step_initial: torch.Tensor,
        step_initial_noise: torch.Tensor,
        step_noises_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common_arguments = {
            "initial_noise": step_initial_noise,
            "langevin_noise": step_noises_value,
            "step_size": step_size,
            "track_grad": track_grad,
            "temporal_gradient_mode": gradient_mode,
        }
        if energy_belief:
            return model.belief.particles_from_noise(
                step_observation,
                step_previous_particles,
                step_previous_action,
                step_initial,
                potential_branch=potential_branch,
                **common_arguments,
            )
        return model.belief.particles_from_noise(
            step_observation,
            step_previous_particles,
            step_previous_action,
            step_initial,
            **common_arguments,
        )

    if track_grad and gradient_mode == "full":
        # Non-reentrant checkpointing supports the autograd.grad calls used to
        # construct energy scores and preserves the exact forward computation.
        return checkpoint(
            particle_update,
            observation,
            previous_particles,
            previous_action,
            initial,
            initial_noise,
            step_noises,
            use_reentrant=False,
        )
    return particle_update(
        observation,
        previous_particles,
        previous_action,
        initial,
        initial_noise,
        step_noises,
    )


def replay_belief_sequence(
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
        )
        replayed_particles.append(particles)
        replayed_scores.append(scores)
        previous_particles = particles

    return torch.stack(replayed_particles), torch.stack(replayed_scores)


def _policy_objective(
    model: ScoreBeliefActorCritic,
    condition: torch.Tensor,
    rollout: RolloutBatch,
    environment_indices: torch.Tensor,
    advantages: torch.Tensor,
    model_config: dict,
    ppo_config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the saved actions for one sequence minibatch."""

    old_log_prob = _time_environment_flatten(rollout.old_log_prob[:, environment_indices])
    action = _time_environment_flatten(rollout.actions[:, environment_indices])
    minibatch_advantage = _time_environment_flatten(advantages[:, environment_indices])

    if model.action_kind == "discrete":
        assert model.categorical_head is not None
        distribution = Categorical(logits=model.categorical_head(condition))
        new_log_prob = distribution.log_prob(action.long())
        entropy = distribution.entropy().mean()
        policy_loss, approximate_kl, clip_fraction = _clipped_policy_loss(
            new_log_prob,
            old_log_prob,
            minibatch_advantage,
            float(ppo_config["clip_coef"]),
        )
    elif model.continuous_policy_kind == "diffusion":
        assert model.diffusion_policy is not None
        assert rollout.diffusion_chains is not None
        chains = _time_environment_flatten(rollout.diffusion_chains[:, environment_indices])
        new_log_prob, step_entropy = model.diffusion_policy.log_prob_chain(condition, chains)
        diffusion_steps = new_log_prob.shape[1]
        step_weights = diffusion_advantage_weights(
            diffusion_steps,
            float(model_config["diffusion_advantage_discount"]),
            device=new_log_prob.device,
            dtype=new_log_prob.dtype,
        )
        expanded_advantage = minibatch_advantage[:, None] * step_weights[None, :]
        entropy = step_entropy.mean()
        policy_loss, approximate_kl, clip_fraction = _clipped_policy_loss(
            new_log_prob,
            old_log_prob,
            expanded_advantage,
            float(ppo_config["clip_coef"]),
        )
    else:
        assert model.gaussian_policy is not None
        assert rollout.pre_tanh_actions is not None
        pre_tanh_actions = _time_environment_flatten(
            rollout.pre_tanh_actions[:, environment_indices]
        )
        new_log_prob, gaussian_entropy = model.gaussian_policy.log_prob(
            condition,
            pre_tanh_action=pre_tanh_actions,
        )
        entropy = gaussian_entropy.mean()
        policy_loss, approximate_kl, clip_fraction = _clipped_policy_loss(
            new_log_prob,
            old_log_prob,
            minibatch_advantage,
            float(ppo_config["clip_coef"]),
        )
    return policy_loss, approximate_kl, clip_fraction, entropy


def update_ppo(
    model: ScoreBeliefActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    model_config: dict,
    ppo_config: dict,
    *,
    generator: torch.Generator | None = None,
) -> PPOMetrics:
    """Update all model components through ULA and the recurrent belief.

    Environment trajectories are the fixed on-policy samples required by PPO.
    Only the differentiable, model-free belief computation is reparameterized
    and replayed.  Sequence minibatches are required because randomly shuffled
    time steps would sever the recurrent gradient.
    """

    rollout_steps, num_environments = rollout.rollout_shape
    configured_minibatch_size = int(ppo_config["minibatch_size"])
    sequence_microbatch_size = int(
        ppo_config.get("sequence_microbatch_size", configured_minibatch_size)
    )
    if configured_minibatch_size < rollout_steps:
        raise ValueError(
            "ppo.minibatch_size must be at least ppo.rollout_steps for belief sequence replay"
        )
    if configured_minibatch_size % rollout_steps:
        raise ValueError(
            "ppo.minibatch_size must be divisible by ppo.rollout_steps for belief sequence replay"
        )
    if sequence_microbatch_size < rollout_steps:
        raise ValueError("ppo.sequence_microbatch_size must be at least ppo.rollout_steps")
    if sequence_microbatch_size > configured_minibatch_size:
        raise ValueError("ppo.sequence_microbatch_size must not exceed ppo.minibatch_size")
    if sequence_microbatch_size % rollout_steps:
        raise ValueError("ppo.sequence_microbatch_size must be divisible by ppo.rollout_steps")
    if configured_minibatch_size % sequence_microbatch_size:
        raise ValueError("ppo.minibatch_size must be divisible by ppo.sequence_microbatch_size")
    environments_per_minibatch = min(
        configured_minibatch_size // rollout_steps,
        num_environments,
    )
    environments_per_microbatch = sequence_microbatch_size // rollout_steps

    value_transform = ValueTransform.from_config(ppo_config)
    advantages = rollout.advantages
    if ppo_config["normalize_advantage"]:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approximate_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
        "initial_energy_gradient_norm": 0.0,
        "observation_energy_gradient_norm": 0.0,
        "transition_energy_gradient_norm": 0.0,
    }
    batches = 0

    for _ in range(int(ppo_config["epochs"])):
        permutation = torch.randperm(
            num_environments,
            device=advantages.device,
            generator=generator,
        )
        for start in range(0, num_environments, environments_per_minibatch):
            minibatch_indices = permutation[start : start + environments_per_minibatch]
            minibatch_environments = int(minibatch_indices.numel())
            minibatch_metrics = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approximate_kl": 0.0,
                "clip_fraction": 0.0,
            }
            optimizer.zero_grad(set_to_none=True)

            # Environment trajectories are independent.  Splitting only this
            # dimension lowers peak memory without truncating time or changing
            # the effective PPO minibatch/optimizer update.
            for micro_start in range(0, minibatch_environments, environments_per_microbatch):
                environment_indices = minibatch_indices[
                    micro_start : micro_start + environments_per_microbatch
                ]
                microbatch_weight = int(environment_indices.numel()) / minibatch_environments
                particles, scores = replay_belief_sequence(
                    model,
                    rollout,
                    environment_indices,
                    model_config,
                )
                condition = model.encode(
                    _time_environment_flatten(particles),
                    _time_environment_flatten(scores),
                )
                new_values = model.value(condition)
                policy_loss, approximate_kl, clip_fraction, entropy = _policy_objective(
                    model,
                    condition,
                    rollout,
                    environment_indices,
                    advantages,
                    model_config,
                    ppo_config,
                )
                returns = _time_environment_flatten(rollout.returns[:, environment_indices])
                value_loss = 0.5 * (new_values - value_transform.to_target(returns)).square().mean()
                total_loss = (
                    policy_loss
                    + float(ppo_config["value_coef"]) * value_loss
                    - float(ppo_config["entropy_coef"]) * entropy
                )
                (total_loss * microbatch_weight).backward()

                microbatch_values = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                }
                for name, value in microbatch_values.items():
                    minibatch_metrics[name] += float(value.detach().cpu()) * microbatch_weight

                # Do not retain the last microbatch graph while constructing
                # the next one; backward has already accumulated its gradients.
                del (
                    particles,
                    scores,
                    condition,
                    new_values,
                    policy_loss,
                    approximate_kl,
                    clip_fraction,
                    entropy,
                    returns,
                    value_loss,
                    total_loss,
                    microbatch_values,
                )

            energy_gradient_norms = {
                "initial_energy_gradient_norm": _module_gradient_norm(
                    getattr(model.belief, "initial_energy", None),
                    reference=model,
                ),
                "observation_energy_gradient_norm": _module_gradient_norm(
                    getattr(model.belief, "observation_energy", None),
                    reference=model,
                ),
                "transition_energy_gradient_norm": _module_gradient_norm(
                    getattr(model.belief, "transition_energy", None),
                    reference=model,
                ),
            }
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(ppo_config["max_grad_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()

            for name, value in minibatch_metrics.items():
                totals[name] += value
            totals["gradient_norm"] += float(gradient_norm.detach().cpu())
            for name, value in energy_gradient_norms.items():
                totals[name] += float(value.detach().cpu())
            batches += 1

    if not batches:
        raise RuntimeError("PPO update produced no minibatches")
    return PPOMetrics(**{name: value / batches for name, value in totals.items()})
