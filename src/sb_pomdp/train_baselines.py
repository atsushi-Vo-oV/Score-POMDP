"""PPO training for the feed-forward, recurrent, and oracle baselines.

The comparison baselines deliberately share the rollout, evaluation, and
artifact contract of :mod:`sb_pomdp.train`.  Recurrent PPO replays complete
rollout sequences plus the active-episode prefix.  It therefore supports the
same ``full`` versus one-environment-step ``tbptt_1`` temporal-gradient
comparison as the score-belief model.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .artifacts import SeedArtifacts
from .baselines import (
    RECURRENT_BASELINE_TYPES,
    FeedForwardBaselineActorCritic,
    RecurrentBaseline,
    TemporalGradientMode,
    make_baseline_actor_critic,
)
from .buffer import generalised_advantage_estimate
from .config import ExperimentConfig
from .envs import make_env
from .ppo import PPOMetrics, diffusion_advantage_weights
from .train import (
    EPISODE_FIELDS,
    EVALUATION_FIELDS,
    LOGGER,
    METRIC_FIELDS,
    SyncEnvironmentBatch,
    _checkpoint_payload,
    _checkpoint_tensor,
    _environment_actions,
    _finite_numeric_list,
    _load_resume_checkpoint,
    _restore_global_rng_states,
    _truncate_uncommitted_csv_rows,
    configure_logging,
    encode_action,
    final_step_features,
    resolve_device,
    runtime_metadata,
    save_checkpoint,
    seed_everything,
    truncated_step_indices,
)
from .value_transform import ValueTransform

BaselineMethod = Literal["observation_mlp", "gru", "rnn", "oracle_state", "particle_filter"]
BASELINE_METHODS = frozenset({"observation_mlp", "gru", "rnn", "oracle_state", "particle_filter"})
# The baseline trainer keeps its own checkpoint dialect.  Its trainer_state has a
# different shape from the score trainer's, so the two identifiers must never
# collide: feeding either trainer the other's checkpoint has to fail loudly
# instead of restoring a partially compatible state.
BASELINE_CHECKPOINT_FORMAT_VERSION = "baseline-v1"
BASELINE_TRAINER_STATE_FORMAT_VERSION = 1


@dataclass(slots=True)
class RecurrentReplayPrefix:
    """Realised GRU inputs from one active episode before a rollout."""

    inputs: torch.Tensor
    episode_starts: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.inputs.shape[0])


@dataclass(slots=True)
class RecurrentReplayHistory:
    """Realised inputs retained from the start of one active episode."""

    inputs: list[torch.Tensor] = dataclass_field(default_factory=list)
    episode_starts: list[torch.Tensor] = dataclass_field(default_factory=list)

    def append(
        self,
        inputs: torch.Tensor,
        episode_starts: torch.Tensor,
        environment_index: int,
    ) -> None:
        self.inputs.append(inputs[environment_index].detach().clone())
        self.episode_starts.append(episode_starts[environment_index].detach().clone())

    def clear(self) -> None:
        self.inputs.clear()
        self.episode_starts.clear()

    def snapshot(
        self,
        *,
        input_template: torch.Tensor,
        start_template: torch.Tensor,
    ) -> RecurrentReplayPrefix:
        if self.inputs:
            return RecurrentReplayPrefix(
                inputs=torch.stack(self.inputs),
                episode_starts=torch.stack(self.episode_starts),
            )
        return RecurrentReplayPrefix(
            inputs=input_template.new_empty((0, input_template.shape[-1])),
            episode_starts=start_template.new_empty((0,)),
        )

    def state_dict(self) -> dict[str, list[torch.Tensor]]:
        """Return the CPU tensors a resumable checkpoint stores for this episode."""

        return {
            "inputs": [value.detach().cpu().clone() for value in self.inputs],
            "episode_starts": [value.detach().cpu().clone() for value in self.episode_starts],
        }


@dataclass(slots=True)
class BaselineRollout:
    """One time-major baseline rollout kept on the training device."""

    inputs: torch.Tensor
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    episode_starts: torch.Tensor
    pre_tanh_actions: torch.Tensor | None
    diffusion_chains: torch.Tensor | None
    hidden_states: torch.Tensor | None
    recurrent_prefixes: tuple[RecurrentReplayPrefix, ...] = ()

    @property
    def rollout_shape(self) -> tuple[int, int]:
        return int(self.rewards.shape[0]), int(self.rewards.shape[1])

    @property
    def size(self) -> int:
        time_steps, environments = self.rollout_shape
        return time_steps * environments


def _make_model(
    environment: Any,
    method: BaselineMethod,
    model_config: dict[str, Any],
    comparison_config: dict[str, Any],
    device: torch.device,
) -> FeedForwardBaselineActorCritic | RecurrentBaseline:
    spec = environment.action_spec
    input_dim = environment.observation_dim
    kind: Literal["observation", "oracle_state", "gru", "rnn", "particle_filter"]
    if method == "observation_mlp":
        kind = "observation"
    elif method == "oracle_state":
        kind = "oracle_state"
    elif method == "particle_filter":
        kind = "particle_filter"
        input_dim += spec.feature_dim
    elif method == "rnn":
        kind = "rnn"
        input_dim += spec.feature_dim
    else:
        kind = "gru"
        input_dim += spec.feature_dim

    shared_head_kwargs = {
        "policy_condition_dim": int(model_config["d_model"]),
        "head_hidden_dims": model_config["policy_hidden"],
        "continuous_policy_kind": model_config["continuous_policy_kind"],
        "diffusion_steps": int(model_config["diffusion_steps"]),
        "diffusion_beta_start": float(model_config["diffusion_beta_start"]),
        "diffusion_beta_end": float(model_config["diffusion_beta_end"]),
        "diffusion_min_std": float(model_config["diffusion_min_std"]),
    }
    recurrent_kwargs: dict[str, Any] = {}
    hidden_dims = model_config["policy_hidden"]
    if method == "gru":
        hidden_dims = comparison_config["gru_encoder_hidden"]
        recurrent_kwargs = {
            "recurrent_hidden_dim": int(comparison_config["gru_hidden_dim"]),
            **shared_head_kwargs,
        }
    elif method == "rnn":
        hidden_dims = comparison_config["rnn_encoder_hidden"]
        recurrent_kwargs = {
            "recurrent_hidden_dim": int(comparison_config["rnn_hidden_dim"]),
            **shared_head_kwargs,
        }
    elif method == "particle_filter":
        hidden_dims = comparison_config["pf_hidden"]
        recurrent_kwargs = {
            "pf_num_particles": int(model_config["num_particles"]),
            "pf_particle_dim": int(comparison_config["pf_particle_dim"]),
            "pf_soft_alpha": float(comparison_config["pf_soft_alpha"]),
            **shared_head_kwargs,
        }

    model = make_baseline_actor_critic(
        kind,
        observation_dim=input_dim,
        state_dim=environment.oracle_dim,
        action_kind=spec.kind,
        hidden_dims=hidden_dims,
        discrete_actions=spec.n,
        action_low=None if spec.low is None else spec.low.tolist(),
        action_high=None if spec.high is None else spec.high.tolist(),
        **recurrent_kwargs,
    )
    return model.to(device)


def _oracle_batch(batch: SyncEnvironmentBatch, device: torch.device) -> torch.Tensor:
    features = np.stack([environment.oracle_features() for environment in batch.environments])
    return torch.as_tensor(features, dtype=torch.float32, device=device)


def _current_inputs(
    method: BaselineMethod,
    observation: np.ndarray,
    batch: SyncEnvironmentBatch,
    previous_action: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if method == "oracle_state":
        return _oracle_batch(batch, device)
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    if method in ("gru", "rnn", "particle_filter"):
        return torch.cat((observation_tensor, previous_action), dim=-1)
    return observation_tensor


def _truncation_bootstrap_values(
    model: FeedForwardBaselineActorCritic | RecurrentBaseline,
    method: BaselineMethod,
    *,
    infos: Sequence[Mapping[str, Any]],
    hidden: torch.Tensor | None,
    action_features: torch.Tensor,
    device: torch.device,
    value_transform: ValueTransform | None = None,
) -> torch.Tensor:
    """Value the post-step state that a time-limit truncation discarded.

    This mirrors :func:`sb_pomdp.train.truncation_bootstrap_values` step for
    step so both arms share one protocol.  The recurrent baseline advances a
    *clone* of the per-environment hidden state by exactly one step over the
    genuine final observation and the executed action, under ``no_grad`` and
    without touching the online hidden state or the replay histories; the
    feed-forward baselines simply value their final input.  The returned
    ``[batch]`` tensor is zero wherever the step did not truncate, so a true
    terminal state never bootstraps.
    """

    values = torch.zeros(len(infos), dtype=torch.float32, device=device)
    indices = truncated_step_indices(infos)
    if not indices:
        return values
    selector = torch.as_tensor(indices, dtype=torch.long, device=device)
    if method == "oracle_state":
        inputs = final_step_features(
            infos, indices, name="final_oracle_features", device=device
        )
    else:
        inputs = final_step_features(infos, indices, name="final_observation", device=device)
        if method in ("gru", "rnn", "particle_filter"):
            inputs = torch.cat(
                (inputs, action_features.detach().index_select(0, selector)), dim=-1
            )
    with torch.no_grad():
        if isinstance(model, RECURRENT_BASELINE_TYPES):
            if hidden is None:
                raise RuntimeError("the recurrent baseline is missing its rollout hidden state")
            final_hidden = hidden.detach().index_select(1, selector).clone()
            features, _ = model.step(
                inputs,
                final_hidden,
                torch.zeros(len(indices), dtype=torch.bool, device=device),
            )
            bootstrap = model.heads.value(features)
        else:
            bootstrap = model.value(inputs)
    bootstrap = (value_transform or ValueTransform()).to_raw(bootstrap)
    values.index_copy_(0, selector, bootstrap.detach().to(dtype=values.dtype))
    return values


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
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_coefficient).float().mean()
    return policy_loss, approximate_kl, clip_fraction


def _normalise_advantages(advantages: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def _empty_metric_totals() -> dict[str, float]:
    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approximate_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
    }


def _accumulate_metrics(
    totals: dict[str, float],
    *,
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    entropy: torch.Tensor,
    approximate_kl: torch.Tensor,
    clip_fraction: torch.Tensor,
    gradient_norm: torch.Tensor,
) -> None:
    values = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approximate_kl": approximate_kl,
        "clip_fraction": clip_fraction,
        "gradient_norm": gradient_norm,
    }
    for name, value in values.items():
        totals[name] += float(value.detach().cpu())


def _finish_metrics(
    totals: dict[str, float],
    batches: int,
) -> PPOMetrics:
    if batches <= 0:
        raise RuntimeError("baseline PPO update produced no minibatches")
    return PPOMetrics(
        **{name: value / batches for name, value in totals.items()},
        initial_energy_gradient_norm=float("nan"),
        observation_energy_gradient_norm=float("nan"),
        transition_energy_gradient_norm=float("nan"),
    )


def _optimise_minibatch(
    model: FeedForwardBaselineActorCritic | RecurrentBaseline,
    optimizer: torch.optim.Optimizer,
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy_values: torch.Tensor,
    ppo_config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_loss, approximate_kl, clip_fraction = _clipped_policy_loss(
        new_log_prob,
        old_log_prob,
        advantages,
        float(ppo_config["clip_coef"]),
    )
    entropy = entropy_values.mean()
    value_targets = ValueTransform.from_config(ppo_config).to_target(returns)
    value_loss = 0.5 * (new_values - value_targets).square().mean()
    total_loss = (
        policy_loss
        + float(ppo_config["value_coef"]) * value_loss
        - float(ppo_config["entropy_coef"]) * entropy
    )
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(ppo_config["max_grad_norm"]),
        error_if_nonfinite=True,
    )
    optimizer.step()
    return policy_loss, value_loss, entropy, approximate_kl, clip_fraction, gradient_norm


def _update_feedforward(
    model: FeedForwardBaselineActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: BaselineRollout,
    ppo_config: dict[str, Any],
) -> PPOMetrics:
    size = rollout.size
    minibatch_size = min(int(ppo_config["minibatch_size"]), size)

    def flatten(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(size, *value.shape[2:])

    inputs = flatten(rollout.inputs)
    actions = flatten(rollout.actions)
    old_log_prob = flatten(rollout.old_log_prob)
    returns = flatten(rollout.returns)
    advantages = _normalise_advantages(
        flatten(rollout.advantages), bool(ppo_config["normalize_advantage"])
    )
    pre_tanh_actions = (
        None if rollout.pre_tanh_actions is None else flatten(rollout.pre_tanh_actions)
    )
    totals = _empty_metric_totals()
    batches = 0

    for _ in range(int(ppo_config["epochs"])):
        permutation = torch.randperm(size, device=inputs.device)
        for start in range(0, size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            evaluation = model.evaluate_actions(
                inputs[indices],
                actions[indices],
                pre_tanh_actions=(
                    None if pre_tanh_actions is None else pre_tanh_actions[indices]
                ),
            )
            values = _optimise_minibatch(
                model,
                optimizer,
                new_log_prob=evaluation.log_prob,
                old_log_prob=old_log_prob[indices],
                advantages=advantages[indices],
                new_values=evaluation.value,
                returns=returns[indices],
                entropy_values=evaluation.entropy,
                ppo_config=ppo_config,
            )
            _accumulate_metrics(
                totals,
                policy_loss=values[0],
                value_loss=values[1],
                entropy=values[2],
                approximate_kl=values[3],
                clip_fraction=values[4],
                gradient_norm=values[5],
            )
            batches += 1
    return _finish_metrics(totals, batches)


def _replay_recurrent_prefix(
    model: RecurrentBaseline,
    prefix: RecurrentReplayPrefix,
    *,
    temporal_gradient_mode: TemporalGradientMode,
) -> torch.Tensor:
    """Replay one active-episode prefix with the current GRU parameters."""

    hidden = model.initial_hidden(1, device=prefix.inputs.device, dtype=prefix.inputs.dtype)
    if not prefix.length:
        return hidden
    if prefix.inputs.ndim != 2 or prefix.inputs.shape[-1] != model.observation_dim:
        raise ValueError("a recurrent prefix has an invalid input shape")
    if tuple(prefix.episode_starts.shape) != (prefix.length,):
        raise ValueError("a recurrent prefix has an invalid episode-start shape")
    if not bool(prefix.episode_starts[0]) or bool(prefix.episode_starts[1:].any()):
        raise ValueError("a recurrent prefix must contain exactly one episode start at index zero")

    if temporal_gradient_mode == "tbptt_1":
        # Prefix features have no PPO loss.  Only its final numerical hidden
        # state is required, and the current rollout must not differentiate
        # through any earlier environment step in one-step TBPTT mode.
        with torch.no_grad():
            _, hidden = model.encode_sequence(
                prefix.inputs[:, None],
                hidden,
                prefix.episode_starts[:, None],
                temporal_gradient_mode=temporal_gradient_mode,
            )
        return hidden

    _, hidden = model.encode_sequence(
        prefix.inputs[:, None],
        hidden,
        prefix.episode_starts[:, None],
        temporal_gradient_mode=temporal_gradient_mode,
    )
    return hidden


def replay_recurrent_sequence(
    model: RecurrentBaseline,
    rollout: BaselineRollout,
    environment_indices: torch.Tensor,
    *,
    temporal_gradient_mode: TemporalGradientMode,
) -> torch.Tensor:
    """Replay active prefixes and a complete rollout without cutting time.

    Realised observations and previous actions are fixed PPO data.  In
    ``full`` mode, every output retains the recurrent graph back to its episode
    start, including across rollout boundaries.  ``tbptt_1`` produces the same
    hidden-state forward values while detaching before every GRU transition.
    """

    if temporal_gradient_mode not in {"full", "tbptt_1"}:
        raise ValueError("temporal_gradient_mode must be 'full' or 'tbptt_1'")
    _, num_envs = rollout.rollout_shape
    if len(rollout.recurrent_prefixes) != num_envs:
        raise ValueError("one recurrent replay prefix is required per rollout environment")

    initial_hiddens: list[torch.Tensor] = []
    for environment_index in environment_indices.detach().cpu().tolist():
        prefix = rollout.recurrent_prefixes[environment_index]
        if not prefix.length and not bool(rollout.episode_starts[0, environment_index]):
            raise RuntimeError("a non-initial recurrent rollout is missing its episode prefix")
        initial_hiddens.append(
            _replay_recurrent_prefix(
                model,
                prefix,
                temporal_gradient_mode=temporal_gradient_mode,
            )
        )

    initial_hidden = torch.cat(initial_hiddens, dim=1)
    features, _ = model.encode_sequence(
        rollout.inputs[:, environment_indices],
        initial_hidden,
        rollout.episode_starts[:, environment_indices],
        temporal_gradient_mode=temporal_gradient_mode,
    )
    return features


@torch.no_grad()
def _recondition_recurrent_hidden(
    model: RecurrentBaseline,
    histories: list[RecurrentReplayHistory],
    *,
    input_template: torch.Tensor,
    start_template: torch.Tensor,
) -> torch.Tensor:
    """Rebuild online hidden states after PPO changes the GRU parameters."""

    hidden_states = [
        _replay_recurrent_prefix(
            model,
            history.snapshot(
                input_template=input_template,
                start_template=start_template,
            ),
            temporal_gradient_mode="tbptt_1",
        )
        for history in histories
    ]
    return torch.cat(hidden_states, dim=1)


def _update_recurrent(
    model: RecurrentBaseline,
    optimizer: torch.optim.Optimizer,
    rollout: BaselineRollout,
    model_config: dict[str, Any],
    ppo_config: dict[str, Any],
    *,
    temporal_gradient_mode: TemporalGradientMode,
) -> PPOMetrics:
    time_steps, num_envs = rollout.rollout_shape
    configured_minibatch_size = int(ppo_config["minibatch_size"])
    if configured_minibatch_size < time_steps:
        raise ValueError(
            "ppo.minibatch_size must be at least ppo.rollout_steps for recurrent replay"
        )
    if configured_minibatch_size % time_steps:
        raise ValueError(
            "ppo.minibatch_size must be divisible by ppo.rollout_steps for recurrent replay"
        )
    environments_per_minibatch = min(configured_minibatch_size // time_steps, num_envs)

    advantages = _normalise_advantages(rollout.advantages, bool(ppo_config["normalize_advantage"]))
    totals = _empty_metric_totals()
    batches = 0

    for _ in range(int(ppo_config["epochs"])):
        permutation = torch.randperm(num_envs, device=rollout.inputs.device)
        processed_transitions = 0
        for start_index in range(0, num_envs, environments_per_minibatch):
            selected = permutation[start_index : start_index + environments_per_minibatch]
            features = replay_recurrent_sequence(
                model,
                rollout,
                selected,
                temporal_gradient_mode=temporal_gradient_mode,
            )
            actions = rollout.actions[:, selected]
            old_log_prob = rollout.old_log_prob[:, selected]
            minibatch_advantages = advantages[:, selected]
            returns = rollout.returns[:, selected]
            pre_tanh_actions = (
                None if rollout.pre_tanh_actions is None else rollout.pre_tanh_actions[:, selected]
            )
            diffusion_chains = (
                None
                if rollout.diffusion_chains is None
                else rollout.diffusion_chains[:, selected]
            )

            # Flatten only after recurrent encoding, retaining each selected
            # environment's complete time graph.
            transitions = time_steps * int(selected.numel())
            flat_features = features.reshape(transitions, features.shape[-1])
            flat_actions = actions.reshape(transitions, *actions.shape[2:])
            flat_pre_tanh = (
                None
                if pre_tanh_actions is None
                else pre_tanh_actions.reshape(transitions, *pre_tanh_actions.shape[2:])
            )
            flat_diffusion_chain = (
                None
                if diffusion_chains is None
                else diffusion_chains.reshape(
                    transitions,
                    *diffusion_chains.shape[2:],
                )
            )
            evaluation = model.heads.evaluate_actions(
                flat_features,
                flat_actions,
                pre_tanh_actions=flat_pre_tanh,
                diffusion_chains=flat_diffusion_chain,
            )
            flat_advantages = minibatch_advantages.reshape(transitions)
            flat_old_log_prob = old_log_prob.reshape(
                transitions,
                *old_log_prob.shape[2:],
            )
            if model.action_kind == "continuous" and model.continuous_policy_kind == "diffusion":
                diffusion_steps = evaluation.log_prob.shape[-1]
                weights = diffusion_advantage_weights(
                    diffusion_steps,
                    float(model_config["diffusion_advantage_discount"]),
                    device=evaluation.log_prob.device,
                    dtype=evaluation.log_prob.dtype,
                )
                flat_advantages = flat_advantages[:, None] * weights[None, :]
            values = _optimise_minibatch(
                model,
                optimizer,
                new_log_prob=evaluation.log_prob,
                old_log_prob=flat_old_log_prob,
                advantages=flat_advantages,
                new_values=evaluation.value,
                returns=returns.reshape(transitions),
                entropy_values=evaluation.entropy,
                ppo_config=ppo_config,
            )
            _accumulate_metrics(
                totals,
                policy_loss=values[0],
                value_loss=values[1],
                entropy=values[2],
                approximate_kl=values[3],
                clip_fraction=values[4],
                gradient_norm=values[5],
            )
            batches += 1
            processed_transitions += transitions
        if processed_transitions != rollout.size:
            raise RuntimeError(
                "recurrent PPO did not consume exactly one rollout per epoch: "
                f"processed={processed_transitions}, expected={rollout.size}"
            )
    return _finish_metrics(totals, batches)


@torch.no_grad()
def evaluate_baseline(
    model: FeedForwardBaselineActorCritic | RecurrentBaseline,
    *,
    method: BaselineMethod,
    task: str,
    environment_config: dict[str, Any],
    episodes: int,
    seed: int,
    device: torch.device,
    deterministic: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate on fixed held-out seeds and preserve the caller's RNG state."""

    was_training = model.training
    model.eval()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    returns: list[float] = []
    lengths: list[int] = []
    saturation_fractions: list[float] = []
    arrays: dict[str, np.ndarray] = {}

    try:
        for episode in range(episodes):
            environment = make_env(task, environment_config, seed=seed + episode)
            observation, info = environment.reset(seed=seed + episode)
            previous_action = torch.zeros(
                1,
                environment.action_spec.feature_dim,
                dtype=torch.float32,
                device=device,
            )
            hidden = (
                model.initial_hidden(1, device=device)
                if isinstance(model, RECURRENT_BASELINE_TYPES)
                else None
            )
            episode_start = torch.ones(1, dtype=torch.bool, device=device)
            episode_observations: list[np.ndarray] = []
            episode_actions: list[np.ndarray] = []
            episode_rewards: list[float] = []
            episode_states: list[np.ndarray] = []
            episode_return = 0.0

            while True:
                observation_tensor = torch.as_tensor(
                    observation[None], dtype=torch.float32, device=device
                )
                if method == "oracle_state":
                    inputs = torch.as_tensor(
                        environment.oracle_features()[None],
                        dtype=torch.float32,
                        device=device,
                    )
                elif method in ("gru", "rnn", "particle_filter"):
                    inputs = torch.cat((observation_tensor, previous_action), dim=-1)
                else:
                    inputs = observation_tensor

                if isinstance(model, RECURRENT_BASELINE_TYPES):
                    assert hidden is not None
                    sample, _, hidden = model.act_step(
                        inputs,
                        hidden,
                        episode_start,
                        deterministic=deterministic,
                        generator=generator,
                    )
                else:
                    sample, _ = model.act(
                        inputs,
                        deterministic=deterministic,
                        generator=generator,
                    )
                environment_action = _environment_actions(environment.action_spec, sample.action)[0]
                episode_observations.append(np.asarray(observation, dtype=np.float32))
                episode_actions.append(np.asarray(environment_action).reshape(-1))
                episode_states.append(np.asarray(info["state"], dtype=np.float32))
                observation, reward, terminated, truncated, info = environment.step(
                    environment_action
                )
                episode_rewards.append(float(reward))
                episode_return += float(reward)
                previous_action = encode_action(environment.action_spec, sample.action)
                if environment.action_spec.is_continuous:
                    saturation_fractions.append(
                        float((previous_action.abs() >= 0.99).float().mean().cpu())
                    )
                episode_start = torch.zeros(1, dtype=torch.bool, device=device)
                if terminated or truncated:
                    break

            prefix = f"episode_{episode:03d}"
            arrays[f"{prefix}_observations"] = np.asarray(episode_observations, dtype=np.float32)
            arrays[f"{prefix}_actions"] = np.asarray(episode_actions, dtype=np.float32)
            arrays[f"{prefix}_rewards"] = np.asarray(episode_rewards, dtype=np.float32)
            arrays[f"{prefix}_states"] = np.asarray(episode_states, dtype=np.float32)
            returns.append(episode_return)
            lengths.append(len(episode_rewards))
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        model.train(was_training)

    return (
        {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "action_saturation_fraction": (
                float(np.mean(saturation_fractions)) if saturation_fractions else float("nan")
            ),
        },
        arrays,
    )


def _baseline_trainer_state_payload(
    *,
    environments: list[Any],
    observation: np.ndarray,
    previous_action: torch.Tensor,
    episode_starts: torch.Tensor,
    recurrent_histories: list[RecurrentReplayHistory],
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
    completed_returns: list[float],
    completed_lengths: list[int],
    metric_rows: list[dict[str, Any]],
    best_evaluation: float,
    latest_evaluation: dict[str, float] | None,
    wall_time_seconds: float,
) -> dict[str, Any]:
    """Return every mutable trainer value a later baseline segment has to inherit.

    The online recurrent hidden state is deliberately absent: at an update
    boundary it is exactly what :func:`_recondition_recurrent_hidden` produces
    from the retained episode prefixes and the committed parameters, so the
    resumed segment rebuilds it through the same code path the uninterrupted run
    uses rather than trusting a second, independently stored copy.
    """

    return {
        "format_version": BASELINE_TRAINER_STATE_FORMAT_VERSION,
        "environments": [environment.state_dict() for environment in environments],
        "observation": torch.as_tensor(observation, dtype=torch.float32).cpu().clone(),
        "previous_action": previous_action.detach().cpu().clone(),
        "episode_starts": episode_starts.detach().cpu().clone(),
        "recurrent_histories": [history.state_dict() for history in recurrent_histories],
        "episode_returns": episode_returns.copy(),
        "episode_lengths": episode_lengths.copy(),
        "completed_returns": list(completed_returns),
        "completed_lengths": list(completed_lengths),
        "metric_rows": [dict(row) for row in metric_rows],
        "best_evaluation": float(best_evaluation),
        "latest_evaluation": (
            None if latest_evaluation is None else dict(latest_evaluation)
        ),
        "wall_time_seconds": float(wall_time_seconds),
    }


def _restore_recurrent_history(
    state_dict: Any,
    *,
    history_index: int,
    input_dim: int,
    device: torch.device,
) -> RecurrentReplayHistory:
    """Rebuild one active-episode prefix, rejecting any malformed stored tensor."""

    if not isinstance(state_dict, Mapping):
        raise TypeError(f"checkpoint recurrent history {history_index} must be a mapping")
    fields_and_shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "inputs": ((input_dim,), torch.float32),
        "episode_starts": ((), torch.bool),
    }
    if set(state_dict) != set(fields_and_shapes):
        raise ValueError(f"checkpoint recurrent history {history_index} has invalid fields")
    lengths: set[int] = set()
    restored: dict[str, list[torch.Tensor]] = {}
    for field_name, (shape, dtype) in fields_and_shapes.items():
        values = state_dict[field_name]
        if not isinstance(values, list):
            raise TypeError(
                f"checkpoint recurrent history {history_index}.{field_name} must be a list"
            )
        lengths.add(len(values))
        restored[field_name] = [
            _checkpoint_tensor(
                value,
                name=f"recurrent_histories[{history_index}].{field_name}[{item_index}]",
                shape=shape,
                dtype=dtype,
                device=device,
            )
            for item_index, value in enumerate(values)
        ]
    if len(lengths) != 1:
        raise ValueError(
            f"checkpoint recurrent history {history_index} fields have unequal lengths"
        )
    history = RecurrentReplayHistory(**restored)
    if history.episode_starts and (
        not bool(history.episode_starts[0])
        or any(bool(value) for value in history.episode_starts[1:])
    ):
        raise ValueError(
            f"checkpoint recurrent history {history_index} must contain exactly one "
            "episode start at index zero"
        )
    return history


def _restore_baseline_trainer_state(
    state_dict: Any,
    *,
    environments: list[Any],
    update: int,
    rollout_batch_size: int,
    recurrent_input_dim: int | None,
    device: torch.device,
) -> tuple[
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
    list[RecurrentReplayHistory],
    np.ndarray,
    np.ndarray,
    list[float],
    list[int],
    list[dict[str, Any]],
    float,
    dict[str, float] | None,
    float,
]:
    """Validate one baseline ``trainer_state`` and return its restored values.

    Every stored tensor is checked for shape, dtype, and finiteness, and the
    episode accumulators, environment states, and retained prefixes are checked
    against each other, so an inconsistent checkpoint fails before it can produce
    a silently divergent continuation.  ``recurrent_input_dim`` is the GRU input
    width, or ``None`` for the feed-forward baselines, which keep no episode
    prefixes and must therefore carry an empty history list.
    """

    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "checkpoint lacks a complete trainer_state; legacy checkpoints cannot be resumed"
        )
    expected_fields = {
        "format_version",
        "environments",
        "observation",
        "previous_action",
        "episode_starts",
        "recurrent_histories",
        "episode_returns",
        "episode_lengths",
        "completed_returns",
        "completed_lengths",
        "metric_rows",
        "best_evaluation",
        "latest_evaluation",
        "wall_time_seconds",
    }
    if set(state_dict) != expected_fields:
        missing = sorted(expected_fields - set(state_dict))
        extra = sorted(set(state_dict) - expected_fields)
        raise ValueError(
            "checkpoint trainer_state is incomplete or incompatible; "
            f"missing={missing}, extra={extra}"
        )
    if state_dict["format_version"] != BASELINE_TRAINER_STATE_FORMAT_VERSION:
        raise ValueError(
            "unsupported baseline trainer_state format_version: "
            f"{state_dict['format_version']!r}"
        )

    environment_states = state_dict["environments"]
    if not isinstance(environment_states, list) or len(environment_states) != len(environments):
        raise ValueError("checkpoint has the wrong number of environment states")
    for environment, environment_state in zip(environments, environment_states):
        environment.load_state_dict(environment_state)

    first_environment = environments[0]
    batch_size = len(environments)
    observation = _checkpoint_tensor(
        state_dict["observation"],
        name="trainer_state.observation",
        shape=(batch_size, first_environment.observation_dim),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    previous_action = _checkpoint_tensor(
        state_dict["previous_action"],
        name="trainer_state.previous_action",
        shape=(batch_size, first_environment.action_spec.feature_dim),
        dtype=torch.float32,
        device=device,
    )
    episode_starts = _checkpoint_tensor(
        state_dict["episode_starts"],
        name="trainer_state.episode_starts",
        shape=(batch_size,),
        dtype=torch.bool,
        device=device,
    )

    history_states = state_dict["recurrent_histories"]
    expected_histories = 0 if recurrent_input_dim is None else batch_size
    if not isinstance(history_states, list) or len(history_states) != expected_histories:
        raise ValueError("checkpoint has the wrong number of recurrent histories")
    recurrent_histories = [
        _restore_recurrent_history(
            history_state,
            history_index=index,
            input_dim=0 if recurrent_input_dim is None else recurrent_input_dim,
            device=device,
        )
        for index, history_state in enumerate(history_states)
    ]

    episode_returns = state_dict["episode_returns"]
    if (
        not isinstance(episode_returns, np.ndarray)
        or episode_returns.shape != (batch_size,)
        or episode_returns.dtype != np.float64
        or not np.all(np.isfinite(episode_returns))
    ):
        raise ValueError("checkpoint episode_returns must be a finite float64 batch array")
    episode_lengths = state_dict["episode_lengths"]
    if (
        not isinstance(episode_lengths, np.ndarray)
        or episode_lengths.shape != (batch_size,)
        or episode_lengths.dtype != np.int64
        or np.any(episode_lengths < 0)
    ):
        raise ValueError("checkpoint episode_lengths must be a non-negative int64 batch array")
    for index, environment_state in enumerate(environment_states):
        expected_length = int(episode_lengths[index])
        if int(environment_state["elapsed_steps"]) != expected_length:
            raise ValueError(
                f"checkpoint episode length/environment state mismatch for environment {index}"
            )
        if bool(episode_starts[index]) != (expected_length == 0):
            raise ValueError(
                f"checkpoint episode start/length mismatch for environment {index}"
            )
        if recurrent_input_dim is not None and (
            len(recurrent_histories[index].inputs) != expected_length
        ):
            raise ValueError(
                f"checkpoint episode length/recurrent history mismatch for environment {index}"
            )

    completed_returns = _finite_numeric_list(
        state_dict["completed_returns"], name="completed_returns", integer=False
    )
    completed_lengths = _finite_numeric_list(
        state_dict["completed_lengths"], name="completed_lengths", integer=True
    )
    if len(completed_returns) != len(completed_lengths):
        raise ValueError("checkpoint completed episode return/length counts differ")

    raw_metric_rows = state_dict["metric_rows"]
    if not isinstance(raw_metric_rows, list) or len(raw_metric_rows) != update:
        raise ValueError("checkpoint metric_rows count does not match its update")
    metric_rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(raw_metric_rows, start=1):
        if not isinstance(raw_row, Mapping) or set(raw_row) != set(METRIC_FIELDS):
            raise ValueError(f"checkpoint metric row {row_index} has invalid fields")
        row = dict(raw_row)
        if row["update"] != row_index or row["global_step"] != row_index * rollout_batch_size:
            raise ValueError(f"checkpoint metric row {row_index} has invalid progress values")
        metric_rows.append(row)

    best_evaluation = state_dict["best_evaluation"]
    if (
        isinstance(best_evaluation, (bool, np.bool_))
        or not isinstance(best_evaluation, (int, float, np.integer, np.floating))
        or (not np.isfinite(best_evaluation) and float(best_evaluation) != -math.inf)
    ):
        raise ValueError("checkpoint best_evaluation must be finite or negative infinity")
    best_evaluation = float(best_evaluation)
    latest_evaluation = state_dict["latest_evaluation"]
    if latest_evaluation is not None:
        evaluation_fields = {
            "mean_return",
            "std_return",
            "mean_length",
            "action_saturation_fraction",
        }
        if (
            not isinstance(latest_evaluation, Mapping)
            or set(latest_evaluation) != evaluation_fields
        ):
            raise ValueError("checkpoint latest_evaluation has invalid fields")
        latest_evaluation = {key: float(latest_evaluation[key]) for key in evaluation_fields}
        if not all(
            np.isfinite(value)
            for key, value in latest_evaluation.items()
            if key != "action_saturation_fraction"
        ):
            raise ValueError("checkpoint latest_evaluation contains invalid values")

    wall_time_seconds = state_dict["wall_time_seconds"]
    if (
        isinstance(wall_time_seconds, (bool, np.bool_))
        or not isinstance(wall_time_seconds, (int, float, np.integer, np.floating))
        or not np.isfinite(wall_time_seconds)
        or float(wall_time_seconds) < 0.0
    ):
        raise ValueError("checkpoint wall_time_seconds must be finite and non-negative")
    return (
        observation.numpy(),
        previous_action,
        episode_starts,
        recurrent_histories,
        episode_returns.copy(),
        episode_lengths.copy(),
        completed_returns,
        completed_lengths,
        metric_rows,
        best_evaluation,
        latest_evaluation,
        float(wall_time_seconds),
    )


def train_baseline_single(
    config: ExperimentConfig,
    *,
    method: BaselineMethod,
    task: str,
    seed: int,
    artifacts: SeedArtifacts,
    max_updates: int | None = None,
    verbose: bool = True,
    result_dir: str,
    resume_checkpoint: str | Path | None = None,
    evaluate_at_end: bool = True,
    segment_updates: int | None = None,
) -> dict[str, Any]:
    """Train one comparison baseline with PPO and write one seed artifact tree.

    ``segment_updates`` runs at most that many further updates and commits a
    resumable checkpoint, and ``resume_checkpoint`` continues from one.  Both
    mirror :func:`sb_pomdp.train.train_single` so a baseline shard can be chained
    exactly like a score shard.
    """

    if method not in BASELINE_METHODS:
        raise ValueError(f"unsupported baseline method: {method!r}")
    if artifacts.method is not None and artifacts.method != method:
        raise ValueError(
            f"artifact method {artifacts.method!r} does not match requested method {method!r}"
        )
    if not isinstance(result_dir, str) or not result_dir:
        raise ValueError("result_dir must be a non-empty relative artifact description")
    if not isinstance(evaluate_at_end, bool):
        raise TypeError("evaluate_at_end must be boolean")
    if max_updates is not None and segment_updates is not None:
        raise ValueError("max_updates and segment_updates are mutually exclusive")
    if max_updates is not None and (
        isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates <= 0
    ):
        raise ValueError("max_updates must be a positive integer when supplied")
    if segment_updates is not None and (
        isinstance(segment_updates, bool)
        or not isinstance(segment_updates, int)
        or segment_updates <= 0
    ):
        raise ValueError("segment_updates must be positive when supplied")

    resolved = config.to_dict()
    experiment_config = resolved["experiment"]
    environment_config = {"name": task, **config.environment_for_task(task)}
    model_config = resolved["model"]
    ppo_config = resolved["ppo"]
    comparison_config = resolved["comparison"]
    device = resolve_device(experiment_config["device"])
    seed_everything(seed, bool(experiment_config["deterministic"]))
    configure_logging(artifacts.path / "run.log", verbose=verbose)

    rollout_batch_size = int(ppo_config["num_envs"]) * int(ppo_config["rollout_steps"])
    configured_updates = math.ceil(int(ppo_config["total_steps"]) / rollout_batch_size)
    bootstrap_on_truncation = bool(ppo_config["bootstrap_on_truncation"])
    if str(ppo_config.get("algorithm", "ppo")) != "ppo":
        raise NotImplementedError(
            "baseline trainers support only ppo.algorithm=ppo for now"
        )
    value_transform = ValueTransform.from_config(ppo_config)
    checkpoint: dict[str, Any] | None = None
    start_update = 0
    if resume_checkpoint is not None:
        checkpoint = _load_resume_checkpoint(
            resume_checkpoint,
            config=resolved,
            task=task,
            seed=seed,
            method=method,
            rollout_batch_size=rollout_batch_size,
            configured_updates=configured_updates,
            format_version=BASELINE_CHECKPOINT_FORMAT_VERSION,
        )
        start_update = int(checkpoint["update"])
    if segment_updates is not None:
        total_updates = min(configured_updates, start_update + segment_updates)
    elif max_updates is not None:
        total_updates = min(configured_updates, max_updates)
    else:
        total_updates = configured_updates
    if total_updates < start_update or (
        total_updates == start_update and start_update != configured_updates
    ):
        raise ValueError(
            "requested training target must be greater than the resume checkpoint update "
            f"({start_update})"
        )

    artifacts.write_resolved_config(resolved)
    metadata = runtime_metadata(device)
    metadata.update(
        {
            "task": task,
            "seed": seed,
            "method": method,
            "max_updates": max_updates,
            "segment_updates": segment_updates,
            "resume_checkpoint": (
                None if resume_checkpoint is None else str(Path(resume_checkpoint).expanduser())
            ),
        }
    )
    artifacts.write_metadata(metadata)
    LOGGER.info("Starting method=%s task=%s seed=%d device=%s", method, task, seed, device)

    environments = [
        make_env(task, environment_config, seed=seed * 10_000 + index)
        for index in range(int(ppo_config["num_envs"]))
    ]
    batch = SyncEnvironmentBatch(environments)
    first_environment = environments[0]
    model = _make_model(
        first_environment,
        method,
        model_config,
        comparison_config,
        device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(ppo_config["learning_rate"]), eps=1e-5
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Model parameters: %d", parameter_count)

    observation = batch.reset([seed * 10_000 + index for index in range(batch.size)])
    previous_action = torch.zeros(
        batch.size,
        first_environment.action_spec.feature_dim,
        dtype=torch.float32,
        device=device,
    )
    episode_starts = torch.ones(batch.size, dtype=torch.bool, device=device)
    hidden = (
        model.initial_hidden(batch.size, device=device)
        if isinstance(model, RECURRENT_BASELINE_TYPES)
        else None
    )
    recurrent_histories = (
        [RecurrentReplayHistory() for _ in range(batch.size)]
        if isinstance(model, RECURRENT_BASELINE_TYPES)
        else []
    )

    episode_returns = np.zeros(batch.size, dtype=np.float64)
    episode_lengths = np.zeros(batch.size, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    metric_rows: list[dict[str, Any]] = []
    episode_appender = artifacts.csv_appender(EPISODE_FIELDS, name="episodes")
    best_evaluation = -math.inf
    latest_evaluation: dict[str, float] | None = None
    accumulated_wall_time = 0.0
    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
        except (TypeError, ValueError, RuntimeError, KeyError) as exc:
            raise ValueError("resume checkpoint model or optimizer state is incompatible") from exc
        (
            observation,
            previous_action,
            episode_starts,
            recurrent_histories,
            episode_returns,
            episode_lengths,
            completed_returns,
            completed_lengths,
            metric_rows,
            best_evaluation,
            latest_evaluation,
            accumulated_wall_time,
        ) = _restore_baseline_trainer_state(
            checkpoint["trainer_state"],
            environments=environments,
            update=start_update,
            rollout_batch_size=rollout_batch_size,
            recurrent_input_dim=(
                model.observation_dim if isinstance(model, RECURRENT_BASELINE_TYPES) else None
            ),
            device=device,
        )
        if isinstance(model, RECURRENT_BASELINE_TYPES):
            # The committed segment ended with a reconditioned hidden state, so
            # the resumed one replays the retained prefixes through the very same
            # helper instead of restoring a separately stored tensor.
            hidden = _recondition_recurrent_hidden(
                model,
                recurrent_histories,
                input_template=_current_inputs(
                    method,
                    observation,
                    batch,
                    previous_action,
                    device,
                ),
                start_template=episode_starts,
            )
        _restore_global_rng_states(checkpoint, device=device)
        LOGGER.info(
            "Resuming method=%s task=%s seed=%d from update=%d step=%d",
            method,
            task,
            seed,
            start_update,
            start_update * rollout_batch_size,
        )
        _truncate_uncommitted_csv_rows(
            artifacts,
            update=start_update,
            global_step=start_update * rollout_batch_size,
        )
        if not artifacts.metrics_path.exists() or artifacts.metrics_path.stat().st_size == 0:
            artifacts.csv_appender(METRIC_FIELDS).append_many(metric_rows)
    segment_started = time.perf_counter()

    def current_checkpoint_payload(update: int, global_step: int) -> dict[str, Any]:
        wall_time_seconds = accumulated_wall_time + (time.perf_counter() - segment_started)
        trainer_state = _baseline_trainer_state_payload(
            environments=environments,
            observation=observation,
            previous_action=previous_action,
            episode_starts=episode_starts,
            recurrent_histories=recurrent_histories,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            completed_returns=completed_returns,
            completed_lengths=completed_lengths,
            metric_rows=metric_rows,
            best_evaluation=best_evaluation,
            latest_evaluation=latest_evaluation,
            wall_time_seconds=wall_time_seconds,
        )
        return _checkpoint_payload(
            model,  # type: ignore[arg-type]
            optimizer,
            task=task,
            seed=seed,
            update=update,
            global_step=global_step,
            config=resolved,
            trainer_state=trainer_state,
            method=method,
            format_version=BASELINE_CHECKPOINT_FORMAT_VERSION,
        )

    for update in range(start_update + 1, total_updates + 1):
        update_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        input_steps: list[torch.Tensor] = []
        action_steps: list[torch.Tensor] = []
        log_prob_steps: list[torch.Tensor] = []
        value_steps: list[torch.Tensor] = []
        reward_steps: list[torch.Tensor] = []
        done_steps: list[torch.Tensor] = []
        truncation_value_steps: list[torch.Tensor] = []
        truncation_bootstrap_count = 0
        episode_start_steps: list[torch.Tensor] = []
        pre_tanh_steps: list[torch.Tensor] = []
        diffusion_chain_steps: list[torch.Tensor] = []
        update_returns: list[float] = []
        update_lengths: list[int] = []
        action_saturation_fractions: list[float] = []
        episode_rows: list[dict[str, Any]] = []
        model.eval()
        recurrent_prefixes: tuple[RecurrentReplayPrefix, ...] = ()
        if isinstance(model, RECURRENT_BASELINE_TYPES):
            prefix_input_template = _current_inputs(
                method,
                observation,
                batch,
                previous_action,
                device,
            )
            recurrent_prefixes = tuple(
                history.snapshot(
                    input_template=prefix_input_template,
                    start_template=episode_starts,
                )
                for history in recurrent_histories
            )

        for rollout_index in range(int(ppo_config["rollout_steps"])):
            inputs = _current_inputs(method, observation, batch, previous_action, device)
            input_steps.append(inputs.detach())
            episode_start_steps.append(episode_starts.detach())
            with torch.no_grad():
                if isinstance(model, RECURRENT_BASELINE_TYPES):
                    assert hidden is not None
                    sample, old_value, hidden = model.act_step(
                        inputs,
                        hidden,
                        episode_starts,
                        deterministic=False,
                    )
                else:
                    sample, old_value = model.act(inputs, deterministic=False)
            if first_environment.action_spec.is_continuous:
                if sample.diffusion_chain is not None:
                    diffusion_chain_steps.append(sample.diffusion_chain.detach())
                elif sample.pre_tanh_action is not None:
                    pre_tanh_steps.append(sample.pre_tanh_action.detach())
                else:
                    raise RuntimeError(
                        "continuous baseline did not expose its stochastic policy path"
                    )
            environment_action = _environment_actions(first_environment.action_spec, sample.action)
            next_observation, reward, done, step_infos = batch.step(environment_action)
            done_tensor = torch.as_tensor(done, dtype=torch.bool, device=device)
            # The encoded action is needed before it is zeroed for the finished
            # environments: a truncation bootstrap conditions the discarded final
            # observation on the action that produced it.
            action_features = encode_action(
                first_environment.action_spec, sample.action
            ).detach()
            if bootstrap_on_truncation:
                truncation_value_steps.append(
                    _truncation_bootstrap_values(
                        model,
                        method,
                        value_transform=value_transform,
                        infos=step_infos,
                        hidden=hidden,
                        action_features=action_features,
                        device=device,
                    )
                )
                truncation_bootstrap_count += len(truncated_step_indices(step_infos))
            action_steps.append(sample.action.detach())
            log_prob_steps.append(sample.log_prob.detach())
            value_steps.append(value_transform.to_raw(old_value.detach()))
            reward_steps.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
            done_steps.append(done_tensor)

            if isinstance(model, RECURRENT_BASELINE_TYPES):
                for environment_index, history in enumerate(recurrent_histories):
                    history.append(inputs, episode_starts, environment_index)
                    if bool(done[environment_index]):
                        history.clear()

            previous_action = action_features
            if first_environment.action_spec.is_continuous:
                action_saturation_fractions.append(
                    float((previous_action.abs() >= 0.99).float().mean().cpu())
                )
            previous_action = previous_action.clone()
            previous_action[done_tensor] = 0.0
            observation = next_observation
            episode_starts = done_tensor
            episode_returns += reward
            episode_lengths += 1

            for environment_index in np.flatnonzero(done):
                episode_return = float(episode_returns[environment_index])
                episode_length = int(episode_lengths[environment_index])
                update_returns.append(episode_return)
                update_lengths.append(episode_length)
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                episode_rows.append(
                    {
                        "global_step": (
                            (update - 1) * rollout_batch_size + (rollout_index + 1) * batch.size
                        ),
                        "environment": int(environment_index),
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                    }
                )
                episode_returns[environment_index] = 0.0
                episode_lengths[environment_index] = 0

        # One durable CSV append per update, rather than one write per episode.
        episode_appender.append_many(episode_rows)
        if bootstrap_on_truncation:
            # Logged rather than added to metrics.csv: METRIC_FIELDS is a closed
            # CSV schema shared by every run and reader, and making it depend on
            # a config flag would fork the artifact contract for one diagnostic.
            truncation_sum = float(torch.stack(truncation_value_steps).sum().cpu())
            LOGGER.info(
                "truncation_bootstrap update=%d truncations=%d value_mean=%s",
                update,
                truncation_bootstrap_count,
                (
                    f"{truncation_sum / truncation_bootstrap_count:.6f}"
                    if truncation_bootstrap_count
                    else "n/a"
                ),
            )
        with torch.no_grad():
            bootstrap_inputs = _current_inputs(method, observation, batch, previous_action, device)
            if isinstance(model, RECURRENT_BASELINE_TYPES):
                assert hidden is not None
                bootstrap_features, _ = model.step(bootstrap_inputs, hidden, episode_starts)
                last_value = value_transform.to_raw(model.heads.value(bootstrap_features))
            else:
                last_value = value_transform.to_raw(model.value(bootstrap_inputs))

        rewards = torch.stack(reward_steps)
        dones = torch.stack(done_steps)
        old_values = torch.stack(value_steps)
        advantages, returns = generalised_advantage_estimate(
            rewards,
            dones,
            old_values,
            last_value.detach(),
            gamma=float(ppo_config["gamma"]),
            gae_lambda=float(ppo_config["gae_lambda"]),
            truncation_values=(
                torch.stack(truncation_value_steps) if bootstrap_on_truncation else None
            ),
        )
        rollout = BaselineRollout(
            inputs=torch.stack(input_steps),
            actions=torch.stack(action_steps),
            old_log_prob=torch.stack(log_prob_steps),
            old_values=old_values,
            rewards=rewards,
            dones=dones,
            advantages=advantages,
            returns=returns,
            episode_starts=torch.stack(episode_start_steps),
            pre_tanh_actions=(torch.stack(pre_tanh_steps) if pre_tanh_steps else None),
            diffusion_chains=(
                torch.stack(diffusion_chain_steps) if diffusion_chain_steps else None
            ),
            hidden_states=None,
            recurrent_prefixes=recurrent_prefixes,
        )
        model.train()
        if isinstance(model, RECURRENT_BASELINE_TYPES):
            temporal_gradient_mode: TemporalGradientMode = model_config["belief_gradient_mode"]
            ppo_metrics = _update_recurrent(
                model,
                optimizer,
                rollout,
                model_config,
                ppo_config,
                temporal_gradient_mode=temporal_gradient_mode,
            )
            recondition_inputs = _current_inputs(
                method,
                observation,
                batch,
                previous_action,
                device,
            )
            hidden = _recondition_recurrent_hidden(
                model,
                recurrent_histories,
                input_template=recondition_inputs,
                start_template=episode_starts,
            )
        else:
            ppo_metrics = _update_feedforward(model, optimizer, rollout, ppo_config)

        global_step = update * rollout_batch_size
        duration = max(time.perf_counter() - update_started, 1e-9)
        row = {
            "update": update,
            "global_step": global_step,
            "policy_loss": ppo_metrics.policy_loss,
            "value_loss": ppo_metrics.value_loss,
            "entropy": ppo_metrics.entropy,
            "approximate_kl": ppo_metrics.approximate_kl,
            "clip_fraction": ppo_metrics.clip_fraction,
            "gradient_norm": ppo_metrics.gradient_norm,
            "initial_energy_gradient_norm": ppo_metrics.initial_energy_gradient_norm,
            "observation_energy_gradient_norm": ppo_metrics.observation_energy_gradient_norm,
            "transition_energy_gradient_norm": ppo_metrics.transition_energy_gradient_norm,
            "episode_return_mean": (
                float(np.mean(update_returns)) if update_returns else float("nan")
            ),
            "episode_length_mean": (
                float(np.mean(update_lengths)) if update_lengths else float("nan")
            ),
            "episodes_completed": len(update_returns),
            "score_l2_mean": float("nan"),
            "particle_abs_max": float("nan"),
            "particle_clip_fraction": float("nan"),
            "initial_score_l2_mean": float("nan"),
            "recursive_score_l2_mean": float("nan"),
            "initial_particle_l2_mean": float("nan"),
            "recursive_particle_l2_mean": float("nan"),
            "initial_belief_particle_count": float("nan"),
            "recursive_belief_particle_count": float("nan"),
            "score_nonfinite_count": float("nan"),
            "particle_nonfinite_count": float("nan"),
            "action_saturation_fraction": (
                float(np.mean(action_saturation_fractions))
                if action_saturation_fractions
                else float("nan")
            ),
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
            "steps_per_second": rollout_batch_size / duration,
        }
        metric_rows.append(row)
        artifacts.append_metrics(row, METRIC_FIELDS)
        if update % int(experiment_config["log_interval_updates"]) == 0:
            LOGGER.info(
                "method=%s update=%d/%d step=%d policy=%.4f value=%.4f return=%s fps=%.1f",
                method,
                update,
                total_updates,
                global_step,
                ppo_metrics.policy_loss,
                ppo_metrics.value_loss,
                "n/a" if not update_returns else f"{np.mean(update_returns):.3f}",
                row["steps_per_second"],
            )

        should_evaluate = (
            update % int(experiment_config["eval_interval_updates"]) == 0
            or update == configured_updates
            or (
                evaluate_at_end
                and segment_updates is None
                and update == total_updates
            )
        )
        if should_evaluate:
            evaluation_seed = seed + 1_000_000
            deterministic_mode = (
                "greedy"
                if first_environment.action_spec.is_discrete
                else "mean_chain"
                if isinstance(model, RECURRENT_BASELINE_TYPES)
                and model.continuous_policy_kind == "diffusion"
                else "mean_action"
            )
            for mode, deterministic_evaluation in (
                (deterministic_mode, True),
                ("stochastic", False),
            ):
                evaluation, trajectories = evaluate_baseline(
                    model,
                    method=method,
                    task=task,
                    environment_config=environment_config,
                    episodes=int(experiment_config["eval_episodes"]),
                    seed=evaluation_seed,
                    device=device,
                    deterministic=deterministic_evaluation,
                )
                evaluation_row = {
                    "update": update,
                    "global_step": global_step,
                    "mode": mode,
                    "evaluation_seed": evaluation_seed,
                    **evaluation,
                }
                artifacts.csv_appender(EVALUATION_FIELDS, name="evaluation").append(evaluation_row)
                artifacts.save_npz(
                    trajectories,
                    name=f"evaluation_{mode}_trajectories_{update:06d}",
                )
                LOGGER.info(
                    "evaluation method=%s mode=%s update=%d return=%.3f +/- %.3f",
                    method,
                    mode,
                    update,
                    evaluation["mean_return"],
                    evaluation["std_return"],
                )
                if deterministic_evaluation:
                    latest_evaluation = evaluation

            assert latest_evaluation is not None
            if latest_evaluation["mean_return"] > best_evaluation:
                best_evaluation = latest_evaluation["mean_return"]
                save_checkpoint(
                    current_checkpoint_payload(update, global_step),
                    artifacts.checkpoint_path(tag="best"),
                )

        should_save_step_checkpoint = (
            update % int(experiment_config["checkpoint_interval_updates"]) == 0
            or update == total_updates
        )
        if should_save_step_checkpoint or segment_updates is not None:
            payload = current_checkpoint_payload(update, global_step)
            if should_save_step_checkpoint:
                save_checkpoint(payload, artifacts.checkpoint_path(step=global_step))
            save_checkpoint(payload, artifacts.checkpoint_path(tag="latest"))

    numeric_fields = [field for field in METRIC_FIELDS if field not in {"update", "global_step"}]
    arrays: dict[str, np.ndarray] = {
        "update": np.asarray([row["update"] for row in metric_rows], dtype=np.int64),
        "global_step": np.asarray([row["global_step"] for row in metric_rows], dtype=np.int64),
        "episode_returns": np.asarray(completed_returns, dtype=np.float32),
        "episode_lengths": np.asarray(completed_lengths, dtype=np.int32),
    }
    for field in numeric_fields:
        arrays[field] = np.asarray([row[field] for row in metric_rows], dtype=np.float64)
    artifacts.save_npz(arrays)

    # Identical to the score trainer: ``latest_evaluation`` is the deterministic
    # readout, so the legacy key and its explicit alias hold the same number.
    readout_return_mean = None if latest_evaluation is None else latest_evaluation["mean_return"]
    summary = {
        "task": task,
        "seed": seed,
        # Identical to the score trainer's rule: reaching ``configured_updates``
        # is what forces the final evaluation above, so an intermediate segment
        # stays 'partial' and never reaches the completed-only aggregation.
        "status": "completed" if total_updates == configured_updates else "partial",
        "updates": total_updates,
        "global_steps": total_updates * rollout_batch_size,
        "episodes_completed": len(completed_returns),
        "training_return_mean": (float(np.mean(completed_returns)) if completed_returns else None),
        "evaluation_return_mean": readout_return_mean,
        "deterministic_readout_return_mean": readout_return_mean,
        "parameter_count": parameter_count,
        "wall_time_seconds": accumulated_wall_time + (time.perf_counter() - segment_started),
        "result_dir": result_dir,
    }
    LOGGER.info(
        "%s method=%s task=%s seed=%d at update=%d/%d",
        "Completed" if total_updates == configured_updates else "Paused",
        method,
        task,
        seed,
        total_updates,
        configured_updates,
    )
    return summary


__all__ = [
    "BASELINE_CHECKPOINT_FORMAT_VERSION",
    "BASELINE_METHODS",
    "BASELINE_TRAINER_STATE_FORMAT_VERSION",
    "BaselineMethod",
    "BaselineRollout",
    "RecurrentReplayHistory",
    "RecurrentReplayPrefix",
    "evaluate_baseline",
    "replay_recurrent_sequence",
    "train_baseline_single",
]
