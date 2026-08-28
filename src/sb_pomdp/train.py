"""Single-task training loop for the score-belief PPO experiment."""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import platform
import random
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional

from .artifacts import RunArtifacts, SeedArtifacts
from .buffer import BeliefReplayPrefix, RolloutBuilder
from .config import ExperimentConfig, load_config, normalize_legacy_resolved_config
from .envs import ActionSpec, make_env
from .policies import ScoreBeliefActorCritic
from .ppo import update_ppo
from .value_transform import ValueTransform

LOGGER = logging.getLogger("sb_pomdp")

CHECKPOINT_FORMAT_VERSION = 4
TRAINER_STATE_FORMAT_VERSION = 1
# Keep a rejected-resume diagnostic readable when a long run kept many steps.
_ROLLBACK_CANDIDATE_LIMIT = 5

METRIC_FIELDS = (
    "update",
    "global_step",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "clip_fraction",
    "gradient_norm",
    "initial_energy_gradient_norm",
    "observation_energy_gradient_norm",
    "transition_energy_gradient_norm",
    "episode_return_mean",
    "episode_length_mean",
    "episodes_completed",
    "score_l2_mean",
    "initial_score_l2_mean",
    "recursive_score_l2_mean",
    "initial_particle_l2_mean",
    "recursive_particle_l2_mean",
    "initial_belief_particle_count",
    "recursive_belief_particle_count",
    "score_nonfinite_count",
    "particle_nonfinite_count",
    "particle_abs_max",
    "particle_clip_fraction",
    "action_saturation_fraction",
    "gpu_peak_memory_mb",
    "steps_per_second",
)

EPISODE_FIELDS = ("global_step", "environment", "episode_return", "episode_length")
EVALUATION_FIELDS = (
    "update",
    "global_step",
    "mode",
    "evaluation_seed",
    "mean_return",
    "std_return",
    "mean_length",
    "action_saturation_fraction",
)
SUMMARY_FIELDS = (
    "task",
    "seed",
    "status",
    "updates",
    "global_steps",
    "episodes_completed",
    "training_return_mean",
    # ``evaluation_return_mean`` has always held the *deterministic readout*
    # return of the final evaluation, never the stochastic policy return that
    # the README designates as the primary metric.  The name is kept for
    # compatibility with every artifact the running campaign produced;
    # ``deterministic_readout_return_mean`` is an explicit alias carrying the
    # identical value so a reader never has to guess.  The primary metric lives
    # in the ``stochastic_*`` columns of ``comparison.csv``.
    "evaluation_return_mean",
    "deterministic_readout_return_mean",
    "parameter_count",
    "wall_time_seconds",
    "result_dir",
    "error",
)
# Summary keys introduced after the production campaign was submitted, mapped to
# the pre-existing key that already carried the same value.  Extend this table -
# and nothing else - when another alias has to stay comparable with artifacts
# the campaign wrote; :func:`normalize_legacy_summary` is the single place that
# applies it.
_LEGACY_SUMMARY_ALIASES: dict[str, str] = {
    "deterministic_readout_return_mean": "evaluation_return_mean",
}


def normalize_legacy_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *summary* with the post-campaign summary aliases filled.

    A ``summary.json`` or ``manifest.json`` written before an alias in
    ``_LEGACY_SUMMARY_ALIASES`` existed lacks the key but already carries its
    value under the original name, so filling it in here makes such a row *equal*
    to one the current trainer would write.  Rows that already define the alias,
    and rows that lack the original key as well, are returned unchanged, which
    keeps a genuinely malformed row visible to the caller's validation.
    """

    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping")
    result = dict(summary)
    for alias, original in _LEGACY_SUMMARY_ALIASES.items():
        if alias not in result and original in result:
            result[alias] = result[original]
    return result


def configure_logging(log_path: Path, *, verbose: bool = True) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    if verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def _start_timed_phase(
    phase: str,
    *,
    update: int,
    total_updates: int,
    device: torch.device,
) -> float:
    """Log a phase boundary after draining earlier CUDA work."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    LOGGER.info(
        "phase=%s status=start update=%d/%d",
        phase,
        update,
        total_updates,
    )
    return time.perf_counter()


def _finish_timed_phase(
    phase: str,
    started: float,
    *,
    update: int,
    total_updates: int,
    device: torch.device,
) -> float:
    """Synchronize phase work, log its elapsed time, and return the duration."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    LOGGER.info(
        "phase=%s status=end update=%d/%d seconds=%.3f",
        phase,
        update,
        total_updates,
        duration,
    )
    return duration


def source_sha256() -> str:
    """Return the code identity used to guard an exact segmented resume."""

    package_root = Path(__file__).resolve().parent
    source_digest = hashlib.sha256()
    for source_path in sorted(package_root.glob("*.py"), key=lambda path: path.name):
        source_digest.update(source_path.name.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(source_path.read_bytes())
        source_digest.update(b"\0")
    return source_digest.hexdigest()


def runtime_metadata(
    device: torch.device,
    *,
    source_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Describe this process, reusing a separately committed source identity."""

    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device": str(device),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "source_sha256": source_fingerprint or source_sha256(),
    }
    if device.type == "cuda":
        metadata["cuda"] = torch.version.cuda
        metadata["gpu"] = torch.cuda.get_device_name(device)
    for name in ("PJM_JOBID", "PJM_JOBNAME", "PJM_RSCGRP", "PJM_STEPNUM"):
        if value := os.environ.get(name):
            metadata[name.lower()] = value
    return metadata


class SyncEnvironmentBatch:
    """Minimal auto-reset vector wrapper for the self-contained environments."""

    def __init__(self, environments: list[Any]) -> None:
        if not environments:
            raise ValueError("at least one environment is required")
        self.environments = environments
        self.action_spec: ActionSpec = environments[0].action_spec

    @property
    def size(self) -> int:
        return len(self.environments)

    def reset(self, seeds: list[int]) -> np.ndarray:
        if len(seeds) != self.size:
            raise ValueError("one reset seed is required per environment")
        observations = [env.reset(seed=seed)[0] for env, seed in zip(self.environments, seeds)]
        return np.stack(observations).astype(np.float32, copy=False)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step every environment, auto-resetting the ones that finished.

        The returned observation array is the auto-reset one that the next
        policy input needs.  Because that array discards the genuine post-step
        observation, each finished environment's info additionally carries
        ``final_observation`` and ``final_oracle_features`` from *before* the
        reset, alongside the separate ``terminated``/``truncated`` flags.  A
        time-limit truncation is the only case where the discarded state is
        still worth a value bootstrap (``ppo.bootstrap_on_truncation``); the
        combined ``done`` array keeps meaning "the episode ended here".
        """

        observations: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[bool] = []
        infos: list[dict[str, Any]] = []
        for index, environment in enumerate(self.environments):
            observation, reward, terminated, truncated, info = environment.step(actions[index])
            done = bool(terminated or truncated)
            terminal_info = dict(info)
            terminal_info["terminated"] = bool(terminated)
            terminal_info["truncated"] = bool(truncated)
            if done:
                terminal_info["final_observation"] = np.asarray(observation, dtype=np.float32)
                terminal_info["final_oracle_features"] = np.asarray(
                    environment.oracle_features(), dtype=np.float32
                )
                observation, reset_info = environment.reset()
                terminal_info["reset_state"] = reset_info["state"]
            observations.append(observation)
            rewards.append(float(reward))
            dones.append(done)
            infos.append(terminal_info)
        return (
            np.stack(observations).astype(np.float32, copy=False),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            infos,
        )


@dataclass(slots=True)
class BeliefContext:
    observation: torch.Tensor
    previous_particles: torch.Tensor
    previous_action: torch.Tensor
    initial: torch.Tensor

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "observation": self.observation.detach().cpu().clone(),
            "previous_particles": self.previous_particles.detach().cpu().clone(),
            "previous_action": self.previous_action.detach().cpu().clone(),
            "initial": self.initial.detach().cpu().clone(),
        }


@dataclass(slots=True)
class BeliefReplayHistory:
    """Reparameterization inputs for one currently active episode."""

    observations: list[torch.Tensor] = dataclass_field(default_factory=list)
    previous_actions: list[torch.Tensor] = dataclass_field(default_factory=list)
    initial: list[torch.Tensor] = dataclass_field(default_factory=list)
    ula_initial_noise: list[torch.Tensor] = dataclass_field(default_factory=list)
    ula_step_noises: list[torch.Tensor] = dataclass_field(default_factory=list)

    def state_dict(self) -> dict[str, list[torch.Tensor]]:
        return {
            "observations": [value.detach().cpu().clone() for value in self.observations],
            "previous_actions": [
                value.detach().cpu().clone() for value in self.previous_actions
            ],
            "initial": [value.detach().cpu().clone() for value in self.initial],
            "ula_initial_noise": [
                value.detach().cpu().clone() for value in self.ula_initial_noise
            ],
            "ula_step_noises": [
                value.detach().cpu().clone() for value in self.ula_step_noises
            ],
        }

    def append(
        self,
        context: BeliefContext,
        environment_index: int,
        ula_initial_noise: torch.Tensor,
        ula_step_noises: torch.Tensor,
    ) -> None:
        self.observations.append(context.observation[environment_index].detach().clone())
        self.previous_actions.append(
            context.previous_action[environment_index].detach().clone()
        )
        self.initial.append(context.initial[environment_index].detach().clone())
        self.ula_initial_noise.append(
            ula_initial_noise[environment_index].detach().clone()
        )
        self.ula_step_noises.append(
            ula_step_noises[environment_index].detach().clone()
        )

    def clear(self) -> None:
        self.observations.clear()
        self.previous_actions.clear()
        self.initial.clear()
        self.ula_initial_noise.clear()
        self.ula_step_noises.clear()

    def snapshot(
        self,
        context: BeliefContext,
        environment_index: int,
        *,
        langevin_steps: int,
    ) -> BeliefReplayPrefix:
        if self.observations:
            return BeliefReplayPrefix(
                observations=torch.stack(self.observations),
                previous_actions=torch.stack(self.previous_actions),
                initial=torch.stack(self.initial),
                ula_initial_noise=torch.stack(self.ula_initial_noise),
                ula_step_noises=torch.stack(self.ula_step_noises),
            )
        particle_shape = tuple(context.previous_particles.shape[1:])
        return BeliefReplayPrefix(
            observations=context.observation.new_empty(
                (0, context.observation.shape[-1])
            ),
            previous_actions=context.previous_action.new_empty(
                (0, context.previous_action.shape[-1])
            ),
            initial=context.initial.new_empty((0,)),
            ula_initial_noise=context.previous_particles.new_empty((0, *particle_shape)),
            ula_step_noises=context.previous_particles.new_empty(
                (0, langevin_steps, *particle_shape)
            ),
        )


def _recondition_belief_context(
    model: ScoreBeliefActorCritic,
    context: BeliefContext,
    histories: list[BeliefReplayHistory],
    model_config: dict[str, Any],
) -> BeliefContext:
    """Rebuild ongoing episode states after PPO changed the belief parameters."""

    rebuilt = context.previous_particles.detach().clone()
    for environment_index, history in enumerate(histories):
        if not history.observations:
            if not bool(context.initial[environment_index]):
                raise RuntimeError("a non-initial belief context is missing its episode history")
            rebuilt[environment_index].zero_()
            continue
        previous_particles = torch.zeros_like(rebuilt[environment_index : environment_index + 1])
        for time_index in range(len(history.observations)):
            previous_particles, _ = model.belief.particles_from_noise(
                history.observations[time_index][None],
                previous_particles,
                history.previous_actions[time_index][None],
                history.initial[time_index][None],
                initial_noise=history.ula_initial_noise[time_index][None],
                langevin_noise=history.ula_step_noises[time_index][None],
                step_size=float(model_config["langevin_step_size"]),
                track_grad=False,
            )
        rebuilt[environment_index] = previous_particles[0]
    return BeliefContext(
        observation=context.observation,
        previous_particles=rebuilt,
        previous_action=context.previous_action,
        initial=context.initial,
    )


def encode_action(action_spec: ActionSpec, action: torch.Tensor) -> torch.Tensor:
    if action_spec.is_discrete:
        assert action_spec.n is not None
        return functional.one_hot(action.long(), num_classes=action_spec.n).float()
    assert action_spec.low is not None and action_spec.high is not None
    # ActionSpec bounds are intentionally read-only; construct owned tensors to
    # avoid PyTorch's non-writable NumPy view warning.
    low = torch.tensor(action_spec.low.tolist(), dtype=action.dtype, device=action.device)
    high = torch.tensor(action_spec.high.tolist(), dtype=action.dtype, device=action.device)
    return (2.0 * (action - low) / (high - low) - 1.0).clamp(-1.0, 1.0)


def model_for_environment(
    environment: Any,
    model_config: dict[str, Any],
    device: torch.device,
) -> ScoreBeliefActorCritic:
    spec = environment.action_spec
    model = ScoreBeliefActorCritic(
        observation_dim=environment.observation_dim,
        state_dim=environment.state_dim,
        action_kind=spec.kind,
        action_feature_dim=spec.feature_dim,
        discrete_actions=spec.n,
        action_low=None if spec.low is None else spec.low.tolist(),
        action_high=None if spec.high is None else spec.high.tolist(),
        model_config=model_config,
    )
    return model.to(device)


def _sample_belief(
    model: ScoreBeliefActorCritic,
    context: BeliefContext,
    model_config: dict[str, Any],
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_noise, ula_step_noises = model.belief.draw_particle_noise(
        context.observation,
        num_particles=int(model_config["num_particles"]),
        num_steps=int(model_config["langevin_steps"]),
        generator=generator,
    )
    return model.belief.particles_from_noise(
        context.observation,
        context.previous_particles,
        context.previous_action,
        context.initial,
        initial_noise=initial_noise,
        langevin_noise=ula_step_noises,
        step_size=float(model_config["langevin_step_size"]),
        track_grad=False,
    )


def _sample_belief_for_rollout(
    model: ScoreBeliefActorCritic,
    context: BeliefContext,
    model_config: dict[str, Any],
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a belief and return the exogenous ULA noise needed for PPO replay."""

    initial_noise, ula_step_noises = model.belief.draw_particle_noise(
        context.observation,
        num_particles=int(model_config["num_particles"]),
        num_steps=int(model_config["langevin_steps"]),
        generator=generator,
    )
    particles, scores = model.belief.particles_from_noise(
        context.observation,
        context.previous_particles,
        context.previous_action,
        context.initial,
        initial_noise=initial_noise,
        langevin_noise=ula_step_noises,
        step_size=float(model_config["langevin_step_size"]),
        track_grad=False,
    )
    return particles, scores, initial_noise, ula_step_noises


def _environment_actions(action_spec: ActionSpec, action: torch.Tensor) -> np.ndarray:
    array = action.detach().cpu().numpy()
    if action_spec.is_discrete:
        return array.astype(np.int64, copy=False)
    return array.astype(np.float32, copy=False)


@torch.no_grad()
def _policy_from_particles(
    model: ScoreBeliefActorCritic,
    particles: torch.Tensor,
    scores: torch.Tensor,
    *,
    deterministic: bool,
    generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    condition = model.encode(particles, scores)
    sample = model.sample_policy(
        condition,
        deterministic=deterministic,
        generator=generator,
    )
    value = model.value(condition)
    return (
        sample.action,
        sample.log_prob,
        sample.entropy,
        sample.diffusion_chain,
        sample.pre_tanh_action,
        value,
    )


def _next_context(
    *,
    observation: np.ndarray,
    particles: torch.Tensor,
    action_features: torch.Tensor,
    done: np.ndarray,
    device: torch.device,
) -> BeliefContext:
    done_tensor = torch.as_tensor(done, dtype=torch.bool, device=device)
    next_particles = particles.detach().clone()
    next_action = action_features.detach().clone()
    next_particles[done_tensor] = 0.0
    next_action[done_tensor] = 0.0
    return BeliefContext(
        observation=torch.as_tensor(observation, dtype=torch.float32, device=device),
        previous_particles=next_particles,
        previous_action=next_action,
        initial=done_tensor,
    )


def truncated_step_indices(infos: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return the environments whose episode ended on the horizon, not a terminal state."""

    return [index for index, info in enumerate(infos) if bool(info.get("truncated", False))]


def final_step_features(
    infos: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    """Stack one pre-reset array of the finished environments onto *device*."""

    missing = [index for index in indices if name not in infos[index]]
    if missing:
        raise RuntimeError(
            f"environments {missing} finished without recording {name!r}; "
            "truncation bootstrapping needs the pre-reset step of SyncEnvironmentBatch"
        )
    stacked = np.stack([np.asarray(infos[index][name], dtype=np.float32) for index in indices])
    return torch.as_tensor(stacked, dtype=torch.float32, device=device)


def truncation_bootstrap_values(
    model: ScoreBeliefActorCritic,
    model_config: dict[str, Any],
    *,
    infos: Sequence[Mapping[str, Any]],
    particles: torch.Tensor,
    action_features: torch.Tensor,
    device: torch.device,
    value_transform: ValueTransform | None = None,
) -> torch.Tensor:
    """Value the post-step state that a time-limit truncation discarded.

    ``V`` is a function of the belief condition, so the belief is advanced one
    step with the executed action and the genuine final observation on a *clone*
    of the per-environment context.  Neither the online context nor the replay
    histories are touched: the resulting belief belongs to no episode and is
    never trained on.  The returned ``[batch]`` tensor is zero wherever the step
    did not truncate, including a true terminal state, which must not bootstrap.

    The ULA noise is drawn from the global stream, exactly like the rollout's
    own belief samples.  That is intentional and only ever happens when
    ``ppo.bootstrap_on_truncation`` is enabled: the stream is part of the
    checkpoint, so a resumed segment stays exact, and nothing is consumed on the
    default path because this function is not called there.
    """

    values = torch.zeros(particles.shape[0], dtype=torch.float32, device=device)
    indices = truncated_step_indices(infos)
    if not indices:
        return values
    selector = torch.as_tensor(indices, dtype=torch.long, device=device)
    final_context = BeliefContext(
        observation=final_step_features(
            infos, indices, name="final_observation", device=device
        ),
        previous_particles=particles.detach().index_select(0, selector).clone(),
        previous_action=action_features.detach().index_select(0, selector).clone(),
        initial=torch.zeros(len(indices), dtype=torch.bool, device=device),
    )
    with torch.no_grad():
        final_particles, final_scores = _sample_belief(model, final_context, model_config)
        bootstrap = (value_transform or ValueTransform()).to_raw(
            model.value(model.encode(final_particles, final_scores))
        )
    values.index_copy_(0, selector, bootstrap.detach().to(dtype=values.dtype))
    return values


def evaluate(
    model: ScoreBeliefActorCritic,
    *,
    task: str,
    environment_config: dict[str, Any],
    model_config: dict[str, Any],
    episodes: int,
    seed: int,
    device: torch.device,
    deterministic: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
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
            context = BeliefContext(
                observation=torch.as_tensor(observation[None], dtype=torch.float32, device=device),
                previous_particles=torch.zeros(
                    1,
                    int(model_config["num_particles"]),
                    environment.state_dim,
                    dtype=torch.float32,
                    device=device,
                ),
                previous_action=torch.zeros(
                    1,
                    environment.action_spec.feature_dim,
                    dtype=torch.float32,
                    device=device,
                ),
                initial=torch.ones(1, dtype=torch.bool, device=device),
            )
            episode_observations: list[np.ndarray] = []
            episode_actions: list[np.ndarray] = []
            episode_rewards: list[float] = []
            episode_states: list[np.ndarray] = []
            episode_return = 0.0

            while True:
                particles, scores = _sample_belief(
                    model, context, model_config, generator=generator
                )
                action, _, _, _, _, _ = _policy_from_particles(
                    model,
                    particles,
                    scores,
                    deterministic=deterministic,
                    generator=generator,
                )
                environment_action = _environment_actions(environment.action_spec, action)[0]
                episode_observations.append(context.observation[0].detach().cpu().numpy())
                episode_actions.append(np.asarray(environment_action).reshape(-1))
                episode_states.append(np.asarray(info["state"], dtype=np.float32))
                next_observation, reward, terminated, truncated, info = environment.step(
                    environment_action
                )
                episode_rewards.append(float(reward))
                episode_return += float(reward)
                action_features = encode_action(environment.action_spec, action)
                if environment.action_spec.is_continuous:
                    saturation_fractions.append(
                        float((action_features.abs() >= 0.99).float().mean().cpu())
                    )
                context = BeliefContext(
                    observation=torch.as_tensor(
                        next_observation[None], dtype=torch.float32, device=device
                    ),
                    previous_particles=particles,
                    previous_action=action_features,
                    initial=torch.zeros(1, dtype=torch.bool, device=device),
                )
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


def _trainer_state_payload(
    *,
    environments: list[Any],
    context: BeliefContext,
    belief_histories: list[BeliefReplayHistory],
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
    completed_returns: list[float],
    completed_lengths: list[int],
    metric_rows: list[dict[str, Any]],
    best_evaluation: float,
    latest_evaluation: dict[str, float] | None,
    wall_time_seconds: float,
) -> dict[str, Any]:
    return {
        "format_version": TRAINER_STATE_FORMAT_VERSION,
        "environments": [environment.state_dict() for environment in environments],
        "belief_context": context.state_dict(),
        "belief_histories": [history.state_dict() for history in belief_histories],
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


def _checkpoint_payload(
    model: ScoreBeliefActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    task: str,
    seed: int,
    update: int,
    global_step: int,
    config: dict[str, Any],
    trainer_state: dict[str, Any] | None = None,
    method: str | None = None,
    format_version: Any = CHECKPOINT_FORMAT_VERSION,
) -> dict[str, Any]:
    """Return one checkpoint payload for the trainer identified by ``format_version``.

    ``format_version`` names the writing trainer's checkpoint dialect.  The score
    trainer keeps :data:`CHECKPOINT_FORMAT_VERSION`; the baseline trainer passes
    its own identifier so neither trainer can silently resume the other's state.
    """

    payload = {
        "format_version": format_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "task": task,
        "seed": seed,
        "update": update,
        "global_step": global_step,
        "config": config,
        "method": method,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if trainer_state is not None:
        payload["trainer_state"] = trainer_state
    return payload


def _checkpoint_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"checkpoint {name} must be a tensor")
    if tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError(
            f"checkpoint {name} must have shape={shape} and dtype={dtype}, "
            f"got shape={tuple(value.shape)} and dtype={value.dtype}"
        )
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"checkpoint {name} contains non-finite values")
    return value.detach().to(device=device).clone()


def _restore_belief_context(
    state_dict: Any,
    *,
    batch_size: int,
    observation_dim: int,
    num_particles: int,
    state_dim: int,
    action_feature_dim: int,
    device: torch.device,
) -> BeliefContext:
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint trainer_state.belief_context must be a mapping")
    expected = {"observation", "previous_particles", "previous_action", "initial"}
    if set(state_dict) != expected:
        raise ValueError("checkpoint belief_context has missing or unexpected fields")
    return BeliefContext(
        observation=_checkpoint_tensor(
            state_dict["observation"],
            name="belief_context.observation",
            shape=(batch_size, observation_dim),
            dtype=torch.float32,
            device=device,
        ),
        previous_particles=_checkpoint_tensor(
            state_dict["previous_particles"],
            name="belief_context.previous_particles",
            shape=(batch_size, num_particles, state_dim),
            dtype=torch.float32,
            device=device,
        ),
        previous_action=_checkpoint_tensor(
            state_dict["previous_action"],
            name="belief_context.previous_action",
            shape=(batch_size, action_feature_dim),
            dtype=torch.float32,
            device=device,
        ),
        initial=_checkpoint_tensor(
            state_dict["initial"],
            name="belief_context.initial",
            shape=(batch_size,),
            dtype=torch.bool,
            device=device,
        ),
    )


def _restore_belief_history(
    state_dict: Any,
    *,
    history_index: int,
    observation_dim: int,
    num_particles: int,
    state_dim: int,
    action_feature_dim: int,
    langevin_steps: int,
    device: torch.device,
) -> BeliefReplayHistory:
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"checkpoint belief history {history_index} must be a mapping")
    fields_and_shapes = {
        "observations": ((observation_dim,), torch.float32),
        "previous_actions": ((action_feature_dim,), torch.float32),
        "initial": ((), torch.bool),
        "ula_initial_noise": ((num_particles, state_dim), torch.float32),
        "ula_step_noises": (
            (langevin_steps, num_particles, state_dim),
            torch.float32,
        ),
    }
    if set(state_dict) != set(fields_and_shapes):
        raise ValueError(f"checkpoint belief history {history_index} has invalid fields")
    lengths: set[int] = set()
    restored: dict[str, list[torch.Tensor]] = {}
    for field_name, (shape, dtype) in fields_and_shapes.items():
        values = state_dict[field_name]
        if not isinstance(values, list):
            raise TypeError(
                f"checkpoint belief history {history_index}.{field_name} must be a list"
            )
        lengths.add(len(values))
        restored[field_name] = [
            _checkpoint_tensor(
                value,
                name=f"belief_histories[{history_index}].{field_name}[{item_index}]",
                shape=shape,
                dtype=dtype,
                device=device,
            )
            for item_index, value in enumerate(values)
        ]
    if len(lengths) != 1:
        raise ValueError(f"checkpoint belief history {history_index} fields have unequal lengths")
    return BeliefReplayHistory(**restored)


def _finite_numeric_list(value: Any, *, name: str, integer: bool) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"checkpoint {name} must be a list")
    restored: list[Any] = []
    for index, item in enumerate(value):
        if isinstance(item, (bool, np.bool_)):
            raise TypeError(f"checkpoint {name}[{index}] has an invalid value")
        if integer:
            if not isinstance(item, (int, np.integer)) or int(item) <= 0:
                raise ValueError(f"checkpoint {name}[{index}] must be a positive integer")
            restored.append(int(item))
        else:
            if not isinstance(item, (int, float, np.integer, np.floating)) or not np.isfinite(
                item
            ):
                raise ValueError(f"checkpoint {name}[{index}] must be finite")
            restored.append(float(item))
    return restored


def _restore_trainer_state(
    state_dict: Any,
    *,
    environments: list[Any],
    update: int,
    rollout_batch_size: int,
    model_config: dict[str, Any],
    device: torch.device,
) -> tuple[
    BeliefContext,
    list[BeliefReplayHistory],
    np.ndarray,
    np.ndarray,
    list[float],
    list[int],
    list[dict[str, Any]],
    float,
    dict[str, float] | None,
    float,
]:
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "checkpoint lacks a complete trainer_state; legacy checkpoints cannot be resumed"
        )
    expected_fields = {
        "format_version",
        "environments",
        "belief_context",
        "belief_histories",
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
    if state_dict["format_version"] != TRAINER_STATE_FORMAT_VERSION:
        raise ValueError(
            "unsupported trainer_state format_version: "
            f"{state_dict['format_version']!r}"
        )

    environment_states = state_dict["environments"]
    if not isinstance(environment_states, list) or len(environment_states) != len(environments):
        raise ValueError("checkpoint has the wrong number of environment states")
    for environment, environment_state in zip(environments, environment_states):
        environment.load_state_dict(environment_state)

    first_environment = environments[0]
    batch_size = len(environments)
    num_particles = int(model_config["num_particles"])
    langevin_steps = int(model_config["langevin_steps"])
    context = _restore_belief_context(
        state_dict["belief_context"],
        batch_size=batch_size,
        observation_dim=first_environment.observation_dim,
        num_particles=num_particles,
        state_dim=first_environment.state_dim,
        action_feature_dim=first_environment.action_spec.feature_dim,
        device=device,
    )

    history_states = state_dict["belief_histories"]
    if not isinstance(history_states, list) or len(history_states) != batch_size:
        raise ValueError("checkpoint has the wrong number of belief histories")
    belief_histories = [
        _restore_belief_history(
            history_state,
            history_index=index,
            observation_dim=first_environment.observation_dim,
            num_particles=num_particles,
            state_dim=first_environment.state_dim,
            action_feature_dim=first_environment.action_spec.feature_dim,
            langevin_steps=langevin_steps,
            device=device,
        )
        for index, history_state in enumerate(history_states)
    ]
    for index, history in enumerate(belief_histories):
        if bool(context.initial[index]) != (not history.observations):
            raise ValueError(
                f"checkpoint belief context/history mismatch for environment {index}"
            )

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
    for index, (environment_state, history) in enumerate(
        zip(environment_states, belief_histories)
    ):
        expected_length = int(episode_lengths[index])
        if len(history.observations) != expected_length:
            raise ValueError(
                f"checkpoint episode length/belief history mismatch for environment {index}"
            )
        if int(environment_state["elapsed_steps"]) != expected_length:
            raise ValueError(
                f"checkpoint episode length/environment state mismatch for environment {index}"
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
        if not isinstance(latest_evaluation, Mapping) or set(latest_evaluation) != evaluation_fields:
            raise ValueError("checkpoint latest_evaluation has invalid fields")
        latest_evaluation = {
            key: float(latest_evaluation[key]) for key in evaluation_fields
        }
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
        context,
        belief_histories,
        episode_returns.copy(),
        episode_lengths.copy(),
        completed_returns,
        completed_lengths,
        metric_rows,
        best_evaluation,
        latest_evaluation,
        float(wall_time_seconds),
    )


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Atomically replace a checkpoint after its bytes reach stable storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"._{uuid4().hex[:12]}.pt.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_finite_checkpoint_tree(
    value: Any,
    *,
    name: str,
    seen: set[int] | None = None,
) -> None:
    """Reject non-finite model/optimizer values before they enter a trainer."""

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"resume checkpoint {name} contains non-finite values")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not bool(np.isfinite(value).all()):
            raise ValueError(f"resume checkpoint {name} contains non-finite values")
        return
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        if not bool(np.isfinite(value)):
            raise ValueError(f"resume checkpoint {name} contains a non-finite value")
        return

    if not isinstance(value, (Mapping, list, tuple)):
        return
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_checkpoint_tree(
                item,
                name=f"{name}.{key}",
                seen=seen,
            )
    else:
        for index, item in enumerate(value):
            _require_finite_checkpoint_tree(
                item,
                name=f"{name}[{index}]",
                seen=seen,
            )


def _load_resume_checkpoint(
    path: str | Path,
    *,
    config: dict[str, Any],
    task: str,
    seed: int,
    method: str | None,
    rollout_batch_size: int,
    configured_updates: int,
    format_version: Any = CHECKPOINT_FORMAT_VERSION,
) -> dict[str, Any]:
    """Load and fully validate one resumable checkpoint of ``format_version``.

    ``format_version`` is the writing trainer's checkpoint dialect, so a payload
    produced by another trainer is rejected before any of its state is trusted.
    """

    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"could not load resume checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, Mapping):
        raise TypeError("resume checkpoint must contain a mapping")
    if "trainer_state" not in checkpoint:
        raise ValueError(
            "resume checkpoint lacks trainer_state and cannot continue exactly; "
            "start from a checkpoint written by the resumable trainer"
        )
    required = {
        "format_version",
        "model",
        "optimizer",
        "task",
        "seed",
        "update",
        "global_step",
        "config",
        "method",
        "trainer_state",
        "torch_rng_state",
        "cuda_rng_state",
        "numpy_rng_state",
        "python_rng_state",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"resume checkpoint is incomplete; missing={sorted(missing)}")
    if checkpoint["format_version"] != format_version:
        raise ValueError(
            "resume checkpoint was written in a different checkpoint format; "
            f"expected format_version={format_version!r}, "
            f"got {checkpoint['format_version']!r}"
        )
    checkpoint_config = checkpoint["config"]
    if isinstance(checkpoint_config, Mapping):
        # A checkpoint written before a config key existed must stay resumable
        # at that key's legacy default.
        checkpoint_config = normalize_legacy_resolved_config(checkpoint_config)
    if checkpoint_config != config:
        raise ValueError("resume checkpoint config does not exactly match the requested config")
    if checkpoint["task"] != task:
        raise ValueError(
            f"resume checkpoint task mismatch: expected {task!r}, got {checkpoint['task']!r}"
        )
    if checkpoint["seed"] != seed:
        raise ValueError(
            f"resume checkpoint seed mismatch: expected {seed!r}, got {checkpoint['seed']!r}"
        )
    if checkpoint["method"] != method:
        raise ValueError(
            "resume checkpoint method mismatch: "
            f"expected {method!r}, got {checkpoint['method']!r}"
        )
    update = checkpoint["update"]
    if (
        isinstance(update, (bool, np.bool_))
        or not isinstance(update, (int, np.integer))
        or not 1 <= int(update) <= configured_updates
    ):
        raise ValueError("resume checkpoint update is outside the configured training range")
    update = int(update)
    global_step = checkpoint["global_step"]
    expected_global_step = update * rollout_batch_size
    if (
        isinstance(global_step, (bool, np.bool_))
        or not isinstance(global_step, (int, np.integer))
        or int(global_step) != expected_global_step
    ):
        raise ValueError(
            "resume checkpoint global_step is inconsistent with update and rollout batch size: "
            f"expected {expected_global_step}, got {global_step!r}"
        )
    try:
        _require_finite_checkpoint_tree(checkpoint["model"], name="model")
        _require_finite_checkpoint_tree(checkpoint["optimizer"], name="optimizer")
    except ValueError as exc:
        raise ValueError(
            f"{exc}{_checkpoint_rollback_guidance(checkpoint_path)}"
        ) from exc
    return dict(checkpoint)


def _restore_global_rng_states(checkpoint: Mapping[str, Any], *, device: torch.device) -> None:
    torch_rng_state = checkpoint["torch_rng_state"]
    if (
        not isinstance(torch_rng_state, torch.Tensor)
        or torch_rng_state.device.type != "cpu"
        or torch_rng_state.dtype != torch.uint8
        or torch_rng_state.ndim != 1
    ):
        raise ValueError("resume checkpoint torch_rng_state is invalid")
    cuda_rng_state = checkpoint["cuda_rng_state"]
    if cuda_rng_state is not None:
        if not torch.cuda.is_available():
            raise ValueError("resume checkpoint contains CUDA RNG state but CUDA is unavailable")
        if (
            not isinstance(cuda_rng_state, list)
            or len(cuda_rng_state) != torch.cuda.device_count()
            or any(
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.uint8
                or value.ndim != 1
                for value in cuda_rng_state
            )
        ):
            raise ValueError("resume checkpoint cuda_rng_state is invalid for this host")
    elif device.type == "cuda":
        raise ValueError("resume checkpoint is missing CUDA RNG state for CUDA training")

    try:
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("resume checkpoint contains invalid global RNG state") from exc


def _checkpoint_rollback_guidance(checkpoint_path: str | Path) -> str:
    """Name the checkpoints a rejected resume can be restarted from.

    A rejected checkpoint is never repaired in place, so the only recovery is to
    resume from an earlier committed checkpoint of the same run.  The candidates
    are the siblings of ``checkpoint_path`` inside the run's ``checkpoints/``
    directory: ``step_*.pt`` newest first, then the tagged checkpoints.  The
    returned text is appended to a validation failure, so it starts with a
    separator and is empty for an unreadable directory.
    """

    rejected = Path(checkpoint_path).expanduser()
    directory = rejected.parent
    try:
        candidates = [path for path in directory.glob("*.pt") if path.name != rejected.name]
    except OSError:
        return ""

    def ordering(path: Path) -> tuple[int, int, str]:
        stem = path.stem
        if stem.startswith("step_") and stem[len("step_") :].isdigit():
            return (0, -int(stem[len("step_") :]), stem)
        return (1, 0, stem)

    names = [path.name for path in sorted(candidates, key=ordering)]
    if not names:
        return (
            f"; {directory} holds no other checkpoint, so this run has to restart "
            "from update 0"
        )
    shown = names[:_ROLLBACK_CANDIDATE_LIMIT]
    listed = ", ".join(shown)
    remaining = "" if len(shown) == len(names) else f" (newest {len(shown)} of {len(names)})"
    return f"; resume instead from an older checkpoint in {directory}: {listed}{remaining}"


def _csv_progress_value(row: Mapping[str, str], field: str, *, path: Path) -> int:
    """Read one integer progress column from a persisted CSV row."""

    value = row.get(field)
    if value is None:
        raise ValueError(
            f"cannot align {path} with the resume checkpoint: it has no {field!r} column"
        )
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"cannot align {path} with the resume checkpoint: a row carries "
            f"{field}={value!r} instead of an integer; repair or remove that row "
            "before resuming"
        ) from exc


def _rows_through(field: str, limit: int, *, path: Path) -> Callable[[Mapping[str, str]], bool]:
    """Return a predicate keeping the rows whose ``field`` is at or below ``limit``."""

    def keep(row: Mapping[str, str]) -> bool:
        return _csv_progress_value(row, field, path=path) <= limit

    return keep


def _truncate_uncommitted_csv_rows(
    artifacts: SeedArtifacts,
    *,
    update: int,
    global_step: int,
) -> dict[str, int]:
    """Drop CSV rows recorded past the update a resume checkpoint committed.

    ``episodes.csv``, ``metrics.csv``, and ``evaluation.csv`` are fsynced before
    the checkpoint of the same update, so a process killed inside that window
    leaves rows describing work the resumed run repeats.  Each file is rewritten
    atomically, keeping its header and every row at or before ``update``
    (``global_step`` for ``episodes.csv``, whose rows carry no update column).
    Returns the number of removed rows per artifact.
    """

    episodes_path = artifacts.path / "episodes.csv"
    evaluation_path = artifacts.path / "evaluation.csv"
    removed = {
        "episodes.csv": artifacts.truncate_csv(
            _rows_through("global_step", global_step, path=episodes_path),
            name="episodes",
        ),
        "metrics.csv": artifacts.truncate_csv(
            _rows_through("update", update, path=artifacts.metrics_path),
            name="metrics",
        ),
        "evaluation.csv": artifacts.truncate_csv(
            _rows_through("update", update, path=evaluation_path),
            name="evaluation",
        ),
    }
    for name, count in removed.items():
        if count:
            LOGGER.warning(
                "Discarded %d %s row(s) written after the committed update=%d",
                count,
                name,
                update,
            )
    return removed


def train_single(
    config: ExperimentConfig,
    *,
    task: str,
    seed: int,
    artifacts: SeedArtifacts,
    max_updates: int | None = None,
    verbose: bool = True,
    method: str | None = None,
    result_dir: str | None = None,
    resume_checkpoint: str | Path | None = None,
    evaluate_at_end: bool = True,
    segment_updates: int | None = None,
) -> dict[str, Any]:
    if not isinstance(evaluate_at_end, bool):
        raise TypeError("evaluate_at_end must be boolean")
    if max_updates is not None and segment_updates is not None:
        raise ValueError("max_updates and segment_updates are mutually exclusive")
    if max_updates is not None and (
        isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates <= 0
    ):
        raise ValueError("max_updates must be positive when supplied")
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
    device = resolve_device(experiment_config["device"])
    seed_everything(seed, bool(experiment_config["deterministic"]))
    configure_logging(artifacts.path / "run.log", verbose=verbose)

    rollout_batch_size = int(ppo_config["num_envs"]) * int(ppo_config["rollout_steps"])
    configured_updates = math.ceil(int(ppo_config["total_steps"]) / rollout_batch_size)
    bootstrap_on_truncation = bool(ppo_config["bootstrap_on_truncation"])
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
    LOGGER.info("Starting task=%s seed=%d device=%s", task, seed, device)

    environments = [
        make_env(task, environment_config, seed=seed * 10_000 + index)
        for index in range(int(ppo_config["num_envs"]))
    ]
    batch = SyncEnvironmentBatch(environments)
    first_environment = environments[0]
    model = model_for_environment(first_environment, model_config, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(ppo_config["learning_rate"]), eps=1e-5
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Model parameters: %d", parameter_count)

    reset_seeds = [seed * 10_000 + index for index in range(batch.size)]
    observation = batch.reset(reset_seeds)
    context = BeliefContext(
        observation=torch.as_tensor(observation, dtype=torch.float32, device=device),
        previous_particles=torch.zeros(
            batch.size,
            int(model_config["num_particles"]),
            first_environment.state_dim,
            dtype=torch.float32,
            device=device,
        ),
        previous_action=torch.zeros(
            batch.size,
            first_environment.action_spec.feature_dim,
            dtype=torch.float32,
            device=device,
        ),
        initial=torch.ones(batch.size, dtype=torch.bool, device=device),
    )
    belief_histories = [BeliefReplayHistory() for _ in range(batch.size)]

    episode_returns = np.zeros(batch.size, dtype=np.float64)
    episode_lengths = np.zeros(batch.size, dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
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
        try:
            (
                context,
                belief_histories,
                episode_returns,
                episode_lengths,
                completed_returns,
                completed_lengths,
                metric_rows,
                best_evaluation,
                latest_evaluation,
                accumulated_wall_time,
            ) = _restore_trainer_state(
                checkpoint["trainer_state"],
                environments=environments,
                update=start_update,
                rollout_batch_size=rollout_batch_size,
                model_config=model_config,
                device=device,
            )
        except (TypeError, ValueError) as exc:
            # Rejected state (non-finite beliefs above all) stays fail-fast, but
            # the operator needs the concrete rollback target in the message.
            assert resume_checkpoint is not None
            raise type(exc)(
                f"{exc}{_checkpoint_rollback_guidance(resume_checkpoint)}"
            ) from exc
        _restore_global_rng_states(checkpoint, device=device)
        LOGGER.info(
            "Resuming task=%s seed=%d from update=%d step=%d",
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
        trainer_state = _trainer_state_payload(
            environments=environments,
            context=context,
            belief_histories=belief_histories,
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
            model,
            optimizer,
            task=task,
            seed=seed,
            update=update,
            global_step=global_step,
            config=resolved,
            trainer_state=trainer_state,
            method=method,
        )

    for update in range(start_update + 1, total_updates + 1):
        update_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rollout_started = _start_timed_phase(
            "rollout_collection",
            update=update,
            total_updates=total_updates,
            device=device,
        )
        prefixes = tuple(
            history.snapshot(
                context,
                environment_index,
                langevin_steps=int(model_config["langevin_steps"]),
            )
            for environment_index, history in enumerate(belief_histories)
        )
        builder = RolloutBuilder(belief_prefixes=prefixes)
        update_returns: list[float] = []
        update_lengths: list[int] = []
        # Keep rollout diagnostics on the accelerator and transfer their small
        # aggregate once at the phase boundary.  Calling ``.cpu()``/``.item()``
        # inside every environment step serialises the CUDA stream and can make
        # a diagnostic substantially more expensive than the computation it
        # measures.
        diagnostic_float = torch.zeros((), dtype=torch.float64, device=device)
        diagnostic_count = torch.zeros((), dtype=torch.int64, device=device)
        score_norm_sum_device = diagnostic_float.clone()
        score_norm_count = 0
        initial_score_norm_sum_device = diagnostic_float.clone()
        initial_score_norm_count_device = diagnostic_count.clone()
        recursive_score_norm_sum_device = diagnostic_float.clone()
        recursive_score_norm_count_device = diagnostic_count.clone()
        initial_particle_norm_sum_device = diagnostic_float.clone()
        initial_particle_norm_count_device = diagnostic_count.clone()
        recursive_particle_norm_sum_device = diagnostic_float.clone()
        recursive_particle_norm_count_device = diagnostic_count.clone()
        score_nonfinite_count_device = diagnostic_count.clone()
        particle_nonfinite_count_device = diagnostic_count.clone()
        particle_abs_max_device = torch.zeros((), dtype=torch.float32, device=device)
        particle_clip_fraction_sum_device = diagnostic_float.clone()
        action_saturation_fraction_sum_device = diagnostic_float.clone()
        action_saturation_fraction_count = 0
        truncation_bootstrap_sum = 0.0
        truncation_bootstrap_count = 0
        episode_rows: list[dict[str, Any]] = []
        model.eval()

        for rollout_index in range(int(ppo_config["rollout_steps"])):
            # Diagnostics describe the belief sampled for the current input,
            # before ``_next_context`` advances/reset the vector environments.
            belief_initial = context.initial.detach()
            particles, scores, ula_initial_noise, ula_step_noises = (
                _sample_belief_for_rollout(model, context, model_config)
            )
            (
                action,
                old_log_prob,
                _,
                diffusion_chain,
                pre_tanh_action,
                old_value,
            ) = _policy_from_particles(model, particles, scores, deterministic=False)
            environment_action = _environment_actions(first_environment.action_spec, action)
            # ``action_features`` is a pure function of the sampled action.  It
            # is materialised here because a truncation bootstrap conditions the
            # discarded final observation on the action that produced it.
            action_features = encode_action(first_environment.action_spec, action)
            next_observation, reward, done, step_infos = batch.step(environment_action)
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=device)
            done_tensor = torch.as_tensor(done, dtype=torch.bool, device=device)
            truncation_values: torch.Tensor | None = None
            if bootstrap_on_truncation:
                truncation_values = truncation_bootstrap_values(
                    model,
                    model_config,
                    infos=step_infos,
                    particles=particles,
                    action_features=action_features,
                    device=device,
                    value_transform=value_transform,
                )
                truncation_bootstrap_sum += float(truncation_values.sum().cpu())
                truncation_bootstrap_count += len(truncated_step_indices(step_infos))

            builder.append(
                observations=context.observation,
                previous_particles=context.previous_particles,
                previous_actions=context.previous_action,
                initial=context.initial,
                particles=particles,
                actions=action,
                old_log_prob=old_log_prob,
                old_values=value_transform.to_raw(old_value),
                rewards=reward_tensor,
                dones=done_tensor,
                ula_initial_noise=ula_initial_noise,
                ula_step_noises=ula_step_noises,
                diffusion_chains=diffusion_chain,
                pre_tanh_actions=pre_tanh_action,
                truncation_values=truncation_values,
            )
            for environment_index, history in enumerate(belief_histories):
                history.append(
                    context,
                    environment_index,
                    ula_initial_noise,
                    ula_step_noises,
                )
                if bool(done[environment_index]):
                    history.clear()
            if first_environment.action_spec.is_continuous:
                action_saturation_fraction_sum_device = (
                    action_saturation_fraction_sum_device
                    + (action_features.abs() >= 0.99).float().mean().double()
                )
                action_saturation_fraction_count += 1
            context = _next_context(
                observation=next_observation,
                particles=particles,
                action_features=action_features,
                done=done,
                device=device,
            )
            episode_returns += reward
            episode_lengths += 1
            score_l2 = scores.norm(dim=-1)
            particle_l2 = particles.norm(dim=-1)
            initial_mask = belief_initial[:, None].expand_as(score_l2)
            recursive_mask = ~initial_mask
            finite_score_l2 = torch.nan_to_num(score_l2, nan=0.0, posinf=0.0, neginf=0.0)
            finite_particle_l2 = torch.nan_to_num(
                particle_l2, nan=0.0, posinf=0.0, neginf=0.0
            )
            score_norm_sum_device = score_norm_sum_device + finite_score_l2.sum().double()
            score_norm_count += score_l2.numel()
            initial_score_norm_sum_device = (
                initial_score_norm_sum_device
                + (finite_score_l2 * initial_mask).sum().double()
            )
            initial_score_norm_count_device = (
                initial_score_norm_count_device + initial_mask.sum()
            )
            initial_particle_norm_sum_device = (
                initial_particle_norm_sum_device
                + (finite_particle_l2 * initial_mask).sum().double()
            )
            initial_particle_norm_count_device = (
                initial_particle_norm_count_device + initial_mask.sum()
            )
            recursive_score_norm_sum_device = (
                recursive_score_norm_sum_device
                + (finite_score_l2 * recursive_mask).sum().double()
            )
            recursive_score_norm_count_device = (
                recursive_score_norm_count_device + recursive_mask.sum()
            )
            recursive_particle_norm_sum_device = (
                recursive_particle_norm_sum_device
                + (finite_particle_l2 * recursive_mask).sum().double()
            )
            recursive_particle_norm_count_device = (
                recursive_particle_norm_count_device + recursive_mask.sum()
            )
            score_nonfinite_count_device = (
                score_nonfinite_count_device + (~torch.isfinite(scores)).sum()
            )
            particle_nonfinite_count_device = (
                particle_nonfinite_count_device + (~torch.isfinite(particles)).sum()
            )
            particle_abs_max_device = torch.maximum(
                particle_abs_max_device, particles.abs().max()
            )
            particle_clip = float(model_config["particle_clip"])
            if particle_clip > 0.0:
                particle_clip_fraction_sum_device = (
                    particle_clip_fraction_sum_device
                    + (particles.abs() >= particle_clip - 1e-6).float().mean().double()
                )
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

        episode_appender.append_many(episode_rows)
        if bootstrap_on_truncation:
            # Reported through the log rather than metrics.csv: METRIC_FIELDS is
            # a closed CSV schema that every existing run and every reader
            # shares, and making it depend on a config flag would fork the
            # artifact contract for one diagnostic.
            LOGGER.info(
                "truncation_bootstrap update=%d truncations=%d value_mean=%s",
                update,
                truncation_bootstrap_count,
                (
                    f"{truncation_bootstrap_sum / truncation_bootstrap_count:.6f}"
                    if truncation_bootstrap_count
                    else "n/a"
                ),
            )
        bootstrap_particles, bootstrap_scores = _sample_belief(model, context, model_config)
        with torch.no_grad():
            last_value = value_transform.to_raw(
                model.value(model.encode(bootstrap_particles, bootstrap_scores))
            )
        rollout = builder.finish(
            last_value=last_value,
            gamma=float(ppo_config["gamma"]),
            gae_lambda=float(ppo_config["gae_lambda"]),
        )
        _finish_timed_phase(
            "rollout_collection",
            rollout_started,
            update=update,
            total_updates=total_updates,
            device=device,
        )
        score_norm_sum = float(score_norm_sum_device.cpu())
        initial_score_norm_sum = float(initial_score_norm_sum_device.cpu())
        initial_score_norm_count = int(initial_score_norm_count_device.cpu())
        recursive_score_norm_sum = float(recursive_score_norm_sum_device.cpu())
        recursive_score_norm_count = int(recursive_score_norm_count_device.cpu())
        initial_particle_norm_sum = float(initial_particle_norm_sum_device.cpu())
        initial_particle_norm_count = int(initial_particle_norm_count_device.cpu())
        recursive_particle_norm_sum = float(recursive_particle_norm_sum_device.cpu())
        recursive_particle_norm_count = int(recursive_particle_norm_count_device.cpu())
        score_nonfinite_count = int(score_nonfinite_count_device.cpu())
        particle_nonfinite_count = int(particle_nonfinite_count_device.cpu())
        particle_abs_max = float(particle_abs_max_device.cpu())
        particle_clip_fraction = float(particle_clip_fraction_sum_device.cpu()) / int(
            ppo_config["rollout_steps"]
        )
        action_saturation_fraction = (
            float(action_saturation_fraction_sum_device.cpu())
            / action_saturation_fraction_count
            if action_saturation_fraction_count
            else float("nan")
        )
        model.train()
        ppo_started = _start_timed_phase(
            "ppo_update",
            update=update,
            total_updates=total_updates,
            device=device,
        )
        ppo_metrics = update_ppo(model, optimizer, rollout, model_config, ppo_config)
        _finish_timed_phase(
            "ppo_update",
            ppo_started,
            update=update,
            total_updates=total_updates,
            device=device,
        )
        recondition_started = _start_timed_phase(
            "recondition",
            update=update,
            total_updates=total_updates,
            device=device,
        )
        context = _recondition_belief_context(
            model,
            context,
            belief_histories,
            model_config,
        )
        _finish_timed_phase(
            "recondition",
            recondition_started,
            update=update,
            total_updates=total_updates,
            device=device,
        )

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
            "score_l2_mean": score_norm_sum / score_norm_count,
            "initial_score_l2_mean": (
                initial_score_norm_sum / initial_score_norm_count
                if initial_score_norm_count
                else float("nan")
            ),
            "recursive_score_l2_mean": (
                recursive_score_norm_sum / recursive_score_norm_count
                if recursive_score_norm_count
                else float("nan")
            ),
            "initial_particle_l2_mean": (
                initial_particle_norm_sum / initial_particle_norm_count
                if initial_particle_norm_count
                else float("nan")
            ),
            "recursive_particle_l2_mean": (
                recursive_particle_norm_sum / recursive_particle_norm_count
                if recursive_particle_norm_count
                else float("nan")
            ),
            "initial_belief_particle_count": initial_particle_norm_count,
            "recursive_belief_particle_count": recursive_particle_norm_count,
            "score_nonfinite_count": score_nonfinite_count,
            "particle_nonfinite_count": particle_nonfinite_count,
            "particle_abs_max": particle_abs_max,
            "particle_clip_fraction": particle_clip_fraction,
            "action_saturation_fraction": action_saturation_fraction,
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
                "update=%d/%d step=%d policy=%.4f value=%.4f return=%s fps=%.1f",
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
            evaluation_started = _start_timed_phase(
                "evaluation",
                update=update,
                total_updates=total_updates,
                device=device,
            )
            evaluation_seed = seed + 1_000_000
            if first_environment.action_spec.is_discrete:
                deterministic_mode = "greedy"
            elif model.continuous_policy_kind == "diffusion":
                deterministic_mode = "mean_chain"
            else:
                deterministic_mode = "mean_action"
            for mode, deterministic_evaluation in (
                (deterministic_mode, True),
                ("stochastic", False),
            ):
                evaluation, trajectories = evaluate(
                    model,
                    task=task,
                    environment_config=environment_config,
                    model_config=model_config,
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
                    "evaluation mode=%s update=%d return=%.3f +/- %.3f",
                    mode,
                    update,
                    evaluation["mean_return"],
                    evaluation["std_return"],
                )
                if deterministic_evaluation:
                    latest_evaluation = evaluation
            _finish_timed_phase(
                "evaluation",
                evaluation_started,
                update=update,
                total_updates=total_updates,
                device=device,
            )
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

    # ``latest_evaluation`` is the deterministic-readout evaluation of the last
    # evaluated update, so both keys below intentionally hold the same number.
    readout_return_mean = None if latest_evaluation is None else latest_evaluation["mean_return"]
    summary = {
        "task": task,
        "seed": seed,
        "status": "completed" if total_updates == configured_updates else "partial",
        "updates": total_updates,
        "global_steps": total_updates * rollout_batch_size,
        "episodes_completed": len(completed_returns),
        "training_return_mean": (float(np.mean(completed_returns)) if completed_returns else None),
        "evaluation_return_mean": readout_return_mean,
        "deterministic_readout_return_mean": readout_return_mean,
        "parameter_count": parameter_count,
        "wall_time_seconds": accumulated_wall_time + (time.perf_counter() - segment_started),
        "result_dir": result_dir or f"{task}/seed_{seed:03d}",
    }
    LOGGER.info(
        "%s task=%s seed=%d at update=%d/%d",
        "Completed" if total_updates == configured_updates else "Paused",
        task,
        seed,
        total_updates,
        configured_updates,
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config, args.override)
    label = f"{config['experiment']['label']}-{args.task}-seed{args.seed}"
    execution_data = config.to_dict()
    execution_data["experiment"]["label"] = label
    execution_data["experiment"]["tasks"] = [args.task]
    execution_data["experiment"]["seeds"] = [args.seed]
    if args.max_updates is not None:
        if args.max_updates <= 0:
            raise ValueError("--max-updates must be positive")
        rollout_batch = execution_data["ppo"]["num_envs"] * execution_data["ppo"]["rollout_steps"]
        execution_data["ppo"]["total_steps"] = min(
            execution_data["ppo"]["total_steps"],
            args.max_updates * rollout_batch,
        )
    config = ExperimentConfig(execution_data, source=config.source)
    run = RunArtifacts.create(config["experiment"]["output_dir"], label)
    run.write_resolved_config(config.to_dict())
    started_at = datetime.now(UTC).isoformat()
    root_metadata = runtime_metadata(resolve_device(config["experiment"]["device"]))
    root_metadata.update({"started_at": started_at, "tasks": [args.task], "seeds": [args.seed]})
    run.write_metadata(root_metadata)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "tasks": [args.task],
        "seeds": [args.seed],
        "max_updates": args.max_updates,
        "runs": [],
    }
    run.write_manifest(manifest)
    artifacts = run.for_seed(args.task, args.seed)
    try:
        summary = train_single(
            config,
            task=args.task,
            seed=args.seed,
            artifacts=artifacts,
            max_updates=args.max_updates,
        )
    except Exception:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        run.write_manifest(manifest)
        raise
    summary["error"] = ""
    run.append_summary(summary, SUMMARY_FIELDS)
    run.write_summary([summary])
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["runs"] = [summary]
    run.write_manifest(manifest)
    print(run.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
