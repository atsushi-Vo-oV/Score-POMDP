"""TD3 (twin delayed deep deterministic policy gradient) on top of every compared belief model.

The belief part of a model is unchanged: the score belief (``ScoreBeliefActorCritic``: energies,
particle transport, set encoder) or a recurrent baseline (GRU, RNN, DPFRL).  Its policy
condition ``c_t`` feeds a deterministic actor ``mu(c)`` (tanh, normalised action space) and twin
critics ``Q_1, Q_2(c, a)`` with Polyak-averaged target copies.  The critic TD loss trains the
belief through ``c_t``; the actor reads a detached condition (shared-encoder TD3).  The target
condition ``c_{t+1}`` comes from the online belief, detached.

Off-policy replay of a belief needs the belief recomputed under the current parameters.  Every
step stores a 63-bit noise seed instead of the noise itself: the particle transport noise (score
models) and the DPFRL resampling noise are regenerated from it, so a replayed belief equals the
one the current parameters would have produced online.  A training sample is a window of
``td3_sequence_length`` steps; the episode prefix before it is replayed without gradient (exact
burn-in), the window itself with gradient.

The update loop follows :mod:`sb_pomdp.prediction_schedule`: ``pretrain`` updates collect with
uniformly random actions and train only the next-observation predictor (they also fill the
replay buffer, playing TD3's random warm-up), ``predict`` updates collect with the current actor
and train the predictor on the freshly collected data, ``ppo`` updates are TD3 updates (the name
is the schedule's).  Each update collects ``num_envs * rollout_steps`` environment steps.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .artifacts import SeedArtifacts
from .config import ExperimentConfig
from .envs import make_env
from .networks import NetworkKinds, make_head_network, use_network_kinds
from .ppo import clip_gradients_or_skip, condition_prediction_loss
from .prediction_schedule import PredictionSchedule
from .train import (
    EPISODE_FIELDS,
    EVALUATION_FIELDS,
    METRIC_FIELDS,
    _log_langevin_schedules,
    belief_latent_dim,
    build_optimizer,
    configure_logging,
    model_for_environment,
    resolve_device,
    runtime_metadata,
    save_checkpoint,
    seed_everything,
)
from .train_baselines import _make_model

LOGGER = logging.getLogger(__name__)
TD3_CHECKPOINT_FORMAT = "td3-v1"
SCORE_METHODS = frozenset({"score_transformer", "score_deepsets", "score_alpha"})
RECURRENT_METHODS = frozenset({"gru", "rnn", "particle_filter"})
TD3_DEFAULTS: dict[str, Any] = {
    "td3_gradient_steps": 16,
    "td3_batch_sequences": 16,
    "td3_sequence_length": 32,
    "td3_tau": 0.005,
    "td3_policy_delay": 2,
    "td3_exploration_noise": 0.1,
    "td3_target_noise": 0.2,
    "td3_target_noise_clip": 0.5,
    "td3_actor_learning_rate": 3e-4,
    "td3_critic_learning_rate": 3e-4,
}


def _generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


class BeliefAdapter:
    """One per-step interface over the score belief and the recurrent baselines.

    ``step`` maps (observation, previous action, episode-start flag, noise seed) and the carried
    state to the policy condition and the next state.  The state is the previous particle set
    ``[B, K, D]`` for score models and the packed hidden ``[layers, B, H]`` for baselines.
    """

    def __init__(self, model: nn.Module, method: str, model_config: Mapping[str, Any], device: torch.device) -> None:
        self.model = model
        self.method = method
        self.score = method in SCORE_METHODS
        self.model_config = model_config
        self.device = device
        if self.score:
            self.condition_dim = int(model_config["d_model"])
        else:
            self.condition_dim = int(model.policy_condition_dim)

    def initial_state(self, batch: int, *, latent_dim: int | None = None) -> torch.Tensor:
        if self.score:
            assert latent_dim is not None
            return torch.zeros(batch, int(self.model_config["num_particles"]), latent_dim, device=self.device)
        return self.model.initial_hidden(batch, device=self.device)

    def mask_state(self, new: torch.Tensor, old: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        if self.score:
            return torch.where(active.view(-1, 1, 1), new, old)
        return torch.where(active.view(1, -1, 1), new, old)

    def step(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
        seeds: np.ndarray,
        *,
        track_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = observation.shape[0]
        with torch.set_grad_enabled(track_grad):
            if self.score:
                belief = self.model.belief
                noises = [
                    belief.draw_particle_noise(
                        observation[index : index + 1],
                        num_particles=int(self.model_config["num_particles"]),
                        num_steps=int(self.model_config["langevin_steps"]),
                        generator=_generator(seeds[index], self.device),
                    )
                    for index in range(batch)
                ]
                initial_noise = torch.cat([n[0] for n in noises])
                step_noise = torch.cat([n[1] for n in noises])
                particles, scores = belief.particles_from_noise(
                    observation,
                    state,
                    previous_action,
                    initial,
                    initial_noise=initial_noise,
                    langevin_noise=step_noise,
                    step_size=float(self.model_config["langevin_step_size"]),
                    track_grad=track_grad,
                )
                condition = self.model.encode(particles, scores)
                return condition, particles
            inputs = torch.cat((observation, previous_action), dim=-1)
            noise_dim = int(getattr(self.model, "noise_dim", 0) or 0)
            if noise_dim:
                noise = torch.cat(
                    [self.model.draw_noise(1, generator=_generator(seeds[index], self.device), device=self.device)
                     for index in range(batch)]
                )
                inputs = torch.cat((inputs, noise), dim=-1)
            features, hidden = self.model.step(inputs, state, initial)
            return features, hidden


class TD3Heads(nn.Module):
    """Deterministic actor and twin critics on the policy condition (normalised actions)."""

    def __init__(self, condition_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.actor = make_head_network(condition_dim, hidden_dims, action_dim)
        self.critic_1 = make_head_network(condition_dim + action_dim, hidden_dims, 1)
        self.critic_2 = make_head_network(condition_dim + action_dim, hidden_dims, 1)

    def act(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(condition))

    def q(self, condition: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat((condition, action), dim=-1)
        return self.critic_1(joint).squeeze(-1), self.critic_2(joint).squeeze(-1)


class SequenceReplay:
    """Per-environment step streams; every step keeps its episode-start index and noise seed."""

    FIELDS = ("observation", "previous_action", "initial", "action", "reward", "terminated", "truncated",
              "seed", "episode_start")

    def __init__(self, num_envs: int, capacity: int, observation_dim: int, action_dim: int) -> None:
        self.num_envs, self.capacity = num_envs, capacity
        self.observation = np.zeros((num_envs, capacity, observation_dim), dtype=np.float32)
        self.previous_action = np.zeros((num_envs, capacity, action_dim), dtype=np.float32)
        self.initial = np.zeros((num_envs, capacity), dtype=bool)
        self.action = np.zeros((num_envs, capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((num_envs, capacity), dtype=np.float32)
        self.terminated = np.zeros((num_envs, capacity), dtype=bool)
        self.truncated = np.zeros((num_envs, capacity), dtype=bool)
        self.seed = np.zeros((num_envs, capacity), dtype=np.int64)
        self.episode_start = np.zeros((num_envs, capacity), dtype=np.int64)
        self.size = 0
        self.current_start = np.zeros(num_envs, dtype=np.int64)

    def add(self, *, observation, previous_action, initial, action, reward, terminated, truncated, seed) -> None:
        if self.size >= self.capacity:
            raise RuntimeError("TD3 replay buffer is full; it is sized to hold the whole run")
        i = self.size
        self.current_start = np.where(initial, i, self.current_start)
        self.observation[:, i] = observation
        self.previous_action[:, i] = previous_action
        self.initial[:, i] = initial
        self.action[:, i] = action
        self.reward[:, i] = reward
        self.terminated[:, i] = terminated
        self.truncated[:, i] = truncated
        self.seed[:, i] = seed
        self.episode_start[:, i] = self.current_start
        self.size += 1

    def state_dict(self) -> dict[str, Any]:
        out = {name: getattr(self, name)[:, : self.size].copy() for name in self.FIELDS}
        out.update(size=self.size, current_start=self.current_start.copy())
        return out

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        size = int(state["size"])
        for name in self.FIELDS:
            getattr(self, name)[:, :size] = state[name]
        self.size, self.current_start = size, np.asarray(state["current_start"], dtype=np.int64)

    def sample(self, rng: np.random.Generator, batch: int, length: int, *, lowest: int = 0) -> dict[str, np.ndarray]:
        """Windows of ``length + 1`` steps (the last one only bootstraps) and their burn-in prefixes."""

        envs = rng.integers(self.num_envs, size=batch)
        starts = rng.integers(lowest, self.size, size=batch)
        offsets = np.arange(length + 1)
        index = np.minimum(starts[:, None] + offsets[None, :], self.size - 1)
        same_episode = (starts[:, None] + offsets[None, :] < self.size) & (
            self.episode_start[envs[:, None], index] == self.episode_start[envs, starts][:, None]
        )
        return {"envs": envs, "starts": starts, "index": index, "valid": same_episode,
                "episode_start": self.episode_start[envs, starts]}


def _replay_conditions(
    adapter: BeliefAdapter,
    replay: SequenceReplay,
    sample: Mapping[str, np.ndarray],
    *,
    latent_dim: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Conditions ``[length + 1, B, d]`` of the sampled windows, exact burn-in before each."""

    envs, starts, index = sample["envs"], sample["starts"], sample["index"]
    prefix = starts - sample["episode_start"]
    batch = len(envs)
    state = adapter.initial_state(batch, latent_dim=latent_dim)

    def gather(rows: np.ndarray) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor(replay.observation[envs, rows], device=device),
            torch.as_tensor(replay.previous_action[envs, rows], device=device),
            torch.as_tensor(replay.initial[envs, rows], device=device),
        )

    for tau in range(int(prefix.max()) if batch else 0):
        rows = np.minimum(sample["episode_start"] + tau, np.maximum(starts - 1, 0))
        active = torch.as_tensor(tau < prefix, device=device)
        observation, previous_action, initial = gather(rows)
        _, new_state = adapter.step(state, observation, previous_action, initial, replay.seed[envs, rows],
                                    track_grad=False)
        state = adapter.mask_state(new_state.detach(), state, active)
    conditions = []
    for k in range(index.shape[1]):
        rows = index[:, k]
        observation, previous_action, initial = gather(rows)
        condition, state = adapter.step(state, observation, previous_action, initial, replay.seed[envs, rows],
                                        track_grad=True)
        conditions.append(condition)
    return torch.stack(conditions)


@torch.no_grad()
def _rebuild_acting_state(adapter: BeliefAdapter, replay: SequenceReplay, initial: np.ndarray, *,
                          latent_dim: int | None, device: torch.device) -> torch.Tensor:
    """Belief state of every unfinished episode, replayed from its stored prefix with the current parameters."""

    num_envs = replay.num_envs
    state = adapter.initial_state(num_envs, latent_dim=latent_dim)
    if not replay.size:
        return state
    prefix = np.where(initial, 0, replay.size - replay.current_start)
    all_envs = np.arange(num_envs)
    for tau in range(int(prefix.max())):
        rows = np.minimum(replay.current_start + tau, replay.size - 1)
        _, new_state = adapter.step(
            state, torch.as_tensor(replay.observation[all_envs, rows], device=device),
            torch.as_tensor(replay.previous_action[all_envs, rows], device=device),
            torch.as_tensor(replay.initial[all_envs, rows], device=device), replay.seed[all_envs, rows],
            track_grad=False)
        state = adapter.mask_state(new_state.detach(), state, torch.as_tensor(tau < prefix, device=device))
    return state


def _to_env_action(normalised: torch.Tensor, spec: Any) -> np.ndarray:
    low = np.asarray(spec.low, dtype=np.float32)
    high = np.asarray(spec.high, dtype=np.float32)
    array = normalised.detach().cpu().numpy()
    return (low + (array + 1.0) * 0.5 * (high - low)).astype(np.float32)


@torch.no_grad()
def _evaluate(adapter, heads, *, task, environment_config, episodes, seed, exploration_noise, deterministic,
              latent_dim, device) -> dict[str, float]:
    returns, lengths = [], []
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        env = make_env(task, environment_config, seed=seed + episode)
        observation, _ = env.reset(seed=seed + episode)
        state = adapter.initial_state(1, latent_dim=latent_dim)
        previous_action = torch.zeros(1, env.action_spec.feature_dim, device=device)
        initial = torch.ones(1, dtype=torch.bool, device=device)
        total, length = 0.0, 0
        while True:
            obs = torch.as_tensor(observation[None], dtype=torch.float32, device=device)
            condition, state = adapter.step(state, obs, previous_action, initial,
                                            rng.integers(1 << 62, size=1), track_grad=False)
            action = heads.act(condition)
            if not deterministic:
                action = (action + exploration_noise * torch.randn_like(action)).clamp(-1.0, 1.0)
            observation, reward, terminated, truncated, _ = env.step(_to_env_action(action, env.action_spec)[0])
            total += float(reward)
            length += 1
            previous_action, initial = action, torch.zeros(1, dtype=torch.bool, device=device)
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(length)
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths))}


def train_td3_single(
    config: ExperimentConfig,
    *,
    method: str,
    task: str,
    seed: int,
    artifacts: SeedArtifacts,
    max_updates: int | None = None,
    verbose: bool = True,
    result_dir: str,
    resume_checkpoint: str | Path | None = None,
    segment_updates: int | None = None,
) -> dict[str, Any]:
    """Train one belief model with TD3 and write one seed artifact tree (resumable in segments)."""

    if method not in SCORE_METHODS | RECURRENT_METHODS:
        raise ValueError(f"TD3 supports the score models and the recurrent baselines, not {method!r}")
    resolved = config.to_dict()
    experiment_config, model_config, ppo_config = resolved["experiment"], resolved["model"], resolved["ppo"]
    td3 = {key: ppo_config.get(key, default) for key, default in TD3_DEFAULTS.items()}
    environment_config = {"name": task, **config.environment_for_task(task)}
    device = resolve_device(experiment_config["device"])
    seed_everything(seed, bool(experiment_config["deterministic"]))
    configure_logging(artifacts.path / "run.log", verbose=verbose)

    num_envs, rollout_steps = int(ppo_config["num_envs"]), int(ppo_config["rollout_steps"])
    rollout_batch_size = num_envs * rollout_steps
    schedule = PredictionSchedule.from_config(
        model_config, rollout_batch_size=rollout_batch_size,
        ppo_updates=math.ceil(int(ppo_config["total_steps"]) / rollout_batch_size),
    )
    configured_updates = schedule.total_updates

    environments = [make_env(task, environment_config, seed=seed * 10_000 + i) for i in range(num_envs)]
    first = environments[0]
    if not first.action_spec.is_continuous:
        raise ValueError("TD3 needs a continuous action space")
    action_dim = first.action_spec.feature_dim
    latent_dim = None
    with use_network_kinds(NetworkKinds.from_config(model_config)):
        if method in SCORE_METHODS:
            model = model_for_environment(first, model_config, device)
            latent_dim = belief_latent_dim(first, model_config)
        else:
            model = _make_model(first, method, model_config, resolved["comparison"], device)
    adapter = BeliefAdapter(model, method, model_config, device)
    heads = TD3Heads(adapter.condition_dim, action_dim, list(model_config["policy_hidden"])).to(device)
    target_heads = copy.deepcopy(heads).requires_grad_(False)
    model_optimizer = build_optimizer(model, dict(ppo_config, learning_rate=float(td3["td3_critic_learning_rate"])))
    critic_optimizer = torch.optim.Adam(
        list(heads.critic_1.parameters()) + list(heads.critic_2.parameters()), lr=float(td3["td3_critic_learning_rate"])
    )
    actor_optimizer = torch.optim.Adam(heads.actor.parameters(), lr=float(td3["td3_actor_learning_rate"]))
    replay = SequenceReplay(num_envs, configured_updates * rollout_steps, first.observation_dim, action_dim)
    rng = np.random.default_rng(seed * 7919 + 17)
    parameter_count = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in heads.parameters())

    start_update, critic_steps = 0, 0
    metric_rows: list[dict[str, Any]] = []
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    best_evaluation, latest_evaluation, accumulated_wall = -math.inf, None, 0.0
    observation = np.stack([env.reset(seed=seed * 10_000 + i)[0] for i, env in enumerate(environments)])
    previous_action = np.zeros((num_envs, action_dim), dtype=np.float32)
    initial = np.ones(num_envs, dtype=bool)
    episode_returns = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)

    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if payload.get("format") != TD3_CHECKPOINT_FORMAT or payload.get("config") != resolved:
            raise ValueError("TD3 resume checkpoint does not match this configuration")
        model.load_state_dict(payload["model"])
        heads.load_state_dict(payload["heads"])
        target_heads.load_state_dict(payload["target_heads"])
        model_optimizer.load_state_dict(payload["optimizers"]["model"])
        critic_optimizer.load_state_dict(payload["optimizers"]["critic"])
        actor_optimizer.load_state_dict(payload["optimizers"]["actor"])
        replay.load_state_dict(payload["replay"])
        for env, state in zip(environments, payload["environments"], strict=True):
            env.load_state_dict(state)
        pending = payload["pending"]
        observation, previous_action = pending["observation"], pending["previous_action"]
        initial, episode_returns, episode_lengths = pending["initial"], pending["episode_returns"], pending["episode_lengths"]
        rng.bit_generator.state = payload["rng"]["numpy"]
        torch.set_rng_state(payload["rng"]["torch"])
        if device.type == "cuda" and payload["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(payload["rng"]["cuda"], device)
        trainer = payload["trainer"]
        start_update, critic_steps = int(payload["update"]), int(trainer["critic_steps"])
        metric_rows, completed_returns = trainer["metric_rows"], trainer["completed_returns"]
        completed_lengths, best_evaluation = trainer["completed_lengths"], trainer["best_evaluation"]
        latest_evaluation, accumulated_wall = trainer["latest_evaluation"], trainer["wall_time_seconds"]
        LOGGER.info("Resuming TD3 method=%s task=%s seed=%d from update=%d", method, task, seed, start_update)

    if segment_updates is not None:
        total_updates = min(configured_updates, start_update + segment_updates)
    elif max_updates is not None:
        total_updates = min(configured_updates, max_updates)
    else:
        total_updates = configured_updates
    artifacts.write_resolved_config(resolved)
    metadata = runtime_metadata(device)
    metadata.update({"task": task, "seed": seed, "method": method, "algorithm": "td3",
                     "max_updates": max_updates, "segment_updates": segment_updates})
    artifacts.write_metadata(metadata)
    LOGGER.info("Starting TD3 method=%s task=%s seed=%d device=%s parameters=%d", method, task, seed, device,
                parameter_count)

    # Acting belief state: rebuild it from the stored prefix of every unfinished episode.
    state = _rebuild_acting_state(adapter, replay, initial, latent_dim=latent_dim, device=device)

    episode_appender = artifacts.csv_appender(EPISODE_FIELDS, name="episodes")
    evaluation_appender = artifacts.csv_appender(EVALUATION_FIELDS, name="evaluation")
    segment_started = time.perf_counter()
    spec = first.action_spec

    def checkpoint_payload(update: int) -> dict[str, Any]:
        return {
            "format": TD3_CHECKPOINT_FORMAT, "update": update, "global_step": update * rollout_batch_size,
            "config": resolved, "task": task, "seed": seed, "method": method,
            "model": model.state_dict(), "heads": heads.state_dict(), "target_heads": target_heads.state_dict(),
            "optimizers": {"model": model_optimizer.state_dict(), "critic": critic_optimizer.state_dict(),
                           "actor": actor_optimizer.state_dict()},
            "replay": replay.state_dict(), "environments": [env.state_dict() for env in environments],
            "pending": {"observation": observation, "previous_action": previous_action, "initial": initial,
                        "episode_returns": episode_returns, "episode_lengths": episode_lengths},
            "rng": {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
            "trainer": {"critic_steps": critic_steps, "metric_rows": metric_rows,
                        "completed_returns": completed_returns, "completed_lengths": completed_lengths,
                        "best_evaluation": best_evaluation, "latest_evaluation": latest_evaluation,
                        "wall_time_seconds": accumulated_wall + (time.perf_counter() - segment_started)},
        }

    for update in range(start_update + 1, total_updates + 1):
        update_started = time.perf_counter()
        phase = schedule.phase(update)
        lowest_recent = replay.size
        update_returns, update_lengths, episode_rows = [], [], []
        model.eval()
        for rollout_index in range(rollout_steps):
            seeds = rng.integers(1 << 62, size=num_envs)
            obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device)
            with torch.no_grad():
                condition, new_state = adapter.step(
                    state, obs_t, torch.as_tensor(previous_action, device=device),
                    torch.as_tensor(initial, device=device), seeds, track_grad=False,
                )
                if phase == "pretrain":
                    action = torch.rand(num_envs, action_dim, device=device) * 2.0 - 1.0
                else:
                    action = heads.act(condition)
                    action = (action + float(td3["td3_exploration_noise"]) * torch.randn_like(action)).clamp(-1, 1)
            state = new_state.detach()
            env_actions = _to_env_action(action, spec)
            results = [env.step(env_actions[i]) for i, env in enumerate(environments)]
            next_obs = np.stack([r[0] for r in results])
            rewards = np.asarray([r[1] for r in results], dtype=np.float32)
            terminated = np.asarray([r[2] for r in results], dtype=bool)
            truncated = np.asarray([r[3] for r in results], dtype=bool)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            replay.add(observation=observation, previous_action=previous_action, initial=initial, action=action_np,
                       reward=rewards, terminated=terminated, truncated=truncated, seed=seeds)
            episode_returns += rewards
            episode_lengths += 1
            done = terminated | truncated
            for i in np.flatnonzero(done):
                if phase != "pretrain":
                    update_returns.append(float(episode_returns[i]))
                    update_lengths.append(int(episode_lengths[i]))
                    completed_returns.append(float(episode_returns[i]))
                    completed_lengths.append(int(episode_lengths[i]))
                    episode_rows.append({"global_step": (update - 1) * rollout_batch_size + (rollout_index + 1) * num_envs,
                                         "environment": int(i), "episode_return": float(episode_returns[i]),
                                         "episode_length": int(episode_lengths[i])})
                next_obs[i] = environments[i].reset()[0]
                episode_returns[i], episode_lengths[i] = 0.0, 0
            observation = next_obs.astype(np.float32)
            previous_action = np.where(done[:, None], 0.0, action_np).astype(np.float32)
            initial = done.copy()
        episode_appender.append_many(episode_rows)

        model.train()
        losses = {"critic": [], "actor": [], "prediction": [], "gradient_norm": []}
        gradient_steps = int(td3["td3_gradient_steps"])
        batch, length = int(td3["td3_batch_sequences"]), int(td3["td3_sequence_length"])
        for _ in range(gradient_steps):
            if phase != "ppo":
                lowest = 0 if phase == "pretrain" else lowest_recent
                sample = replay.sample(rng, batch, length, lowest=lowest)
                conditions = _replay_conditions(adapter, replay, sample, latent_dim=latent_dim, device=device)
                valid = torch.as_tensor(sample["valid"], device=device)
                envs, index = sample["envs"], sample["index"]
                observations = torch.as_tensor(replay.observation[envs[:, None], index], device=device).transpose(0, 1)
                actions = torch.as_tensor(replay.action[envs[:, None], index], device=device).transpose(0, 1)
                # condition_prediction_loss reads the action at t from previous_actions[t + 1]
                # and masks with dones[t]; here the stored action at t is exact and a step
                # outside the episode is masked out.
                shifted = torch.cat((actions[:1], actions[:-1]), dim=0)
                invalid_next = ~valid.transpose(0, 1)
                dones = torch.cat((invalid_next[1:], torch.ones_like(invalid_next[:1])), dim=0)
                loss = condition_prediction_loss(
                    model.prediction_predictor, conditions.reshape(-1, conditions.shape[-1]),
                    observations=observations, previous_actions=shifted, dones=dones,
                )
                model_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm, finite = clip_gradients_or_skip(model.parameters(), float(ppo_config["max_grad_norm"]))
                if finite:
                    model_optimizer.step()
                losses["prediction"].append(float(loss.detach().cpu()))
                losses["gradient_norm"].append(float(norm.detach().cpu()))
                continue
            sample = replay.sample(rng, batch, length)
            conditions = _replay_conditions(adapter, replay, sample, latent_dim=latent_dim, device=device)
            envs, index, valid = sample["envs"], sample["index"], torch.as_tensor(sample["valid"], device=device).T
            actions = torch.as_tensor(replay.action[envs[:, None], index], device=device).transpose(0, 1)
            rewards = torch.as_tensor(replay.reward[envs[:, None], index], device=device).T
            terminal = torch.as_tensor(replay.terminated[envs[:, None], index], device=device).T.float()
            cut = torch.as_tensor(replay.truncated[envs[:, None], index], device=device).T
            mask = (valid[:-1] & (valid[1:] | terminal[:-1].bool()) & ~cut[:-1]).float()
            current, following = conditions[:-1], conditions[1:].detach()
            with torch.no_grad():
                noise = (float(td3["td3_target_noise"]) * torch.randn_like(actions[:-1])).clamp(
                    -float(td3["td3_target_noise_clip"]), float(td3["td3_target_noise_clip"]))
                next_action = (target_heads.act(following) + noise).clamp(-1.0, 1.0)
                target_1, target_2 = target_heads.q(following, next_action)
                target = rewards[:-1] + float(ppo_config["gamma"]) * (1.0 - terminal[:-1]) * torch.minimum(target_1, target_2)
            q_1, q_2 = heads.q(current, actions[:-1])
            critic_loss = (((q_1 - target).square() + (q_2 - target).square()) * mask).sum() / mask.sum().clamp_min(1.0)
            model_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            parameters = list(model.parameters()) + list(heads.critic_1.parameters()) + list(heads.critic_2.parameters())
            norm, finite = clip_gradients_or_skip(parameters, float(ppo_config["max_grad_norm"]))
            if finite:
                model_optimizer.step()
                critic_optimizer.step()
            critic_steps += 1
            losses["critic"].append(float(critic_loss.detach().cpu()))
            losses["gradient_norm"].append(float(norm.detach().cpu()))
            if critic_steps % int(td3["td3_policy_delay"]) == 0:
                detached = current.detach()
                actor_loss = -(heads.q(detached, heads.act(detached))[0] * valid[:-1].float()).sum() / valid[:-1].float().sum().clamp_min(1.0)
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_optimizer.step()
                losses["actor"].append(float(actor_loss.detach().cpu()))
                with torch.no_grad():
                    tau = float(td3["td3_tau"])
                    for target_parameter, parameter in zip(target_heads.parameters(), heads.parameters(), strict=True):
                        target_parameter.mul_(1.0 - tau).add_(tau * parameter)

        # The acting state was computed with the pre-update parameters; rebuild it like PPO's recondition.
        state = _rebuild_acting_state(adapter, replay, initial, latent_dim=latent_dim, device=device)

        global_step = update * rollout_batch_size
        mean = lambda values: float(np.mean(values)) if values else float("nan")  # noqa: E731
        row = {field: float("nan") for field in METRIC_FIELDS}
        row.update({
            "update": update, "global_step": global_step,
            "policy_loss": mean(losses["actor"]) if phase == "ppo" else mean(losses["prediction"]),
            "value_loss": mean(losses["critic"]), "gradient_norm": mean(losses["gradient_norm"]),
            "episode_return_mean": mean(update_returns), "episode_length_mean": mean(update_lengths),
            "episodes_completed": len(update_returns), "gpu_peak_memory_mb": 0.0,
            "steps_per_second": rollout_batch_size / max(time.perf_counter() - update_started, 1e-9),
        })
        metric_rows.append(row)
        artifacts.append_metrics(row, METRIC_FIELDS)
        LOGGER.info("method=%s phase=%s update=%d/%d step=%d critic=%.4f actor=%.4f prediction=%.4f return=%s fps=%.1f",
                    method, "td3" if phase == "ppo" else phase, update, total_updates, global_step, mean(losses["critic"]), mean(losses["actor"]),
                    mean(losses["prediction"]), "n/a" if not update_returns else f"{np.mean(update_returns):.3f}",
                    row["steps_per_second"])
        if method in SCORE_METHODS:
            _log_langevin_schedules(model, update)

        should_evaluate = (phase == "ppo" and update % int(experiment_config["eval_interval_updates"]) == 0) or (
            update == configured_updates)
        if should_evaluate:
            model.eval()
            for mode, deterministic in (("mean_action", True), ("stochastic", False)):
                evaluation = _evaluate(adapter, heads, task=task, environment_config=environment_config,
                                       episodes=int(experiment_config["eval_episodes"]), seed=seed + 1_000_000,
                                       exploration_noise=float(td3["td3_exploration_noise"]),
                                       deterministic=deterministic, latent_dim=latent_dim, device=device)
                evaluation_appender.append({"update": update, "global_step": global_step, "mode": mode,
                                            "evaluation_seed": seed + 1_000_000, **evaluation,
                                            "action_saturation_fraction": float("nan")})
                LOGGER.info("evaluation method=%s mode=%s update=%d return=%.3f +/- %.3f", method, mode, update,
                            evaluation["mean_return"], evaluation["std_return"])
                if deterministic:
                    latest_evaluation = evaluation
            if latest_evaluation["mean_return"] > best_evaluation:
                best_evaluation = latest_evaluation["mean_return"]
                save_checkpoint(checkpoint_payload(update), artifacts.checkpoint_path(tag="best"))
        if update % int(experiment_config["checkpoint_interval_updates"]) == 0 or update == total_updates:
            payload = checkpoint_payload(update)
            if update % int(experiment_config["checkpoint_interval_updates"]) == 0:
                save_checkpoint(payload, artifacts.checkpoint_path(step=global_step))
            save_checkpoint(payload, artifacts.checkpoint_path(tag="latest"))

    arrays = {"update": np.asarray([r["update"] for r in metric_rows], dtype=np.int64),
              "global_step": np.asarray([r["global_step"] for r in metric_rows], dtype=np.int64),
              "episode_returns": np.asarray(completed_returns, dtype=np.float32),
              "episode_lengths": np.asarray(completed_lengths, dtype=np.int32)}
    for field in METRIC_FIELDS:
        if field not in {"update", "global_step"}:
            arrays[field] = np.asarray([r[field] for r in metric_rows], dtype=np.float64)
    artifacts.save_npz(arrays)
    readout = None if latest_evaluation is None else latest_evaluation["mean_return"]
    summary = {
        "task": task, "seed": seed,
        "status": "completed" if total_updates == configured_updates else "partial",
        "updates": total_updates, "global_steps": total_updates * rollout_batch_size,
        "episodes_completed": len(completed_returns),
        "training_return_mean": float(np.mean(completed_returns)) if completed_returns else None,
        "evaluation_return_mean": readout, "deterministic_readout_return_mean": readout,
        "parameter_count": parameter_count,
        "wall_time_seconds": accumulated_wall + (time.perf_counter() - segment_started),
        "result_dir": result_dir,
    }
    LOGGER.info("%s TD3 method=%s task=%s seed=%d at update=%d/%d",
                "Completed" if total_updates == configured_updates else "Paused", method, task, seed,
                total_updates, configured_updates)
    return summary


__all__ = ["BeliefAdapter", "SequenceReplay", "TD3Heads", "TD3_DEFAULTS", "train_td3_single"]
