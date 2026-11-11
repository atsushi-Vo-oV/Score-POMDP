"""Head-only self-imitation on completed episodes and saved policy paths.

Conditions are collected before PPO and never re-encoded. Replay deliberately
optimizes a prioritized, positive-advantage objective without importance-ratio
correction; it is not an unbiased off-policy policy gradient.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch.distributions import Categorical

from .config import SELF_IMITATION_DEFAULTS
from .ppo import clip_gradients_or_skip
from .value_transform import ValueTransform


SIL_METRIC_FIELDS = (
    "sil_policy_loss", "sil_value_loss", "sil_selected_transitions",
    "sil_positive_transitions", "sil_mean_weight", "sil_buffer_size", "sil_updates",
)


def sil_config(ppo_config: Mapping) -> dict:
    return {**SELF_IMITATION_DEFAULTS, **ppo_config.get("self_imitation", {})}


def imitation_weights(advantages: torch.Tensor, config: Mapping, *, scale: torch.Tensor) -> torch.Tensor:
    """Detached weights; exponent is clamped BEFORE exp, including extreme returns."""
    advantage = advantages.detach()
    eligible = advantage > float(config["min_advantage"])
    if config["weighting"] == "relu":
        return torch.where(eligible, advantage.clamp_min(0), 0.0)
    beta = float(config["temperature"]) * scale.detach().clamp_min(1e-6)
    exponent = (advantage / beta).clamp(max=math.log(float(config["max_weight"])))
    return torch.where(eligible, exponent.exp(), 0.0)


def positive_sample(advantages: torch.Tensor, config: Mapping, *, generator=None) -> torch.Tensor:
    """Fresh positive support only, even when priority_exponent is zero."""
    indices = torch.nonzero(advantages > float(config["min_advantage"])).flatten()
    if not indices.numel():
        return indices
    # Log-space priorities avoid overflow for very large positive advantages.
    logits = float(config["priority_exponent"]) * advantages[indices].double().log()
    probabilities = (logits - logits.max()).exp()
    sampled = torch.multinomial(probabilities, int(config["batch_size"]), replacement=True, generator=generator)
    return indices[sampled]


class SelfImitationBuffer:
    """CPU ring of transitions plus per-environment unfinished episode suffixes."""

    def __init__(self, capacity: int, num_envs: int, gamma: float) -> None:
        if capacity <= 0 or num_envs <= 0 or not 0 < gamma <= 1:
            raise ValueError("invalid self-imitation buffer dimensions/gamma")
        self.capacity = capacity
        self.gamma = gamma
        self.size = 0
        self.position = 0
        self.data: dict[str, torch.Tensor] = {}
        self.pending: list[list[dict[str, torch.Tensor]]] = [[] for _ in range(num_envs)]

    def _append_episode(self, episode: list[dict[str, torch.Tensor]], bootstrap: float) -> None:
        future = bootstrap
        for row in reversed(episode):
            future = float(row["reward"]) + self.gamma * future
            row["returns"] = torch.tensor(future, dtype=torch.float32)
        # Insert chronologically: a long episode retains its most recent transitions.
        for row in episode:
            for key, value in row.items():
                if key == "reward":
                    continue
                if key not in self.data:
                    self.data[key] = torch.empty((self.capacity, *value.shape), dtype=value.dtype)
                self.data[key][self.position].copy_(value)
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def observe(self, condition: torch.Tensor, value_condition: torch.Tensor, actions: torch.Tensor,
                rewards: torch.Tensor, dones: torch.Tensor, *, chains=None, pre_tanh=None,
                bootstrap=None) -> None:
        fields = {"condition": condition, "value_condition": value_condition, "actions": actions, "reward": rewards}
        if chains is not None:
            fields["chains"] = chains
        if pre_tanh is not None:
            fields["pre_tanh"] = pre_tanh
        fields = {key: value.detach().cpu() for key, value in fields.items()}
        done = dones.detach().cpu().tolist()
        tails = [0.0] * len(done) if bootstrap is None else bootstrap.detach().cpu().tolist()
        for index, ended in enumerate(done):
            self.pending[index].append({key: value[index].clone() for key, value in fields.items()})
            if ended:
                self._append_episode(self.pending[index], float(tails[index]))
                self.pending[index] = []

    def state_dict(self) -> dict:
        return {
            "version": 1, "capacity": self.capacity, "gamma": self.gamma,
            "size": self.size, "position": self.position,
            "data": {key: value[:self.size].clone() for key, value in self.data.items()},
            "pending": copy.deepcopy(self.pending),
        }

    def load_state_dict(self, state: Mapping) -> None:
        if not isinstance(state, Mapping) or state.get("version") != 1 or state.get("capacity") != self.capacity or state.get("gamma") != self.gamma:
            raise ValueError("incompatible self-imitation checkpoint")
        size, position = state["size"], state["position"]
        if type(size) is not int or type(position) is not int or not 0 <= size <= self.capacity or not 0 <= position < self.capacity or (size < self.capacity and position != size):
            raise ValueError("invalid self-imitation ring state")
        data, pending = state["data"], state["pending"]
        required = {"condition", "value_condition", "actions", "returns"}
        allowed = required | {"chains", "pre_tanh"}
        if not isinstance(data, dict) or (size and not required <= data.keys()) or not data.keys() <= allowed or {"chains", "pre_tanh"} <= data.keys():
            raise ValueError("invalid self-imitation transition fields")
        if not isinstance(pending, list) or len(pending) != len(self.pending):
            raise ValueError("invalid self-imitation pending environments")
        for key, tensor in data.items():
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[0] != size or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"invalid self-imitation tensor: {key}")
        for episode in pending:
            if not isinstance(episode, list):
                raise ValueError("invalid self-imitation pending episode")
            for row in episode:
                if not isinstance(row, dict) or not {"condition", "value_condition", "actions", "reward"} <= row.keys() or not row.keys() <= (allowed - {"returns"}) | {"reward"}:
                    raise ValueError("invalid self-imitation pending fields")
                for key, tensor in row.items():
                    if not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()) or (key in data and (tensor.shape != data[key].shape[1:] or tensor.dtype != data[key].dtype)):
                        raise ValueError("invalid self-imitation pending tensor")
        self.size, self.position = size, position
        self.data = {}
        for key, tensor in data.items():
            self.data[key] = torch.empty((self.capacity, *tensor.shape[1:]), dtype=tensor.dtype)
            self.data[key][:size].copy_(tensor.cpu())
        self.pending = [[{key: tensor.detach().cpu().clone() for key, tensor in row.items()} for row in episode] for episode in pending]


def replay_log_prob(heads: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Joint augmented diffusion path log density, omitting parameter-free terms."""
    condition = batch["condition"]
    if heads.action_kind == "discrete":
        return Categorical(logits=heads.categorical_head(condition)).log_prob(batch["actions"].long())
    if heads.continuous_policy_kind == "diffusion":
        log_prob, _ = heads.diffusion_policy.log_prob_chain(condition, batch["chains"])
        return log_prob.sum(dim=-1)
    return heads.gaussian_policy.log_prob(condition, pre_tanh_action=batch["pre_tanh"])[0]


def update_self_imitation(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                          replay: SelfImitationBuffer, ppo_config: Mapping, *, generator=None) -> dict[str, float]:
    config = sil_config(ppo_config)
    metrics = dict.fromkeys(SIL_METRIC_FIELDS, 0.0)
    metrics["sil_buffer_size"] = float(replay.size)
    if not config["enabled"] or not replay.size or (config["policy_coef"] == config["value_coef"] == 0):
        return metrics
    heads = getattr(model, "heads", model)
    if getattr(heads, "action_value_head", None) is not None:
        raise ValueError("self-imitation currently requires a state-value critic")
    device = next(model.parameters()).device
    transform = ValueTransform.from_config(ppo_config)
    successful = 0
    for _ in range(int(config["updates_per_ppo"])):
        # Refresh ALL priorities after every head update. Stale priorities must
        # never admit transitions whose current R - V is nonpositive.
        values = []
        with torch.no_grad():
            for start in range(0, replay.size, 4096):
                condition = replay.data["value_condition"][start:min(start + 4096, replay.size)].to(device)
                values.append(transform.to_raw(heads.value(condition)).cpu())
        advantage = replay.data["returns"][:replay.size] - torch.cat(values)
        if not bool(torch.isfinite(advantage).all()):
            raise ValueError("non-finite self-imitation advantages")
        positive = int((advantage > float(config["min_advantage"])).sum())
        metrics["sil_positive_transitions"] = float(positive)
        indices = positive_sample(advantage, config, generator=generator)
        if not indices.numel():
            break
        weights = imitation_weights(advantage[indices], config, scale=advantage.std(unbiased=False)).to(device)
        batch = {key: value[indices].to(device) for key, value in replay.data.items()}
        optimizer.zero_grad(set_to_none=True)
        value = heads.value(batch["value_condition"])
        policy_loss = -(replay_log_prob(heads, batch) * weights).mean()
        # Priorities/weights use RAW returns. Regress in the same value target
        # coordinates as PPO (symlog for the production campaigns).
        residual = (transform.to_target(batch["returns"]) - value).clamp_min(0)
        value_loss = 0.5 * residual.square().mean()
        terms = []
        if float(config["policy_coef"]) > 0:
            terms.append(float(config["policy_coef"]) * policy_loss)
        if float(config["value_coef"]) > 0:
            terms.append(float(config["value_coef"]) * value_loss)
        loss = sum(terms)
        if not torch.isfinite(loss):
            raise ValueError("non-finite self-imitation loss")
        loss.backward()
        _, finite = clip_gradients_or_skip(model.parameters(), float(ppo_config["max_grad_norm"]))
        if not finite:
            continue
        optimizer.step()
        successful += 1
        metrics["sil_policy_loss"] += float(policy_loss.detach())
        metrics["sil_value_loss"] += float(value_loss.detach())
        metrics["sil_mean_weight"] += float(weights.mean())
        metrics["sil_selected_transitions"] += indices.numel()
    if successful:
        for key in ("sil_policy_loss", "sil_value_loss", "sil_mean_weight"):
            metrics[key] /= successful
    metrics["sil_updates"] = float(successful)
    return metrics
