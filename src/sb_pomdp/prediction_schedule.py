"""Interleaved next-observation prediction phases around PPO.

The update sequence is

    [pretrain x P] [ppo x I] [predict x Q] [ppo x I] [predict x Q] ... [ppo x rest]

``pretrain`` collects rollouts with uniformly random actions, ``predict`` with the current
policy; both optimise only the next-observation prediction loss of the shared head on the
model's policy condition.  ``ppo`` is the unchanged policy/value update.  The phase is a pure
function of the update index, so segmented resume needs no extra state, and the PPO update
count (``ppo.total_steps``) is unchanged: prediction phases are extra environment steps.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

Phase = Literal["pretrain", "predict", "critic", "ppo"]


@dataclass(frozen=True)
class PredictionSchedule:
    pretrain_updates: int
    interval_updates: int
    phase_updates: int
    ppo_updates: int
    critic_warmup_updates: int = 0

    @classmethod
    def from_config(cls, model_config: Mapping[str, Any], *, rollout_batch_size: int, ppo_updates: int, critic_warmup_updates: int = 0) -> PredictionSchedule:
        pretrain_steps = int(model_config.get("prediction_pretrain_steps", 0))
        return cls(
            pretrain_updates=math.ceil(pretrain_steps / rollout_batch_size) if pretrain_steps > 0 else 0,
            interval_updates=int(model_config.get("prediction_interval_updates", 0)),
            phase_updates=int(model_config.get("prediction_phase_updates", 0)),
            ppo_updates=int(ppo_updates),
            critic_warmup_updates=int(critic_warmup_updates),
        )

    @property
    def enabled(self) -> bool:
        return self.pretrain_updates > 0 or (self.interval_updates > 0 and self.phase_updates > 0)

    @property
    def interleaved_phases(self) -> int:
        """Number of ``predict`` phases between PPO blocks (none after the last PPO block)."""

        if self.interval_updates <= 0 or self.phase_updates <= 0:
            return 0
        return max(0, math.ceil(self.ppo_updates / self.interval_updates) - 1)

    @property
    def total_updates(self) -> int:
        return self.pretrain_updates + self.critic_warmup_updates + self.ppo_updates + self.interleaved_phases * self.phase_updates

    def phase(self, update: int) -> Phase:
        """Phase of the 1-based ``update``."""

        if not 1 <= update <= self.total_updates:
            raise ValueError(f"update {update} is outside 1..{self.total_updates}")
        index = update - 1 - self.pretrain_updates
        if index < 0:
            return "pretrain"
        if index < self.critic_warmup_updates:
            return "critic"
        index -= self.critic_warmup_updates
        if self.interleaved_phases == 0:
            return "ppo"
        block = self.interval_updates + self.phase_updates
        block_index, offset = divmod(index, block)
        if block_index >= self.interleaved_phases:
            return "ppo"
        return "ppo" if offset < self.interval_updates else "predict"

    def ppo_updates_done(self, update: int) -> int:
        """PPO updates completed after the 1-based ``update`` (0 during the pretraining block)."""

        return sum(1 for u in range(1, update + 1) if self.phase(u) == "ppo")


def random_actions(action_spec: Any, batch_size: int, *, device: torch.device) -> torch.Tensor:
    """Uniformly random actions in the trainer's action format (pretraining exploration).

    Discrete: ``long[batch]`` in ``0..n-1``; continuous: ``float[batch, *shape]`` uniform in
    ``[low, high]``.  Uses the global torch RNG, whose state the checkpoints already carry.
    """

    if action_spec.is_discrete:
        return torch.randint(int(action_spec.n), (batch_size,), device=device)
    low = torch.tensor(np.array(action_spec.low, dtype=np.float32), device=device).expand(action_spec.shape)
    high = torch.tensor(np.array(action_spec.high, dtype=np.float32), device=device).expand(action_spec.shape)
    return low + (high - low) * torch.rand(batch_size, *action_spec.shape, device=device)
