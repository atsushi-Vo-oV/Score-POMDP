"""Critic target transforms that leave rewards and advantages untouched.

The policy objective in this project is already scale-free (advantages are
normalised), so a badly scaled task only hurts through the value regression:
targets of O(1e3-1e4) with terminal spikes make the value loss dominate the
shared trunk and the global gradient clip.  A transform of the *regression
target* fixes that without changing the task: rewards, GAE, and advantages are
computed in raw units, the critic merely predicts ``f(return)`` and its
prediction is mapped back with ``f^{-1}`` wherever a raw value is needed.

``symlog`` (as in DreamerV3) compresses heavy tails logarithmically while
staying linear near zero, needs no running statistics, and therefore keeps
checkpoints and exact resume unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch

ValueTargetTransform = Literal["none", "symlog"]
VALUE_TARGET_TRANSFORMS: tuple[str, ...] = ("none", "symlog")


def symlog(value: torch.Tensor) -> torch.Tensor:
    """``sign(x) * log(1 + |x|)``: linear near zero, logarithmic in the tails."""

    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    """Exact inverse of :func:`symlog`."""

    return torch.sign(value) * torch.expm1(torch.abs(value))


class ValueTransform:
    """Map between the critic's network output and raw value units."""

    def __init__(self, kind: str = "none") -> None:
        if kind not in VALUE_TARGET_TRANSFORMS:
            raise ValueError(
                f"unsupported value target transform {kind!r}; "
                f"expected one of {VALUE_TARGET_TRANSFORMS}"
            )
        self.kind: str = kind

    @classmethod
    def from_config(cls, ppo_config: Mapping[str, Any]) -> ValueTransform:
        return cls(str(ppo_config.get("value_target_transform", "none")))

    @property
    def is_identity(self) -> bool:
        return self.kind == "none"

    def to_raw(self, network_output: torch.Tensor) -> torch.Tensor:
        """Raw value units from what the critic network emits."""

        if self.is_identity:
            return network_output
        return symexp(network_output)

    def to_target(self, returns: torch.Tensor) -> torch.Tensor:
        """Regression target in the critic's output space for raw returns."""

        if self.is_identity:
            return returns
        return symlog(returns)


__all__ = [
    "VALUE_TARGET_TRANSFORMS",
    "ValueTargetTransform",
    "ValueTransform",
    "symexp",
    "symlog",
]
