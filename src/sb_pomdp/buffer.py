"""Rollout storage and generalised advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class BeliefReplayPrefix:
    """Raw reparameterization inputs since one environment's episode start."""

    observations: torch.Tensor
    previous_actions: torch.Tensor
    initial: torch.Tensor
    ula_initial_noise: torch.Tensor
    ula_step_noises: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.observations.shape[0])


@dataclass(slots=True)
class RolloutBatch:
    observations: torch.Tensor
    previous_particles: torch.Tensor
    previous_actions: torch.Tensor
    initial: torch.Tensor
    particles: torch.Tensor
    ula_initial_noise: torch.Tensor
    ula_step_noises: torch.Tensor
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    diffusion_chains: torch.Tensor | None
    pre_tanh_actions: torch.Tensor | None
    belief_prefixes: tuple[BeliefReplayPrefix, ...]

    @property
    def rollout_shape(self) -> tuple[int, int]:
        return int(self.rewards.shape[0]), int(self.rewards.shape[1])

    @property
    def size(self) -> int:
        time, environments = self.rollout_shape
        return time * environments

    def flatten(self) -> dict[str, torch.Tensor | None]:
        time, environments = self.rollout_shape
        flat = time * environments

        def first_two(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(flat, *value.shape[2:])

        return {
            "observations": first_two(self.observations),
            "previous_particles": first_two(self.previous_particles),
            "previous_actions": first_two(self.previous_actions),
            "initial": first_two(self.initial),
            "particles": first_two(self.particles),
            "ula_initial_noise": first_two(self.ula_initial_noise),
            "ula_step_noises": first_two(self.ula_step_noises),
            "actions": first_two(self.actions),
            "old_log_prob": first_two(self.old_log_prob),
            "old_values": first_two(self.old_values),
            "advantages": first_two(self.advantages),
            "returns": first_two(self.returns),
            "diffusion_chains": (
                None if self.diffusion_chains is None else first_two(self.diffusion_chains)
            ),
            "pre_tanh_actions": (
                None if self.pre_tanh_actions is None else first_two(self.pre_tanh_actions)
            ),
        }


class RolloutBuilder:
    """Append-only rollout builder that keeps tensors on the training device."""

    _REQUIRED = (
        "observations",
        "previous_particles",
        "previous_actions",
        "initial",
        "particles",
        "ula_initial_noise",
        "ula_step_noises",
        "actions",
        "old_log_prob",
        "old_values",
        "rewards",
        "dones",
    )

    def __init__(
        self,
        *,
        belief_prefixes: tuple[BeliefReplayPrefix, ...] = (),
    ) -> None:
        self._data: dict[str, list[torch.Tensor]] = {name: [] for name in self._REQUIRED}
        self._diffusion_chains: list[torch.Tensor] | None = None
        self._pre_tanh_actions: list[torch.Tensor] | None = None
        self._truncation_values: list[torch.Tensor] | None = None
        self._policy_representation: str | None = None
        self._belief_prefixes = belief_prefixes

    def append(
        self,
        *,
        observations: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_actions: torch.Tensor,
        initial: torch.Tensor,
        particles: torch.Tensor,
        ula_initial_noise: torch.Tensor,
        ula_step_noises: torch.Tensor,
        actions: torch.Tensor,
        old_log_prob: torch.Tensor,
        old_values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        diffusion_chains: torch.Tensor | None = None,
        pre_tanh_actions: torch.Tensor | None = None,
        truncation_values: torch.Tensor | None = None,
    ) -> None:
        if truncation_values is not None and truncation_values.shape != rewards.shape:
            raise ValueError("truncation bootstrap values must have the rewards' [batch] shape")
        if ula_initial_noise.shape != particles.shape:
            raise ValueError(
                "ULA initial noise must have the same [batch, particles, state] shape "
                "as particles"
            )
        if (
            ula_step_noises.ndim != particles.ndim + 1
            or ula_step_noises.shape[0] != particles.shape[0]
            or ula_step_noises.shape[2:] != particles.shape[1:]
            or ula_step_noises.shape[1] <= 0
        ):
            raise ValueError(
                "ULA step noises must have shape [batch, steps, particles, state] "
                "consistent with particles"
            )
        for name, noise in (
            ("ULA initial noise", ula_initial_noise),
            ("ULA step noises", ula_step_noises),
        ):
            if noise.dtype != particles.dtype or noise.device != particles.device:
                raise ValueError(f"{name} must share particles' dtype and device")

        values = locals()
        for name in self._REQUIRED:
            self._data[name].append(values[name].detach())
        # Truncation bootstrapping is a whole-rollout property of the configured
        # PPO target, so a rollout must not mix bootstrapped and unbootstrapped
        # steps.  Callers with the flag on pass zeros when nothing truncated.
        if truncation_values is None:
            if self._truncation_values is not None:
                raise ValueError("cannot mix bootstrapped and unbootstrapped rollout entries")
        else:
            if self._truncation_values is None:
                if len(self._data["rewards"]) > 1:
                    raise ValueError("cannot mix bootstrapped and unbootstrapped rollout entries")
                self._truncation_values = []
            self._truncation_values.append(truncation_values.detach())
        if diffusion_chains is not None and pre_tanh_actions is not None:
            raise ValueError("one rollout entry cannot contain two policy representations")
        representation = (
            "diffusion"
            if diffusion_chains is not None
            else "gaussian"
            if pre_tanh_actions is not None
            else "categorical"
        )
        if self._policy_representation is None:
            self._policy_representation = representation
        elif self._policy_representation != representation:
            raise ValueError("cannot mix policy representations in one rollout")
        if diffusion_chains is not None:
            assert diffusion_chains is not None
            if self._diffusion_chains is None:
                self._diffusion_chains = []
            self._diffusion_chains.append(diffusion_chains.detach())
        if pre_tanh_actions is not None:
            if self._pre_tanh_actions is None:
                self._pre_tanh_actions = []
            self._pre_tanh_actions.append(pre_tanh_actions.detach())

    def finish(
        self,
        *,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        if not self._data["rewards"]:
            raise ValueError("cannot finish an empty rollout")
        stacked = {name: torch.stack(items, dim=0) for name, items in self._data.items()}
        environment_count = int(stacked["rewards"].shape[1])
        if self._belief_prefixes and len(self._belief_prefixes) != environment_count:
            raise ValueError("one belief replay prefix is required per rollout environment")
        for prefix in self._belief_prefixes:
            length = prefix.length
            expected_shapes = {
                "observations": (length, *stacked["observations"].shape[2:]),
                "previous_actions": (length, *stacked["previous_actions"].shape[2:]),
                "initial": (length,),
                "ula_initial_noise": (length, *stacked["ula_initial_noise"].shape[2:]),
                "ula_step_noises": (length, *stacked["ula_step_noises"].shape[2:]),
            }
            for name, expected_shape in expected_shapes.items():
                value = getattr(prefix, name)
                if tuple(value.shape) != expected_shape:
                    raise ValueError(
                        f"belief replay prefix {name} must have shape {expected_shape}"
                    )
                reference = stacked[name]
                if value.dtype != reference.dtype or value.device != reference.device:
                    raise ValueError(
                        f"belief replay prefix {name} must share rollout dtype and device"
                    )
        advantages, returns = generalised_advantage_estimate(
            stacked["rewards"],
            stacked["dones"],
            stacked["old_values"],
            last_value.detach(),
            gamma=gamma,
            gae_lambda=gae_lambda,
            truncation_values=(
                None
                if self._truncation_values is None
                else torch.stack(self._truncation_values, dim=0)
            ),
        )
        chains = (
            None if self._diffusion_chains is None else torch.stack(self._diffusion_chains, dim=0)
        )
        pre_tanh_actions = (
            None if self._pre_tanh_actions is None else torch.stack(self._pre_tanh_actions, dim=0)
        )
        return RolloutBatch(
            **stacked,
            advantages=advantages,
            returns=returns,
            diffusion_chains=chains,
            pre_tanh_actions=pre_tanh_actions,
            belief_prefixes=self._belief_prefixes,
        )


def generalised_advantage_estimate(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    truncation_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute episodic GAE; both termination and time limit close an episode.

    ``truncation_values`` is the optional per-step bootstrap ``V(s_T)`` of the
    post-step state that the auto-reset wrapper discards.  It must be ``[T, B]``
    and zero everywhere except at a time-limit truncation, where it turns that
    step's target into ``r_t + gamma * V(s_T)`` instead of ``r_t``.  ``dones``
    keeps closing the episode either way: ``nonterminal`` still cuts both the
    successor value of the *reset* observation and the lambda-return recursion,
    so no credit leaks across the boundary.  Omitting the argument (the default)
    reproduces the original "a time limit is a true terminal state" convention
    exactly, which is what ``ppo.bootstrap_on_truncation=false`` selects.
    """

    if rewards.shape != dones.shape or rewards.shape != values.shape:
        raise ValueError("reward, done, and value tensors must have identical [T, B] shapes")
    if last_value.shape != rewards.shape[1:]:
        raise ValueError("last_value must have shape [B]")
    if truncation_values is not None and truncation_values.shape != rewards.shape:
        raise ValueError("truncation_values must have the same [T, B] shape as rewards")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for time_index in reversed(range(rewards.shape[0])):
        next_value = last_value if time_index == rewards.shape[0] - 1 else values[time_index + 1]
        nonterminal = 1.0 - dones[time_index].float()
        delta = rewards[time_index] + gamma * next_value * nonterminal - values[time_index]
        if truncation_values is not None:
            delta = delta + gamma * truncation_values[time_index]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[time_index] = running
    return advantages, advantages + values
