"""Actor--critic baselines for the comparison experiments.

The feed-forward models consume either the masked observation or the simulator
state.  The recurrent model consumes masked observations and exposes an
explicit time-major API so PPO can replay complete rollout sequences with
backpropagation through time (BPTT).

The active GRU comparison uses the same categorical or diffusion policy head
as the score-belief model.  Feed-forward auxiliary baselines retain the
tanh-Gaussian continuous head used by their original ablation definitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal

from .networks import make_mlp
from .policies import DiffusionPolicy, PolicySample, TanhGaussianPolicy

BaselineKind = Literal["observation", "oracle_state", "gru"]
ActionKind = Literal["discrete", "continuous"]
TemporalGradientMode = Literal["full", "tbptt_1"]


@dataclass(slots=True)
class ActorCriticEvaluation:
    """Policy likelihood, entropy bonus, and value for a saved action."""

    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class RecurrentActorCriticEvaluation(ActorCriticEvaluation):
    """Sequence evaluation plus the hidden state after the final time step."""

    final_hidden: torch.Tensor


def _validate_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    checked = tuple(hidden_dims)
    if not checked or any(
        isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in checked
    ):
        raise ValueError("hidden_dims must contain positive integers")
    return checked


def _feature_encoder(input_dim: int, hidden_dims: Sequence[int]) -> tuple[nn.Module, int]:
    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("input_dim must be a positive integer")
    widths = _validate_hidden_dims(hidden_dims)
    # ``make_mlp`` deliberately leaves its output layer linear.  Here that
    # output is a hidden representation, so keep a non-linearity even when the
    # configured baseline has only one hidden width (as in the local smoke).
    return nn.Sequential(make_mlp(input_dim, widths[:-1], widths[-1]), nn.SiLU()), widths[-1]


class SquashedDiagonalGaussian(nn.Module):
    """A bounded diagonal-Gaussian policy with replayable log likelihoods."""

    def __init__(
        self,
        feature_dim: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
        *,
        initial_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        inverse_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        low = torch.as_tensor(action_low, dtype=torch.float32).flatten()
        high = torch.as_tensor(action_high, dtype=torch.float32).flatten()
        if (
            feature_dim <= 0
            or low.numel() == 0
            or low.shape != high.shape
            or not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or not torch.all(high > low)
        ):
            raise ValueError("continuous policy requires finite, matching, ordered bounds")
        if not min_log_std <= initial_log_std <= max_log_std:
            raise ValueError("initial_log_std must lie between min_log_std and max_log_std")
        if not 0.0 < inverse_epsilon < 0.5:
            raise ValueError("inverse_epsilon must lie in (0, 0.5)")

        self.action_dim = int(low.numel())
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.inverse_epsilon = float(inverse_epsilon)
        self.mean_head = nn.Linear(feature_dim, self.action_dim)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(initial_log_std)))
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def distribution(self, features: torch.Tensor) -> Independent:
        if features.ndim < 2:
            raise ValueError("features must have a batch dimension and a feature dimension")
        mean = self.mean_head(features)
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        std = log_std.exp().expand_as(mean)
        return Independent(Normal(mean, std), 1)

    def _bounded_from_raw(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action) * self.action_scale + self.action_bias

    def _raw_from_bounded(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action.shape[-1:] != (self.action_dim,):
            raise ValueError(
                f"expected actions ending in dimension {self.action_dim}, got {tuple(action.shape)}"
            )
        normalized = (action - self.action_bias) / self.action_scale
        normalized = normalized.clamp(
            -1.0 + self.inverse_epsilon,
            1.0 - self.inverse_epsilon,
        )
        return torch.atanh(normalized), normalized

    def log_prob(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        distribution = self.distribution(features)
        raw_action, normalized = self._raw_from_bounded(action)
        log_jacobian = (
            self.action_scale.log()
            + torch.log1p(-normalized.square()).clamp_min(
                torch.log(
                    torch.as_tensor(
                        self.inverse_epsilon,
                        dtype=normalized.dtype,
                        device=normalized.device,
                    )
                )
            )
        ).sum(dim=-1)
        return distribution.log_prob(raw_action) - log_jacobian

    def entropy(self, features: torch.Tensor) -> torch.Tensor:
        """Return the base-Gaussian entropy used as the PPO entropy bonus.

        A tanh-transformed Gaussian has no closed-form entropy.  This standard
        deterministic proxy avoids adding Monte Carlo noise to PPO updates.
        """

        return self.distribution(features).entropy()

    def sample(
        self,
        features: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> PolicySample:
        distribution = self.distribution(features)
        if deterministic:
            raw_action = distribution.base_dist.loc
        else:
            noise = torch.randn(
                distribution.base_dist.loc.shape,
                dtype=features.dtype,
                device=features.device,
                generator=generator,
            )
            raw_action = distribution.base_dist.loc + distribution.base_dist.scale * noise
        action = self._bounded_from_raw(raw_action)
        return PolicySample(
            action=action,
            log_prob=self.log_prob(features, action),
            entropy=self.entropy(features),
        )


class _PolicyValueHeads(nn.Module):
    """Categorical, Gaussian, or diffusion actor with a scalar critic."""

    def __init__(
        self,
        feature_dim: int,
        *,
        action_kind: ActionKind,
        discrete_actions: int | None,
        action_low: Sequence[float] | None,
        action_high: Sequence[float] | None,
        hidden_dims: Sequence[int] = (),
        continuous_policy_kind: Literal["gaussian", "diffusion"] = "gaussian",
        diffusion_steps: int = 10,
        diffusion_beta_start: float = 0.01,
        diffusion_beta_end: float = 0.2,
        diffusion_min_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.action_kind = action_kind
        self.continuous_policy_kind = continuous_policy_kind
        if continuous_policy_kind not in {"gaussian", "diffusion"}:
            raise ValueError("continuous_policy_kind must be gaussian or diffusion")
        self.value_head = make_mlp(feature_dim, hidden_dims, 1)

        if action_kind == "discrete":
            if (
                isinstance(discrete_actions, bool)
                or not isinstance(discrete_actions, int)
                or discrete_actions < 2
            ):
                raise ValueError("a discrete policy requires at least two actions")
            self.categorical_head: nn.Module | None = make_mlp(
                feature_dim,
                hidden_dims,
                discrete_actions,
            )
            final_categorical_layer = next(
                layer
                for layer in reversed(list(self.categorical_head.modules()))
                if isinstance(layer, nn.Linear)
            )
            nn.init.orthogonal_(final_categorical_layer.weight, gain=0.01)
            nn.init.zeros_(final_categorical_layer.bias)
            self.gaussian_policy: TanhGaussianPolicy | None = None
            self.diffusion_policy: DiffusionPolicy | None = None
        elif action_kind == "continuous":
            if action_low is None or action_high is None:
                raise ValueError("a continuous policy requires action bounds")
            self.categorical_head = None
            if continuous_policy_kind == "diffusion":
                self.diffusion_policy = DiffusionPolicy(
                    feature_dim,
                    action_low,
                    action_high,
                    hidden_dims,
                    num_steps=diffusion_steps,
                    beta_start=diffusion_beta_start,
                    beta_end=diffusion_beta_end,
                    min_std=diffusion_min_std,
                )
                self.gaussian_policy = None
            else:
                self.diffusion_policy = None
                self.gaussian_policy = TanhGaussianPolicy(
                    feature_dim,
                    action_low,
                    action_high,
                    hidden_dims,
                )
        else:
            raise ValueError(f"unsupported action kind: {action_kind}")

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_head(features).squeeze(-1)

    def sample_policy(
        self,
        features: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> PolicySample:
        if self.action_kind == "continuous":
            if self.continuous_policy_kind == "diffusion":
                assert self.diffusion_policy is not None
                leading_shape = features.shape[:-1]
                flat_features = features.reshape(-1, features.shape[-1])
                action, chain, step_log_prob = self.diffusion_policy.sample(
                    flat_features,
                    deterministic=deterministic,
                    generator=generator,
                )
                _, step_entropy = self.diffusion_policy.log_prob_chain(flat_features, chain)
                return PolicySample(
                    action=action.reshape(*leading_shape, action.shape[-1]),
                    log_prob=step_log_prob.reshape(*leading_shape, step_log_prob.shape[-1]),
                    entropy=step_entropy.reshape(*leading_shape, step_entropy.shape[-1]),
                    diffusion_chain=chain.reshape(
                        *leading_shape,
                        chain.shape[-2],
                        chain.shape[-1],
                    ),
                )
            assert self.gaussian_policy is not None
            action, pre_tanh_action, log_prob = self.gaussian_policy.sample(
                features,
                deterministic=deterministic,
                generator=generator,
            )
            _, entropy = self.gaussian_policy.log_prob(
                features,
                pre_tanh_action=pre_tanh_action,
            )
            return PolicySample(
                action=action,
                log_prob=log_prob,
                entropy=entropy,
                pre_tanh_action=pre_tanh_action,
            )

        assert self.categorical_head is not None
        distribution = Categorical(logits=self.categorical_head(features))
        if deterministic:
            action = distribution.probs.argmax(dim=-1)
        elif generator is None:
            action = distribution.sample()
        else:
            action = torch.multinomial(
                distribution.probs.reshape(-1, distribution.probs.shape[-1]),
                1,
                generator=generator,
            ).reshape(distribution.probs.shape[:-1])
        return PolicySample(
            action=action,
            log_prob=distribution.log_prob(action),
            entropy=distribution.entropy(),
        )

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_tanh_actions: torch.Tensor | None = None,
        diffusion_chains: torch.Tensor | None = None,
    ) -> ActorCriticEvaluation:
        if self.action_kind == "continuous":
            if self.continuous_policy_kind == "diffusion":
                if diffusion_chains is None:
                    raise ValueError("a diffusion policy requires its saved reverse chain")
                if pre_tanh_actions is not None:
                    raise ValueError("a diffusion policy does not accept pre-tanh actions")
                assert self.diffusion_policy is not None
                leading_shape = features.shape[:-1]
                flat_features = features.reshape(-1, features.shape[-1])
                flat_chains = diffusion_chains.reshape(
                    -1,
                    diffusion_chains.shape[-2],
                    diffusion_chains.shape[-1],
                )
                log_prob, entropy = self.diffusion_policy.log_prob_chain(
                    flat_features,
                    flat_chains,
                )
                log_prob = log_prob.reshape(*leading_shape, log_prob.shape[-1])
                entropy = entropy.reshape(*leading_shape, entropy.shape[-1])
            else:
                if diffusion_chains is not None:
                    raise ValueError("a Gaussian policy does not accept a diffusion chain")
                assert self.gaussian_policy is not None
                if pre_tanh_actions is None:
                    log_prob, entropy = self.gaussian_policy.log_prob(features, action=actions)
                else:
                    log_prob, entropy = self.gaussian_policy.log_prob(
                        features,
                        pre_tanh_action=pre_tanh_actions,
                    )
        else:
            assert self.categorical_head is not None
            distribution = Categorical(logits=self.categorical_head(features))
            log_prob = distribution.log_prob(actions.long())
            entropy = distribution.entropy()
        return ActorCriticEvaluation(
            log_prob=log_prob,
            entropy=entropy,
            value=self.value(features),
        )


class FeedForwardBaselineActorCritic(nn.Module):
    """Feed-forward PPO baseline over masked observations or oracle states."""

    def __init__(
        self,
        input_dim: int,
        *,
        input_source: Literal["observation", "oracle_state"],
        action_kind: ActionKind,
        hidden_dims: Sequence[int],
        discrete_actions: int | None = None,
        action_low: Sequence[float] | None = None,
        action_high: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if input_source not in ("observation", "oracle_state"):
            raise ValueError("input_source must be observation or oracle_state")
        self.input_source = input_source
        self.input_dim = input_dim
        self.encoder, feature_dim = _feature_encoder(input_dim, hidden_dims)
        self.heads = _PolicyValueHeads(
            feature_dim,
            action_kind=action_kind,
            discrete_actions=discrete_actions,
            action_low=action_low,
            action_high=action_high,
        )

    @property
    def action_kind(self) -> ActionKind:
        return self.heads.action_kind

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected inputs ending in dimension {self.input_dim}, got {tuple(inputs.shape)}"
            )
        return self.encoder(inputs)

    def value(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.heads.value(self.encode(inputs))

    def sample_policy(
        self,
        inputs: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> PolicySample:
        return self.heads.sample_policy(
            self.encode(inputs),
            deterministic=deterministic,
            generator=generator,
        )

    def act(
        self,
        inputs: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[PolicySample, torch.Tensor]:
        features = self.encode(inputs)
        return (
            self.heads.sample_policy(
                features,
                deterministic=deterministic,
                generator=generator,
            ),
            self.heads.value(features),
        )

    def evaluate_actions(
        self,
        inputs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_tanh_actions: torch.Tensor | None = None,
    ) -> ActorCriticEvaluation:
        return self.heads.evaluate_actions(
            self.encode(inputs),
            actions,
            pre_tanh_actions=pre_tanh_actions,
        )


class ObservationActorCritic(FeedForwardBaselineActorCritic):
    """Feed-forward baseline receiving only the masked observation."""

    def __init__(self, observation_dim: int, **kwargs: object) -> None:
        super().__init__(observation_dim, input_source="observation", **kwargs)


class OracleStateActorCritic(FeedForwardBaselineActorCritic):
    """Feed-forward upper-bound baseline receiving the true simulator state."""

    def __init__(self, state_dim: int, **kwargs: object) -> None:
        super().__init__(state_dim, input_source="oracle_state", **kwargs)


class GRUActorCritic(nn.Module):
    """Observation-only recurrent PPO baseline with a time-major BPTT API."""

    input_source: Literal["observation"] = "observation"

    def __init__(
        self,
        observation_dim: int,
        *,
        action_kind: ActionKind,
        hidden_dims: Sequence[int],
        recurrent_hidden_dim: int,
        recurrent_layers: int = 1,
        policy_condition_dim: int | None = None,
        head_hidden_dims: Sequence[int] = (),
        continuous_policy_kind: Literal["gaussian", "diffusion"] = "gaussian",
        diffusion_steps: int = 10,
        diffusion_beta_start: float = 0.01,
        diffusion_beta_end: float = 0.2,
        diffusion_min_std: float = 0.05,
        discrete_actions: int | None = None,
        action_low: Sequence[float] | None = None,
        action_high: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if recurrent_hidden_dim <= 0 or recurrent_layers <= 0:
            raise ValueError("recurrent dimensions must be positive")
        self.observation_dim = observation_dim
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.recurrent_layers = recurrent_layers
        self.observation_encoder, encoded_dim = _feature_encoder(observation_dim, hidden_dims)
        self.gru = nn.GRU(encoded_dim, recurrent_hidden_dim, num_layers=recurrent_layers)
        condition_dim = recurrent_hidden_dim if policy_condition_dim is None else policy_condition_dim
        if condition_dim <= 0:
            raise ValueError("policy_condition_dim must be positive")
        self.policy_condition_dim = condition_dim
        self.condition_projection: nn.Module = (
            nn.Identity()
            if recurrent_hidden_dim == condition_dim
            else nn.Linear(recurrent_hidden_dim, condition_dim)
        )
        self.heads = _PolicyValueHeads(
            condition_dim,
            action_kind=action_kind,
            discrete_actions=discrete_actions,
            action_low=action_low,
            action_high=action_high,
            hidden_dims=head_hidden_dims,
            continuous_policy_kind=continuous_policy_kind,
            diffusion_steps=diffusion_steps,
            diffusion_beta_start=diffusion_beta_start,
            diffusion_beta_end=diffusion_beta_end,
            diffusion_min_std=diffusion_min_std,
        )

    @property
    def action_kind(self) -> ActionKind:
        return self.heads.action_kind

    @property
    def continuous_policy_kind(self) -> Literal["gaussian", "diffusion"]:
        return self.heads.continuous_policy_kind

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        return torch.zeros(
            self.recurrent_layers,
            batch_size,
            self.recurrent_hidden_dim,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def _validate_sequence_inputs(
        self,
        observations: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> None:
        if observations.ndim != 3 or observations.shape[-1] != self.observation_dim:
            raise ValueError(
                "observations must have shape "
                f"[time, batch, {self.observation_dim}], got {tuple(observations.shape)}"
            )
        time, batch, _ = observations.shape
        expected_hidden = (self.recurrent_layers, batch, self.recurrent_hidden_dim)
        if tuple(initial_hidden.shape) != expected_hidden:
            raise ValueError(
                f"initial_hidden must have shape {expected_hidden}, got {tuple(initial_hidden.shape)}"
            )
        if tuple(episode_starts.shape) != (time, batch):
            raise ValueError(
                f"episode_starts must have shape {(time, batch)}, got {tuple(episode_starts.shape)}"
            )

    def encode_sequence(
        self,
        observations: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        *,
        temporal_gradient_mode: TemporalGradientMode = "full",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``[T, B, O]`` observations while resetting hidden state.

        ``episode_starts[t, b]`` must be true when observation ``t`` begins a
        new episode.  The loop is intentional: arbitrary per-environment reset
        points cannot be represented by one vanilla ``nn.GRU`` invocation.
        ``full`` keeps the hidden-state graph connected from the start of each
        episode.  ``tbptt_1`` preserves the identical forward recurrence but
        detaches the incoming hidden state at every environment-time boundary,
        so each loss differentiates through exactly one GRU transition.
        """

        self._validate_sequence_inputs(observations, initial_hidden, episode_starts)
        if temporal_gradient_mode not in {"full", "tbptt_1"}:
            raise ValueError("temporal_gradient_mode must be 'full' or 'tbptt_1'")
        encoded = self.observation_encoder(observations)
        hidden = initial_hidden
        outputs: list[torch.Tensor] = []
        for time_index in range(observations.shape[0]):
            if temporal_gradient_mode == "tbptt_1":
                hidden = hidden.detach()
            continuation = (~episode_starts[time_index].bool()).to(hidden.dtype)
            hidden = hidden * continuation.view(1, -1, 1)
            output, hidden = self.gru(encoded[time_index : time_index + 1], hidden)
            outputs.append(self.condition_projection(output[0]))
        return torch.stack(outputs, dim=0), hidden

    def step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one vector-environment step using the same path as BPTT."""

        if observation.ndim != 2:
            raise ValueError("a recurrent step observation must have shape [batch, observation]")
        features, next_hidden = self.encode_sequence(
            observation.unsqueeze(0),
            hidden,
            episode_start.unsqueeze(0),
        )
        return features[0], next_hidden

    def act_step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        episode_start: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[PolicySample, torch.Tensor, torch.Tensor]:
        features, next_hidden = self.step(observation, hidden, episode_start)
        sample = self.heads.sample_policy(
            features,
            deterministic=deterministic,
            generator=generator,
        )
        return sample, self.heads.value(features), next_hidden

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        *,
        pre_tanh_actions: torch.Tensor | None = None,
        diffusion_chains: torch.Tensor | None = None,
        temporal_gradient_mode: TemporalGradientMode = "full",
    ) -> RecurrentActorCriticEvaluation:
        """Replay a rollout without flattening time, preserving BPTT."""

        features, final_hidden = self.encode_sequence(
            observations,
            initial_hidden,
            episode_starts,
            temporal_gradient_mode=temporal_gradient_mode,
        )
        evaluation = self.heads.evaluate_actions(
            features,
            actions,
            pre_tanh_actions=pre_tanh_actions,
            diffusion_chains=diffusion_chains,
        )
        return RecurrentActorCriticEvaluation(
            log_prob=evaluation.log_prob,
            entropy=evaluation.entropy,
            value=evaluation.value,
            final_hidden=final_hidden,
        )


def make_baseline_actor_critic(
    kind: BaselineKind,
    *,
    observation_dim: int,
    state_dim: int,
    action_kind: ActionKind,
    hidden_dims: Sequence[int],
    discrete_actions: int | None = None,
    action_low: Sequence[float] | None = None,
    action_high: Sequence[float] | None = None,
    recurrent_hidden_dim: int = 128,
    recurrent_layers: int = 1,
    policy_condition_dim: int | None = None,
    head_hidden_dims: Sequence[int] = (),
    continuous_policy_kind: Literal["gaussian", "diffusion"] = "gaussian",
    diffusion_steps: int = 10,
    diffusion_beta_start: float = 0.01,
    diffusion_beta_end: float = 0.2,
    diffusion_min_std: float = 0.05,
) -> FeedForwardBaselineActorCritic | GRUActorCritic:
    """Build one comparison model from environment dimensions and bounds."""

    common = {
        "action_kind": action_kind,
        "hidden_dims": hidden_dims,
        "discrete_actions": discrete_actions,
        "action_low": action_low,
        "action_high": action_high,
    }
    if kind == "observation":
        return ObservationActorCritic(observation_dim, **common)
    if kind == "oracle_state":
        return OracleStateActorCritic(state_dim, **common)
    if kind == "gru":
        return GRUActorCritic(
            observation_dim,
            **common,
            recurrent_hidden_dim=recurrent_hidden_dim,
            recurrent_layers=recurrent_layers,
            policy_condition_dim=policy_condition_dim,
            head_hidden_dims=head_hidden_dims,
            continuous_policy_kind=continuous_policy_kind,
            diffusion_steps=diffusion_steps,
            diffusion_beta_start=diffusion_beta_start,
            diffusion_beta_end=diffusion_beta_end,
            diffusion_min_std=diffusion_min_std,
        )
    raise ValueError(f"unsupported baseline kind: {kind}")


__all__ = [
    "ActorCriticEvaluation",
    "BaselineKind",
    "FeedForwardBaselineActorCritic",
    "GRUActorCritic",
    "ObservationActorCritic",
    "OracleStateActorCritic",
    "RecurrentActorCriticEvaluation",
    "SquashedDiagonalGaussian",
    "TemporalGradientMode",
    "make_baseline_actor_critic",
]
