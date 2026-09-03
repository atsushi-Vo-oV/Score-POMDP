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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal
from torch.nn import functional

from .networks import init_final_linear, make_head_network, make_trunk_network
from .policies import DiffusionPolicy, PolicySample, TanhGaussianPolicy

BaselineKind = Literal["observation", "oracle_state", "gru", "rnn", "particle_filter"]
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
    return nn.Sequential(make_trunk_network(input_dim, widths[:-1], widths[-1]), nn.SiLU()), widths[-1]


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
        self.value_head = make_head_network(feature_dim, hidden_dims, 1)

        if action_kind == "discrete":
            if (
                isinstance(discrete_actions, bool)
                or not isinstance(discrete_actions, int)
                or discrete_actions < 2
            ):
                raise ValueError("a discrete policy requires at least two actions")
            self.categorical_head: nn.Module | None = make_head_network(
                feature_dim,
                hidden_dims,
                discrete_actions,
            )
            init_final_linear(self.categorical_head, 0.01)
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
    """Observation-only recurrent PPO baseline with a time-major BPTT API.

    ``recurrent_cell`` selects the recurrence: the default gated GRU, or the
    gate-free Elman RNN (``tanh``), which isolates what gating itself buys.
    """

    input_source: Literal["observation"] = "observation"

    def __init__(
        self,
        observation_dim: int,
        *,
        action_kind: ActionKind,
        hidden_dims: Sequence[int],
        recurrent_hidden_dim: int,
        recurrent_layers: int = 1,
        recurrent_cell: Literal["gru", "rnn"] = "gru",
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
        if recurrent_cell not in ("gru", "rnn"):
            raise ValueError("recurrent_cell must be 'gru' or 'rnn'")
        self.observation_dim = observation_dim
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.recurrent_layers = recurrent_layers
        self.recurrent_cell: Literal["gru", "rnn"] = recurrent_cell
        self.observation_encoder, encoded_dim = _feature_encoder(observation_dim, hidden_dims)
        # The attribute keeps its historical name so existing GRU checkpoints
        # load unchanged; each cell stores its own native parameter layout.
        self.gru: nn.GRU | nn.RNN = (
            nn.GRU(encoded_dim, recurrent_hidden_dim, num_layers=recurrent_layers)
            if recurrent_cell == "gru"
            else nn.RNN(
                encoded_dim,
                recurrent_hidden_dim,
                num_layers=recurrent_layers,
                nonlinearity="tanh",
            )
        )
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


class ParticleFilterActorCritic(nn.Module):
    """Deterministic particle-filter-style recurrent PPO baseline.

    The belief is a set of ``K`` latent hypothesis particles with normalised
    log-weights, updated discriminatively from observations and previous
    actions only (no generative model, no reconstruction loss), following the
    PF-RNN/DPFRL lineage.  Two stochastic components of a classical particle
    filter are replaced by deterministic relaxations so that the recurrence is
    an exact function of its stored inputs (which is what the shared
    prefix-replay, reconditioning, and resume machinery requires):

    * transition sampling: ``K`` learned anchor embeddings seed ``K`` distinct
      deterministic hypothesis trajectories through an observation-conditioned
      transition network, instead of drawing transition noise per particle;
    * resampling: the previous weights are mixed with a uniform floor,
      ``alpha * w + (1 - alpha) / K`` (the expectation of PF-RNN's soft
      resampling), which deterministically prevents weight collapse.

    The observation-compatibility network plays the classical likelihood role:
    its output is added to the running log-weights, which are re-normalised
    every step.  The policy/value condition is the weight-averaged particle
    feature, projected to the shared head width.  The public API mirrors
    :class:`GRUActorCritic` exactly; the packed hidden state is
    ``[1, batch, K * (particle_dim + 1)]`` holding particles and log-weights.
    """

    input_source: Literal["observation"] = "observation"

    def __init__(
        self,
        observation_dim: int,
        *,
        action_kind: ActionKind,
        num_particles: int,
        particle_dim: int,
        hidden_dims: Sequence[int],
        soft_alpha: float = 0.9,
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
        variant: Literal["deterministic", "dpfrl"] = "deterministic",
        mgf_features: int = 0,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if (
            isinstance(num_particles, bool)
            or not isinstance(num_particles, int)
            or num_particles < 2
        ):
            raise ValueError("num_particles must be an integer of at least two")
        if isinstance(particle_dim, bool) or not isinstance(particle_dim, int) or particle_dim <= 0:
            raise ValueError("particle_dim must be a positive integer")
        if not 0.0 < float(soft_alpha) <= 1.0:
            raise ValueError("soft_alpha must lie in (0, 1]")
        if variant not in ("deterministic", "dpfrl"):
            raise ValueError("variant must be 'deterministic' or 'dpfrl'")
        if isinstance(mgf_features, bool) or not isinstance(mgf_features, int) or mgf_features < 0:
            raise ValueError("mgf_features must be a non-negative integer")
        widths = _validate_hidden_dims(hidden_dims)
        self.variant: Literal["deterministic", "dpfrl"] = variant
        self.feature_input_dim = observation_dim
        self.num_particles = num_particles
        self.particle_dim = particle_dim
        self.soft_alpha = float(soft_alpha)
        self.mgf_features = int(mgf_features)
        self.recurrent_layers = 1
        # ``dpfrl`` inputs carry the exogenous per-step randomness after the
        # raw features: K*D standard normals (transition noise) and K uniforms
        # (soft-resampling draws).  Storing the noise in the input stream keeps
        # the recurrence an exact function of stored inputs, so prefix replay,
        # reconditioning, and resume work unchanged.
        self.noise_dim = num_particles * (particle_dim + 1) if variant == "dpfrl" else 0
        self.observation_dim = observation_dim + self.noise_dim

        condition_dim = widths[-1] if policy_condition_dim is None else policy_condition_dim
        if condition_dim <= 0:
            raise ValueError("policy_condition_dim must be positive")
        self.policy_condition_dim = condition_dim
        if variant == "deterministic":
            self.anchors = nn.Parameter(torch.randn(num_particles, particle_dim) * 0.1)
            self.initial_net = make_trunk_network(observation_dim + particle_dim, widths, particle_dim)
            self.transition_net = make_trunk_network(particle_dim + observation_dim, widths, particle_dim)
            self.weight_net = make_trunk_network(particle_dim + observation_dim, widths, 1)
            self.feature_net = nn.Sequential(
                make_trunk_network(particle_dim, widths[:-1], widths[-1]), nn.SiLU()
            )
            self.condition_projection: nn.Module = (
                nn.Identity()
                if widths[-1] == condition_dim
                else nn.Linear(widths[-1], condition_dim)
            )
        else:
            encoded_dim = widths[-1]
            # Observation/action encoder shared by every particle (DPFRL encodes
            # the observation before the PF-GRU cell consumes it).
            self.input_net = nn.Sequential(
                make_trunk_network(observation_dim, widths[:-1], encoded_dim), nn.SiLU()
            )
            # PF-GRU cell (Ma et al. 2020): gates r, z; candidate n gets Gaussian
            # noise with a learned, input-dependent scale (reparameterised).
            self.gate = nn.Linear(particle_dim + encoded_dim, 2 * particle_dim)
            self.candidate = nn.Linear(particle_dim + encoded_dim, particle_dim)
            self.noise_scale = nn.Linear(particle_dim + encoded_dim, particle_dim)
            self.initial_mean = nn.Linear(encoded_dim, particle_dim)
            self.initial_noise_scale = nn.Linear(encoded_dim, particle_dim)
            # Discriminative observation function: one linear layer, no
            # activation, producing the log-compatibility added to log-weights.
            self.observation_function = nn.Linear(particle_dim + encoded_dim, 1)
            # Moment-generating-function features M_j = sum_i w_i exp(v_j . h_i).
            self.mgf_vectors = nn.Parameter(
                torch.randn(max(self.mgf_features, 1), particle_dim) / particle_dim**0.5
            )
            self.condition_projection = nn.Linear(particle_dim + self.mgf_features, condition_dim)
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

    @property
    def recurrent_hidden_dim(self) -> int:
        return self.num_particles * (self.particle_dim + 1)

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return the pre-episode packed state.

        Its value is irrelevant to any output: every episode's first step
        carries ``episode_starts`` and rebuilds particles from the first input,
        so zeros act purely as a shape-compatible placeholder, exactly like the
        GRU's zero hidden state.
        """

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

    def _unpack(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = hidden.shape[1]
        packed = hidden[0].reshape(batch, self.num_particles, self.particle_dim + 1)
        return packed[..., : self.particle_dim], packed[..., self.particle_dim]

    def _pack(self, particles: torch.Tensor, log_weights: torch.Tensor) -> torch.Tensor:
        packed = torch.cat((particles, log_weights.unsqueeze(-1)), dim=-1)
        return packed.reshape(1, particles.shape[0], self.recurrent_hidden_dim)

    def draw_noise(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Exogenous randomness for one step: ``[batch, K*D normals | K uniforms]``."""

        if self.noise_dim == 0:
            raise RuntimeError("the deterministic particle filter draws no noise")
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        normals = torch.randn(
            batch_size, self.num_particles * self.particle_dim, generator=generator, device=device, dtype=dtype
        )
        uniforms = torch.rand(batch_size, self.num_particles, generator=generator, device=device, dtype=dtype)
        return torch.cat((normals, uniforms), dim=-1)

    def _split_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.shape[-1] != self.observation_dim:
            raise ValueError(
                f"particle-filter inputs must have width {self.observation_dim} "
                f"(features {self.feature_input_dim} + noise {self.noise_dim}), got {inputs.shape[-1]}"
            )
        features = inputs[..., : self.feature_input_dim]
        noise = inputs[..., self.feature_input_dim :]
        batch = inputs.shape[0]
        normals = noise[..., : self.num_particles * self.particle_dim].reshape(
            batch, self.num_particles, self.particle_dim
        )
        uniforms = noise[..., self.num_particles * self.particle_dim :]
        return features, normals, uniforms

    def _dpfrl_step(
        self,
        inputs: torch.Tensor,
        particles: torch.Tensor,
        log_weights: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, normals, uniforms = self._split_inputs(inputs)
        batch = features.shape[0]
        starts = episode_start.bool().view(batch, 1)
        encoded = self.input_net(features)
        tiled = encoded.unsqueeze(1).expand(batch, self.num_particles, encoded.shape[-1])

        # Initial particles: h_0^i = mu(e_0) + sigma_0(e_0) * xi^i.
        initial_std = functional.softplus(self.initial_noise_scale(encoded)) + 1e-3
        initial = (self.initial_mean(encoded) + initial_std * normals.transpose(0, 1)).transpose(0, 1)
        # PF-GRU transition with reparameterised candidate noise.
        gates = torch.sigmoid(self.gate(torch.cat((particles, tiled), dim=-1)))
        reset_gate, update_gate = gates.chunk(2, dim=-1)
        pre_candidate = self.candidate(torch.cat((reset_gate * particles, tiled), dim=-1))
        std = functional.softplus(self.noise_scale(torch.cat((particles, tiled), dim=-1))) + 1e-3
        candidate = torch.tanh(pre_candidate + std * normals)
        propagated = (1.0 - update_gate) * candidate + update_gate * particles
        new_particles = torch.where(starts.unsqueeze(-1), initial, propagated)

        # Discriminative weight update in log space, normalised (eta).
        uniform_log = torch.full_like(log_weights, -math.log(self.num_particles))
        previous = torch.where(starts, uniform_log, log_weights)
        compatibility = self.observation_function(torch.cat((new_particles, tiled), dim=-1)).squeeze(-1)
        posterior_log = torch.log_softmax(previous + compatibility, dim=-1)

        # Soft resampling every step: ancestors from q = alpha w + (1-alpha)/K,
        # importance-corrected weights w' = w_a / q_a (Ma et al. 2020).
        weights = posterior_log.exp()
        proposal = self.soft_alpha * weights + (1.0 - self.soft_alpha) / self.num_particles
        cumulative = proposal.cumsum(dim=-1)
        cumulative = cumulative / cumulative[..., -1:].clamp_min(1e-12)
        ancestors = torch.searchsorted(
            cumulative.detach(), uniforms.clamp(max=1.0 - 1e-6).contiguous()
        ).clamp(max=self.num_particles - 1)
        resampled = torch.gather(
            new_particles, 1, ancestors.unsqueeze(-1).expand(-1, -1, self.particle_dim)
        )
        corrected = torch.gather(weights, 1, ancestors) / torch.gather(proposal, 1, ancestors)
        new_log_weights = torch.log_softmax(corrected.clamp_min(1e-12).log(), dim=-1)

        # Belief summary: weighted mean particle plus MGF features.
        resampled_weights = new_log_weights.exp()
        mean_particle = (resampled_weights.unsqueeze(-1) * resampled).sum(dim=1)
        summary = [mean_particle]
        if self.mgf_features > 0:
            projections = (resampled @ self.mgf_vectors[: self.mgf_features].T).clamp(-10.0, 10.0)
            summary.append((resampled_weights.unsqueeze(-1) * projections.exp()).sum(dim=1))
        condition = self.condition_projection(torch.cat(summary, dim=-1))
        return condition, resampled, new_log_weights

    def _filter_step(
        self,
        inputs: torch.Tensor,
        particles: torch.Tensor,
        log_weights: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.variant == "dpfrl":
            return self._dpfrl_step(inputs, particles, log_weights, episode_start)
        batch = inputs.shape[0]
        tiled_inputs = inputs.unsqueeze(1).expand(batch, self.num_particles, inputs.shape[-1])
        starts = episode_start.bool().view(batch, 1, 1)

        anchor_inputs = torch.cat(
            (tiled_inputs, self.anchors.unsqueeze(0).expand(batch, -1, -1)), dim=-1
        )
        initial_particles = self.initial_net(anchor_inputs)
        propagated = self.transition_net(torch.cat((particles, tiled_inputs), dim=-1))
        new_particles = torch.where(starts, initial_particles, propagated)

        uniform = torch.full_like(log_weights, 1.0 / self.num_particles)
        previous_weights = torch.where(
            starts.view(batch, 1),
            uniform,
            torch.softmax(log_weights, dim=-1),
        )
        floored = self.soft_alpha * previous_weights + (1.0 - self.soft_alpha) / self.num_particles
        compatibility = self.weight_net(
            torch.cat((new_particles, tiled_inputs), dim=-1)
        ).squeeze(-1)
        new_log_weights = torch.log_softmax(floored.log() + compatibility, dim=-1)

        weights = new_log_weights.exp().unsqueeze(-1)
        pooled = (weights * self.feature_net(new_particles)).sum(dim=1)
        return self.condition_projection(pooled), new_particles, new_log_weights

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
        """Encode ``[T, B, I]`` inputs with per-environment episode resets.

        ``full`` keeps the particle/weight graph connected from the start of
        each episode; ``tbptt_1`` detaches the incoming packed state at every
        environment-time boundary while producing identical forward values.
        """

        self._validate_sequence_inputs(observations, initial_hidden, episode_starts)
        if temporal_gradient_mode not in {"full", "tbptt_1"}:
            raise ValueError("temporal_gradient_mode must be 'full' or 'tbptt_1'")
        hidden = initial_hidden
        outputs: list[torch.Tensor] = []
        for time_index in range(observations.shape[0]):
            if temporal_gradient_mode == "tbptt_1":
                hidden = hidden.detach()
            particles, log_weights = self._unpack(hidden)
            features, particles, log_weights = self._filter_step(
                observations[time_index],
                particles,
                log_weights,
                episode_starts[time_index],
            )
            hidden = self._pack(particles, log_weights)
            outputs.append(features)
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


RECURRENT_BASELINE_TYPES = (GRUActorCritic, ParticleFilterActorCritic)
RecurrentBaseline = GRUActorCritic | ParticleFilterActorCritic


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
    pf_num_particles: int = 16,
    pf_particle_dim: int = 8,
    pf_soft_alpha: float = 0.9,
    pf_variant: Literal["deterministic", "dpfrl"] = "deterministic",
    pf_mgf_features: int = 0,
) -> FeedForwardBaselineActorCritic | RecurrentBaseline:
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
    if kind == "particle_filter":
        return ParticleFilterActorCritic(
            observation_dim,
            **common,
            num_particles=pf_num_particles,
            particle_dim=pf_particle_dim,
            soft_alpha=pf_soft_alpha,
            variant=pf_variant,
            mgf_features=pf_mgf_features,
            policy_condition_dim=policy_condition_dim,
            head_hidden_dims=head_hidden_dims,
            continuous_policy_kind=continuous_policy_kind,
            diffusion_steps=diffusion_steps,
            diffusion_beta_start=diffusion_beta_start,
            diffusion_beta_end=diffusion_beta_end,
            diffusion_min_std=diffusion_min_std,
        )
    if kind in ("gru", "rnn"):
        return GRUActorCritic(
            observation_dim,
            **common,
            recurrent_cell=kind,
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
    "RECURRENT_BASELINE_TYPES",
    "ActorCriticEvaluation",
    "BaselineKind",
    "FeedForwardBaselineActorCritic",
    "GRUActorCritic",
    "ObservationActorCritic",
    "OracleStateActorCritic",
    "ParticleFilterActorCritic",
    "RecurrentActorCriticEvaluation",
    "RecurrentBaseline",
    "SquashedDiagonalGaussian",
    "TemporalGradientMode",
    "make_baseline_actor_critic",
]
