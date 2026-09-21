"""Categorical and diffusion policies conditioned on an encoded belief set."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal
from torch.nn import functional

from .belief import EnergyBelief
from .networks import (
    ActionValueHead,
    AlphaLSEHead,
    AlphaPoolBeliefEncoder,
    BeliefSetEncoder,
    DeepSetsBeliefEncoder,
    ObservationPredictor,
    RewardPredictor,
    expected_action_value,
    init_final_linear,
    make_head_network,
    make_trunk_network,
)


class SinusoidalStepEmbedding(nn.Module):
    def __init__(self, dimension: int = 16) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time embedding dimension must be even and at least four")
        self.dimension = dimension

    def forward(self, step: torch.Tensor, total_steps: int) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=step.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = (step.float() / max(total_steps - 1, 1))[:, None] * frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class DiffusionPolicy(nn.Module):
    """DDPM reverse chain used as an augmented stochastic policy.

    PPO operates on each saved Gaussian denoising transition rather than on an
    intractable marginal density of the final squashed action.  This is the
    central construction used by DPPO-style objectives.
    """

    def __init__(
        self,
        condition_dim: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
        hidden_dims: Sequence[int],
        *,
        num_steps: int,
        beta_start: float,
        beta_end: float,
        min_std: float,
        time_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        low = torch.as_tensor(action_low, dtype=torch.float32).flatten()
        high = torch.as_tensor(action_high, dtype=torch.float32).flatten()
        if (
            low.numel() == 0
            or low.shape != high.shape
            or not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or not torch.all(high > low)
        ):
            raise ValueError("continuous action bounds must be finite, matching, and ordered")
        if num_steps <= 0:
            raise ValueError("diffusion num_steps must be positive")
        if not 0 < beta_start <= beta_end < 1:
            raise ValueError("diffusion betas must satisfy 0 < start <= end < 1")
        if min_std <= 0:
            raise ValueError("diffusion min_std must be positive")

        self.action_dim = low.numel()
        self.num_steps = num_steps
        self.min_std = float(min_std)
        self.time_embedding = SinusoidalStepEmbedding(time_embedding_dim)
        self.denoiser = make_trunk_network(
            condition_dim + self.action_dim + time_embedding_dim,
            hidden_dims,
            self.action_dim,
        )

        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous_alpha_bars = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def transition_distribution(
        self,
        condition: torch.Tensor,
        noisy_action: torch.Tensor,
        step_index: int,
    ) -> Independent:
        if not 0 <= step_index < self.num_steps:
            raise IndexError("diffusion step is out of range")
        batch = condition.shape[0]
        step = torch.full((batch,), step_index, dtype=torch.long, device=condition.device)
        time_features = self.time_embedding(step, self.num_steps).to(condition.dtype)
        predicted_noise = self.denoiser(torch.cat((condition, noisy_action, time_features), dim=-1))

        alpha = self.alphas[step_index].to(condition.dtype)
        alpha_bar = self.alpha_bars[step_index].to(condition.dtype)
        beta = self.betas[step_index].to(condition.dtype)
        mean = noisy_action - beta * predicted_noise / torch.sqrt(1.0 - alpha_bar)
        mean = mean / torch.sqrt(alpha)
        variance = self.posterior_variance[step_index].to(condition.dtype)
        std = torch.sqrt(variance.clamp_min(self.min_std**2))
        return Independent(Normal(mean, torch.ones_like(mean) * std), 1)

    def sample(
        self,
        condition: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = condition.shape[0]
        if deterministic:
            noisy_action = torch.zeros(
                batch, self.action_dim, device=condition.device, dtype=condition.dtype
            )
        else:
            noisy_action = torch.randn(
                batch,
                self.action_dim,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        chain = [noisy_action]
        log_probabilities: list[torch.Tensor] = []
        for step_index in reversed(range(self.num_steps)):
            distribution = self.transition_distribution(condition, noisy_action, step_index)
            if deterministic:
                next_action = distribution.base_dist.loc
            else:
                noise = torch.randn(
                    noisy_action.shape,
                    device=noisy_action.device,
                    dtype=noisy_action.dtype,
                    generator=generator,
                )
                next_action = distribution.base_dist.loc + distribution.base_dist.scale * noise
            log_probabilities.append(distribution.log_prob(next_action))
            noisy_action = next_action
            chain.append(noisy_action)

        raw_action = noisy_action
        action = torch.tanh(raw_action) * self.action_scale + self.action_bias
        return action, torch.stack(chain, dim=1), torch.stack(log_probabilities, dim=1)

    def forward_noised_chain(
        self,
        actions: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Build a forward-diffused chain whose clean endpoint is ``actions``.

        ``actions`` are environment-space actions inside the bounds.  The
        returned ``[batch, num_steps + 1, action_dim]`` chain matches the
        layout produced by :meth:`sample` (index 0 most noisy, index -1 the
        raw pre-tanh action), with each intermediate element drawn from the
        DDPM forward marginal ``q(x_t | x_0)``.  Evaluating
        :meth:`log_prob_chain` on it is the standard variational
        (denoising) surrogate for the log-likelihood of an external action -
        the imitation loss used for demonstration slots.
        """

        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected actions of shape [batch, {self.action_dim}], "
                f"got {tuple(actions.shape)}"
            )
        squashed = (actions - self.action_bias) / self.action_scale
        squashed = squashed.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(squashed)
        pieces = []
        for chain_index in range(self.num_steps):
            alpha_bar = self.alpha_bars[self.num_steps - 1 - chain_index].to(raw.dtype)
            noise = torch.randn(
                raw.shape,
                device=raw.device,
                dtype=raw.dtype,
                generator=generator,
            )
            pieces.append(
                torch.sqrt(alpha_bar) * raw + torch.sqrt(1.0 - alpha_bar) * noise
            )
        pieces.append(raw)
        return torch.stack(pieces, dim=1)

    def log_prob_chain(
        self,
        condition: torch.Tensor,
        chain: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (condition.shape[0], self.num_steps + 1, self.action_dim)
        if tuple(chain.shape) != expected:
            raise ValueError(f"expected diffusion chain shape {expected}, got {tuple(chain.shape)}")
        log_probabilities: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for chain_index, step_index in enumerate(reversed(range(self.num_steps))):
            distribution = self.transition_distribution(
                condition,
                chain[:, chain_index],
                step_index,
            )
            log_probabilities.append(distribution.log_prob(chain[:, chain_index + 1]))
            entropies.append(distribution.entropy())
        return torch.stack(log_probabilities, dim=1), torch.stack(entropies, dim=1)


class TanhGaussianPolicy(nn.Module):
    """Condition-dependent diagonal Gaussian followed by an affine tanh map.

    PPO should store ``pre_tanh_action`` and pass it back to :meth:`log_prob`.
    Reconstructing it from the bounded action is also supported, but the saved
    latent is numerically preferable when tanh is close to saturation.
    """

    def __init__(
        self,
        condition_dim: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
        hidden_dims: Sequence[int],
        *,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        inverse_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        low = torch.as_tensor(action_low, dtype=torch.float32).flatten()
        high = torch.as_tensor(action_high, dtype=torch.float32).flatten()
        if (
            low.numel() == 0
            or low.shape != high.shape
            or not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or not torch.all(high > low)
        ):
            raise ValueError("continuous action bounds must be finite, matching, and ordered")
        if not math.isfinite(log_std_min) or not math.isfinite(log_std_max):
            raise ValueError("Gaussian log standard-deviation bounds must be finite")
        if log_std_min >= log_std_max:
            raise ValueError("Gaussian log_std_min must be smaller than log_std_max")
        if not 0.0 < inverse_epsilon < 1.0:
            raise ValueError("inverse_epsilon must be in (0, 1)")

        self.action_dim = int(low.numel())
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.inverse_epsilon = float(inverse_epsilon)
        self.parameter_network = make_head_network(
            condition_dim,
            hidden_dims,
            2 * self.action_dim,
        )
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def distribution(self, condition: torch.Tensor) -> Independent:
        if condition.ndim < 2:
            raise ValueError("condition must have batch dimensions followed by features")
        mean, raw_log_std = self.parameter_network(condition).chunk(2, dim=-1)
        log_std = raw_log_std.clamp(self.log_std_min, self.log_std_max)
        return Independent(Normal(mean, log_std.exp()), 1)

    def action_from_pre_tanh(self, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        if pre_tanh_action.ndim < 2 or pre_tanh_action.shape[-1] != self.action_dim:
            raise ValueError(
                f"pre_tanh_action must end in dimension {self.action_dim}, "
                f"got {tuple(pre_tanh_action.shape)}"
            )
        return torch.tanh(pre_tanh_action) * self.action_scale + self.action_bias

    def pre_tanh_from_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim < 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action must end in dimension {self.action_dim}, got {tuple(action.shape)}"
            )
        normalised = (action - self.action_bias) / self.action_scale
        tolerance = 8.0 * torch.finfo(action.dtype).eps
        if not torch.isfinite(normalised).all() or torch.any(normalised.abs() > 1.0 + tolerance):
            raise ValueError("action is non-finite or outside the configured bounds")
        normalised = normalised.clamp(
            -1.0 + self.inverse_epsilon,
            1.0 - self.inverse_epsilon,
        )
        return torch.atanh(normalised)

    def log_prob(
        self,
        condition: torch.Tensor,
        *,
        action: torch.Tensor | None = None,
        pre_tanh_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return squashed-action log density and base-Gaussian entropy.

        Exactly one action representation must be provided.  The entropy is
        the usual analytic base-Gaussian surrogate used by tanh-Gaussian PPO;
        the transformed distribution has no equally simple analytic entropy.
        """

        if (action is None) == (pre_tanh_action is None):
            raise ValueError("provide exactly one of action or pre_tanh_action")
        latent = self.pre_tanh_from_action(action) if action is not None else pre_tanh_action
        assert latent is not None
        expected = (*condition.shape[:-1], self.action_dim)
        if tuple(latent.shape) != expected:
            raise ValueError(
                f"expected Gaussian latent shape {expected}, got {tuple(latent.shape)}"
            )

        distribution = self.distribution(condition)
        log_one_minus_tanh_squared = 2.0 * (
            math.log(2.0) - latent - functional.softplus(-2.0 * latent)
        )
        log_abs_jacobian = (self.action_scale.log() + log_one_minus_tanh_squared).sum(dim=-1)
        return distribution.log_prob(latent) - log_abs_jacobian, distribution.entropy()

    def sample(
        self,
        condition: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(condition)
        if deterministic:
            pre_tanh_action = distribution.base_dist.loc
        else:
            noise = torch.randn(
                distribution.base_dist.loc.shape,
                dtype=condition.dtype,
                device=condition.device,
                generator=generator,
            )
            pre_tanh_action = distribution.base_dist.loc + distribution.base_dist.scale * noise
        action = self.action_from_pre_tanh(pre_tanh_action)
        log_prob, _ = self.log_prob(condition, pre_tanh_action=pre_tanh_action)
        return action, pre_tanh_action, log_prob


@dataclass(slots=True)
class PolicySample:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    diffusion_chain: torch.Tensor | None = None
    pre_tanh_action: torch.Tensor | None = None


class ScoreBeliefActorCritic(nn.Module):
    """Joint score-belief, permutation-invariant operator policy, and critic."""

    def __init__(
        self,
        *,
        observation_dim: int,
        state_dim: int,
        action_kind: str,
        action_feature_dim: int,
        discrete_actions: int | None,
        action_low: Sequence[float] | None,
        action_high: Sequence[float] | None,
        model_config: dict,
    ) -> None:
        super().__init__()
        self.action_kind = action_kind
        self.encoder_kind = str(model_config.get("encoder_kind", "transformer"))
        self.continuous_policy_kind = str(model_config.get("continuous_policy_kind", "diffusion"))
        self.policy_head_kind = str(model_config.get("policy_head_kind", "mlp"))
        if self.encoder_kind not in {"transformer", "deep_sets", "alpha_pool"}:
            raise ValueError(f"unsupported belief encoder: {self.encoder_kind}")
        if self.continuous_policy_kind not in {"diffusion", "gaussian"}:
            raise ValueError(f"unsupported continuous policy: {self.continuous_policy_kind}")
        if self.policy_head_kind not in {"mlp", "alpha_lse"}:
            raise ValueError(f"unsupported policy head: {self.policy_head_kind}")
        alpha_pieces = int(model_config.get("alpha_pieces", 16))
        alpha_temperature = float(model_config.get("alpha_temperature", 1.0))
        temperature_learnable = bool(model_config.get("langevin_temperature_learnable", False))
        step_size_learnable = bool(model_config.get("langevin_step_size_learnable", False))
        needs_schedule_length = temperature_learnable or step_size_learnable
        self.belief = EnergyBelief(
            observation_dim,
            state_dim,
            action_feature_dim,
            model_config["energy_hidden"],
            score_clip=model_config["score_clip"],
            particle_clip=model_config["particle_clip"],
            langevin_temperature=float(model_config.get("langevin_temperature", 1.0)),
            langevin_temperature_learnable=temperature_learnable,
            langevin_warm_start=bool(model_config.get("langevin_warm_start", False)),
            langevin_step_size_learnable=step_size_learnable,
            langevin_step_size=(
                float(model_config["langevin_step_size"]) if step_size_learnable else None
            ),
            langevin_steps=(
                int(model_config["langevin_steps"]) if needs_schedule_length else None
            ),
            langevin_schedule_bound=str(
                model_config.get("langevin_schedule_bound", "clamp")
            ),
            energy_network_kind=str(model_config.get("energy_network_kind", "mlp")),
            kan_grid_size=int(model_config.get("kan_grid_size", 8)),
            kan_spline_order=int(model_config.get("kan_spline_order", 3)),
            kan_grid_range=float(model_config.get("kan_grid_range", 3.0)),
            kan_match_parameters=bool(model_config.get("kan_match_parameters", True)),
            transition_proposal=bool(model_config.get("langevin_transition_proposal", False)),
            observation_anchor=bool(model_config.get("langevin_observation_anchor", False)),
            proposal_hidden=list(model_config.get("proposal_hidden", [64])),
        )
        encoder_use_scores = bool(model_config.get("encoder_use_scores", True))
        self.value_encoder_mode = str(model_config.get("value_encoder", "shared"))
        if self.value_encoder_mode not in {"shared", "separate", "detached"}:
            raise ValueError("value_encoder must be shared, separate, or detached")

        def build_encoder() -> nn.Module:
            if self.encoder_kind == "transformer":
                return BeliefSetEncoder(
                    state_dim=state_dim,
                    d_model=model_config["d_model"],
                    num_heads=model_config["num_heads"],
                    num_layers=model_config["num_transformer_layers"],
                    feedforward_dim=model_config["transformer_ff_dim"],
                    dropout=model_config["dropout"],
                    use_scores=encoder_use_scores,
                )
            if self.encoder_kind == "alpha_pool":
                return AlphaPoolBeliefEncoder(
                    state_dim=state_dim,
                    d_model=model_config["d_model"],
                    feedforward_dim=model_config["transformer_ff_dim"],
                    use_scores=bool(model_config.get("alpha_use_scores", False)),
                )
            return DeepSetsBeliefEncoder(
                state_dim=state_dim,
                d_model=model_config["d_model"],
                feedforward_dim=model_config["transformer_ff_dim"],
                dropout=model_config["dropout"],
                use_scores=encoder_use_scores,
            )

        self.encoder: nn.Module = build_encoder()
        # ``separate``: the critic reads the particles through its own encoder, so
        # the value loss never shapes the policy's belief representation.
        self.value_encoder: nn.Module | None = (
            build_encoder() if self.value_encoder_mode == "separate" else None
        )
        condition_dim = model_config["d_model"]
        self.observation_prediction_coef = float(
            model_config.get("observation_prediction_coef", 0.0)
        )
        if self.observation_prediction_coef > 0.0:
            self.observation_predictor: ObservationPredictor | None = (
                ObservationPredictor(
                    state_dim,
                    action_feature_dim,
                    observation_dim,
                    model_config.get("observation_predictor_hidden", [64]),
                )
            )
        else:
            self.observation_predictor = None
        self.reward_prediction_coef = float(model_config.get("reward_prediction_coef", 0.0))
        if self.reward_prediction_coef > 0.0:
            self.reward_predictor: RewardPredictor | None = RewardPredictor(
                state_dim,
                action_feature_dim,
                model_config.get("reward_predictor_hidden", [64]),
            )
        else:
            self.reward_predictor = None
        self.critic_kind = str(model_config.get("critic_kind", "state"))
        if self.critic_kind not in {"state", "action"}:
            raise ValueError("critic_kind must be state or action")
        self.q_value_samples = int(model_config.get("q_value_samples", 8))
        if self.critic_kind == "action":
            # The action-value critic replaces the state-value head entirely.
            self.value_head: nn.Module | None = None
        elif self.policy_head_kind == "alpha_lse":
            # A convex (PWLC) value functional; with the alpha_pool encoder the
            # value is exactly a smooth max of belief-linear functionals.
            self.value_head = AlphaLSEHead(condition_dim, 1, alpha_pieces, alpha_temperature)
        else:
            self.value_head = make_head_network(
                condition_dim,
                model_config["policy_hidden"],
                1,
            )
        if action_kind == "discrete":
            if not discrete_actions:
                raise ValueError("a discrete policy requires a positive action count")
            if self.policy_head_kind == "alpha_lse":
                self.categorical_head: nn.Module | None = AlphaLSEHead(
                    condition_dim,
                    discrete_actions,
                    alpha_pieces,
                    alpha_temperature,
                    init_gain=0.01,
                )
            else:
                self.categorical_head = make_head_network(
                    condition_dim,
                    model_config["policy_hidden"],
                    discrete_actions,
                )
                init_final_linear(self.categorical_head, 0.01)
            self.diffusion_policy: DiffusionPolicy | None = None
            self.gaussian_policy: TanhGaussianPolicy | None = None
        elif action_kind == "continuous":
            if action_low is None or action_high is None:
                raise ValueError("a continuous policy requires action bounds")
            self.categorical_head = None
            if self.continuous_policy_kind == "diffusion":
                self.diffusion_policy = DiffusionPolicy(
                    condition_dim,
                    action_low,
                    action_high,
                    model_config["policy_hidden"],
                    num_steps=model_config["diffusion_steps"],
                    beta_start=model_config["diffusion_beta_start"],
                    beta_end=model_config["diffusion_beta_end"],
                    min_std=model_config["diffusion_min_std"],
                )
                self.gaussian_policy = None
            else:
                self.diffusion_policy = None
                self.gaussian_policy = TanhGaussianPolicy(
                    condition_dim,
                    action_low,
                    action_high,
                    model_config["policy_hidden"],
                    log_std_min=float(model_config.get("gaussian_log_std_min", -5.0)),
                    log_std_max=float(model_config.get("gaussian_log_std_max", 2.0)),
                )
        else:
            raise ValueError(f"unsupported action kind: {action_kind}")
        if self.critic_kind == "action":
            self.action_value_head: ActionValueHead | None = ActionValueHead(
                condition_dim,
                model_config["policy_hidden"],
                discrete_actions=discrete_actions if action_kind == "discrete" else None,
                action_dim=None if action_kind == "discrete" else len(action_low or ()),
            )
        else:
            self.action_value_head = None

    def encode(self, particles: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return self.encoder(particles, scores)

    def encode_value(self, particles: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Condition vector for the critic (own encoder, detached shared, or shared)."""

        if self.value_encoder is not None:
            return self.value_encoder(particles, scores)
        condition = self.encoder(particles, scores)
        return condition.detach() if self.value_encoder_mode == "detached" else condition

    def encode_both(
        self, particles: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy and critic conditions, sharing one encoder pass when possible."""

        condition = self.encoder(particles, scores)
        if self.value_encoder is not None:
            return condition, self.value_encoder(particles, scores)
        if self.value_encoder_mode == "detached":
            return condition, condition.detach()
        return condition, condition

    def value(self, condition: torch.Tensor) -> torch.Tensor:
        """State value: the critic output, or ``E_pi Q`` for the action-value critic."""

        if self.action_value_head is not None:
            return self.expected_action_value(condition)
        assert self.value_head is not None
        return self.value_head(condition).squeeze(-1)

    def action_value(self, condition: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.action_value_head is None:
            raise ValueError("the model has no action-value critic")
        return self.action_value_head(condition, action)

    def expected_action_value(self, condition: torch.Tensor) -> torch.Tensor:
        if self.action_value_head is None:
            raise ValueError("the model has no action-value critic")
        logits = None
        if self.action_kind == "discrete":
            assert self.categorical_head is not None
            logits = self.categorical_head(condition)
        return expected_action_value(
            self.action_value_head,
            condition,
            categorical_logits=logits,
            sample_actions=lambda c: self.sample_policy(c, deterministic=False).action,
            num_samples=self.q_value_samples,
        )

    def sample_policy(
        self,
        condition: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> PolicySample:
        if self.action_kind == "discrete":
            assert self.categorical_head is not None
            distribution = Categorical(logits=self.categorical_head(condition))
            if deterministic:
                action = distribution.probs.argmax(dim=-1)
            elif generator is None:
                # ``Categorical.sample`` draws from the global RNG, which keeps
                # the trainer's default path unchanged.
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

        if self.continuous_policy_kind == "diffusion":
            assert self.diffusion_policy is not None
            action, chain, step_log_prob = self.diffusion_policy.sample(
                condition,
                deterministic=deterministic,
                generator=generator,
            )
            _, step_entropy = self.diffusion_policy.log_prob_chain(condition, chain)
            return PolicySample(
                action=action,
                log_prob=step_log_prob,
                entropy=step_entropy,
                diffusion_chain=chain,
            )

        assert self.gaussian_policy is not None
        action, pre_tanh_action, log_prob = self.gaussian_policy.sample(
            condition,
            deterministic=deterministic,
            generator=generator,
        )
        _, entropy = self.gaussian_policy.log_prob(
            condition,
            pre_tanh_action=pre_tanh_action,
        )
        return PolicySample(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            pre_tanh_action=pre_tanh_action,
        )
