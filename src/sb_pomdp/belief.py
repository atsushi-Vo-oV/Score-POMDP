"""Energy-defined belief scores and unadjusted Langevin particle sampling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from .networks import make_mlp

TemporalGradientMode = Literal["full", "tbptt_1"]
PotentialBranch = Literal["initial", "recursive", "mixed"]


def _langevin_step_sizes(
    num_steps: int,
    step_size: float | Sequence[float],
) -> list[float]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if isinstance(step_size, Sequence):
        steps = [float(value) for value in step_size]
        if len(steps) != num_steps:
            raise ValueError("step-size schedule length must equal num_steps")
    else:
        steps = [float(step_size)] * num_steps
    if any(value <= 0 for value in steps):
        raise ValueError("all Langevin step sizes must be positive")
    return steps


def score_from_potential(
    potential: torch.Tensor,
    query: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Return ``nabla_query sum(potential)`` with an explicit graph policy."""

    (score,) = torch.autograd.grad(
        potential.sum(),
        query,
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    return score


# Bounds for the learned per-step log-temperature schedule; they stop the PPO
# gradient from freezing the chain entirely or re-inflating it without limit.
_LOG_TEMPERATURE_MIN = math.log(1e-3)
_LOG_TEMPERATURE_MAX = math.log(4.0)

# Bounds for the learned per-step log-step-size (drift) schedule; they keep the
# discretised chain away from a frozen or numerically unstable regime.
_LOG_STEP_SIZE_MIN = math.log(1e-4)
_LOG_STEP_SIZE_MAX = math.log(0.5)
_SCHEDULE_BOUNDS = ("clamp", "sigmoid")
_LOGIT_EPS = 1e-6


def _bound_log_schedule(
    raw: torch.Tensor, low: float, high: float, mode: str
) -> torch.Tensor:
    """Map a raw schedule parameter to a bounded log value.

    ``clamp`` is the original hard projection: outside ``[low, high]`` the
    gradient is exactly zero, so a parameter pushed past a bound can never
    return.  ``sigmoid`` parameterises ``low + (high - low) * sigmoid(raw)``,
    which keeps a non-zero gradient everywhere and approaches the bounds only
    asymptotically.
    """

    if mode == "clamp":
        return raw.clamp(low, high)
    if mode == "sigmoid":
        return low + (high - low) * torch.sigmoid(raw)
    raise ValueError(f"unsupported schedule bound: {mode}")


def _unbound_log_schedule(value: float, low: float, high: float, mode: str) -> float:
    """Inverse of :func:`_bound_log_schedule` for parameter initialisation."""

    if mode == "clamp":
        return float(value)
    if mode == "sigmoid":
        fraction = (float(value) - low) / (high - low)
        fraction = min(max(fraction, _LOGIT_EPS), 1.0 - _LOGIT_EPS)
        return math.log(fraction / (1.0 - fraction))
    raise ValueError(f"unsupported schedule bound: {mode}")


class EnergyBelief(nn.Module):
    """Implements the recursive score field in ``sb-pomdp-retry.tex``.

    ``g_0``, ``f``, and ``u`` return scalar log-potentials.  At episode start,
    a dedicated initial-belief network ``g_0(o_0, s)`` is combined with a
    standard-normal base energy.  Later steps use the observation network ``f``
    plus the log-mean-exp transition energy ``u``.  The initial and recursive
    observation networks have independent parameters.  Log-mean-exp differs
    from log-sum-exp only by a score-independent constant and is stable across
    K.

    ``langevin_temperature`` tempers the ULA noise to sample ``exp(-E/tau)``;
    with ``langevin_temperature_learnable`` the per-step ``log tau`` schedule
    becomes a trained parameter.  ``langevin_step_size_learnable`` likewise
    turns the per-step drift schedule ``alpha_l`` into a trained parameter
    initialised at the configured step size, overriding the ``step_size``
    argument of the update.  ``langevin_schedule_bound`` selects how the
    learned schedules are kept inside their ranges (``clamp`` = hard
    projection with a dead gradient outside, ``sigmoid`` = smooth bound with
    a live gradient everywhere).  ``langevin_warm_start`` initialises the chain
    of a recursive step at the previous particles instead of fresh noise, which
    turns the particle positions themselves into a memory carrier.
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        action_feature_dim: int,
        hidden_dims: Sequence[int],
        *,
        score_clip: float = 0.0,
        particle_clip: float = 0.0,
        langevin_temperature: float = 1.0,
        langevin_temperature_learnable: bool = False,
        langevin_warm_start: bool = False,
        langevin_step_size_learnable: bool = False,
        langevin_step_size: float | None = None,
        langevin_steps: int | None = None,
        langevin_schedule_bound: str = "clamp",
    ) -> None:
        super().__init__()
        if langevin_schedule_bound not in _SCHEDULE_BOUNDS:
            raise ValueError("langevin_schedule_bound must be clamp or sigmoid")
        self.langevin_schedule_bound = str(langevin_schedule_bound)
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        self.action_feature_dim = action_feature_dim
        self.score_clip = float(score_clip)
        self.particle_clip = float(particle_clip)
        if not float(langevin_temperature) > 0.0:
            raise ValueError("langevin_temperature must be positive")
        self.langevin_temperature = float(langevin_temperature)
        self.langevin_warm_start = bool(langevin_warm_start)
        if langevin_temperature_learnable:
            if langevin_steps is None or int(langevin_steps) <= 0:
                raise ValueError(
                    "a learnable temperature schedule needs a positive langevin_steps"
                )
            self.langevin_log_temperature: nn.Parameter | None = nn.Parameter(
                torch.full(
                    (int(langevin_steps),),
                    _unbound_log_schedule(
                        math.log(self.langevin_temperature),
                        _LOG_TEMPERATURE_MIN,
                        _LOG_TEMPERATURE_MAX,
                        self.langevin_schedule_bound,
                    ),
                )
            )
        else:
            self.langevin_log_temperature = None
        if langevin_step_size_learnable:
            if langevin_steps is None or int(langevin_steps) <= 0:
                raise ValueError(
                    "a learnable step-size schedule needs a positive langevin_steps"
                )
            if langevin_step_size is None or not float(langevin_step_size) > 0.0:
                raise ValueError(
                    "a learnable step-size schedule needs a positive langevin_step_size"
                )
            self.langevin_log_step_size: nn.Parameter | None = nn.Parameter(
                torch.full(
                    (int(langevin_steps),),
                    _unbound_log_schedule(
                        math.log(float(langevin_step_size)),
                        _LOG_STEP_SIZE_MIN,
                        _LOG_STEP_SIZE_MAX,
                        self.langevin_schedule_bound,
                    ),
                )
            )
        else:
            self.langevin_log_step_size = None
        self.initial_energy = make_mlp(
            observation_dim + state_dim,
            hidden_dims,
            1,
        )
        self.observation_energy = make_mlp(
            observation_dim + state_dim + action_feature_dim,
            hidden_dims,
            1,
        )
        self.transition_energy = make_mlp(
            2 * state_dim + action_feature_dim,
            hidden_dims,
            1,
        )

    def _validate_context(
        self,
        query: torch.Tensor,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
    ) -> None:
        if query.ndim != 3 or query.shape[-1] != self.state_dim:
            raise ValueError("query must have shape [batch, K, state_dim]")
        batch = query.shape[0]
        expected = {
            "observation": (batch, self.observation_dim),
            "previous_action": (batch, self.action_feature_dim),
            "initial": (batch,),
        }
        actual = {
            "observation": tuple(observation.shape),
            "previous_action": tuple(previous_action.shape),
            "initial": tuple(initial.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} must have shape {shape}, got {actual[name]}")
        if previous_particles.ndim != 3 or previous_particles.shape[0] != batch:
            raise ValueError("previous_particles must have shape [batch, K_prev, state_dim]")
        if previous_particles.shape[-1] != self.state_dim:
            raise ValueError("previous particle state dimension does not match")

    def classify_potential_branch(self, initial: torch.Tensor) -> PotentialBranch:
        """Classify one vectorised environment step with one device synchronisation.

        ULA evaluates the score repeatedly for the same environment context.  The
        caller can classify that context once outside the Langevin loop (and
        outside activation-checkpoint recomputation), then reuse the returned
        Python value for every score evaluation.
        """

        if initial.ndim != 1:
            raise ValueError("initial must have shape [batch]")
        initial_count = int(torch.count_nonzero(initial.bool()).item())
        if initial_count == initial.numel():
            return "initial"
        if initial_count == 0:
            return "recursive"
        return "mixed"

    def classify_potential_branches(
        self,
        initial: torch.Tensor,
    ) -> tuple[PotentialBranch, ...]:
        """Classify many ``[batch]`` contexts with one device-to-host transfer."""

        if initial.ndim != 2:
            raise ValueError("initial must have shape [time, batch]")
        batch_size = initial.shape[1]
        initial_counts = torch.count_nonzero(initial.bool(), dim=1).detach().cpu().tolist()
        return tuple(
            "initial"
            if initial_count == batch_size
            else "recursive"
            if initial_count == 0
            else "mixed"
            for initial_count in initial_counts
        )

    def _initial_log_potential(
        self,
        query: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        num_query = query.shape[1]
        obs = observation[:, None, :].expand(-1, num_query, -1)
        initial_input = torch.cat((obs, query), dim=-1)
        initial_value = self.initial_energy(initial_input).squeeze(-1)
        return initial_value - 0.5 * query.square().sum(dim=-1)

    def _recursive_log_potential(
        self,
        query: torch.Tensor,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        num_query = query.shape[1]
        num_previous = previous_particles.shape[1]
        obs = observation[:, None, :].expand(-1, num_query, -1)
        action = previous_action[:, None, :].expand(-1, num_query, -1)
        observation_input = torch.cat((obs, query, action), dim=-1)
        f_value = self.observation_energy(observation_input).squeeze(-1)

        query_pairs = query[:, :, None, :].expand(-1, -1, num_previous, -1)
        previous_pairs = previous_particles[:, None, :, :].expand(-1, num_query, -1, -1)
        action_pairs = previous_action[:, None, None, :].expand(
            -1, num_query, num_previous, -1
        )
        transition_input = torch.cat((query_pairs, previous_pairs, action_pairs), dim=-1)
        u_value = self.transition_energy(transition_input).squeeze(-1)
        transition_term = torch.logsumexp(u_value, dim=-1) - math.log(num_previous)
        return f_value + transition_term

    def _log_potential_for_branch(
        self,
        query: torch.Tensor,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
        potential_branch: PotentialBranch,
    ) -> torch.Tensor:
        """Evaluate only the energy networks selected by a validated context."""

        if potential_branch == "initial":
            return self._initial_log_potential(query, observation)
        if potential_branch == "recursive":
            return self._recursive_log_potential(
                query,
                observation,
                previous_particles,
                previous_action,
            )
        if potential_branch != "mixed":
            raise ValueError("potential_branch must be initial, recursive, or mixed")

        # Boolean subset extraction has dynamic output shapes on CUDA and can
        # synchronise the host on every ULA score call.  A mixed vectorised step
        # genuinely needs all three networks, so retain the dense, synchronisation-
        # free expression in this uncommon case.  Homogeneous steps take the fast
        # paths above and never construct their inactive branch.
        initial_potential = self._initial_log_potential(query, observation)
        recursive_potential = self._recursive_log_potential(
            query,
            observation,
            previous_particles,
            previous_action,
        )
        return torch.where(initial[:, None].bool(), initial_potential, recursive_potential)

    def log_potential(
        self,
        query: torch.Tensor,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the unnormalised scalar potential at all query particles."""

        self._validate_context(query, observation, previous_particles, previous_action, initial)
        potential_branch = self.classify_potential_branch(initial)
        return self._log_potential_for_branch(
            query,
            observation,
            previous_particles,
            previous_action,
            initial,
            potential_branch,
        )

    def _score_at_for_branch(
        self,
        particles: torch.Tensor,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
        potential_branch: PotentialBranch,
        *,
        create_graph: bool,
        detach_query: bool,
    ) -> torch.Tensor:
        """Score implementation for a prevalidated, preclassified context."""

        with torch.enable_grad():
            if detach_query:
                query = particles.detach().requires_grad_(True)
            elif particles.requires_grad:
                query = particles
            else:
                query = particles.detach().requires_grad_(True)
            potential = self._log_potential_for_branch(
                query,
                observation,
                previous_particles,
                previous_action,
                initial,
                potential_branch,
            )
            score = score_from_potential(potential, query, create_graph=create_graph)
            if self.score_clip > 0:
                score = score.clamp(-self.score_clip, self.score_clip)
        return score if create_graph else score.detach()

    def score_at(
        self,
        particles: torch.Tensor,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
        *,
        create_graph: bool,
        detach_query: bool = True,
    ) -> torch.Tensor:
        """Evaluate the learned score at particle coordinates.

        ``detach_query=True`` preserves the original fixed-coordinate API.
        Differentiable Langevin replay sets it to false so policy losses flow
        through every ULA update as a pathwise gradient.  A coordinate that has
        no upstream graph is made a leaf requiring gradients without changing
        its numerical value.
        """

        self._validate_context(
            particles,
            observation,
            previous_particles,
            previous_action,
            initial,
        )
        potential_branch = self.classify_potential_branch(initial)
        return self._score_at_for_branch(
            particles,
            observation,
            previous_particles,
            previous_action,
            initial,
            potential_branch,
            create_graph=create_graph,
            detach_query=detach_query,
        )

    def draw_particle_noise(
        self,
        observation: torch.Tensor,
        *,
        num_particles: int,
        num_steps: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw the exogenous Gaussian noise used by one belief update.

        The returned shapes are ``[batch, K, state_dim]`` for the initial ULA
        state and ``[batch, L, K, state_dim]`` for the per-step increments.
        Saving these tensors allows PPO to replay exactly the same stochastic
        belief computation under the current parameters on every epoch.
        """

        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"observation must have shape [batch, {self.observation_dim}]"
            )
        if num_particles <= 0 or num_steps <= 0:
            raise ValueError("num_particles and num_steps must be positive")
        batch = observation.shape[0]
        initial_noise = torch.randn(
            batch,
            num_particles,
            self.state_dim,
            device=observation.device,
            dtype=observation.dtype,
            generator=generator,
        )
        langevin_noise = torch.randn(
            batch,
            num_steps,
            num_particles,
            self.state_dim,
            device=observation.device,
            dtype=observation.dtype,
            generator=generator,
        )
        return initial_noise, langevin_noise

    def temperature_schedule(self) -> torch.Tensor | None:
        """Effective per-step ``tau_l`` (bounded, exponentiated) or ``None``."""

        if self.langevin_log_temperature is None:
            return None
        return _bound_log_schedule(
            self.langevin_log_temperature,
            _LOG_TEMPERATURE_MIN,
            _LOG_TEMPERATURE_MAX,
            self.langevin_schedule_bound,
        ).exp()

    def step_size_schedule(self) -> torch.Tensor | None:
        """Effective per-step ``alpha_l`` (bounded, exponentiated) or ``None``."""

        if self.langevin_log_step_size is None:
            return None
        return _bound_log_schedule(
            self.langevin_log_step_size,
            _LOG_STEP_SIZE_MIN,
            _LOG_STEP_SIZE_MAX,
            self.langevin_schedule_bound,
        ).exp()

    def particles_from_noise(
        self,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
        *,
        initial_noise: torch.Tensor,
        langevin_noise: torch.Tensor,
        step_size: float | Sequence[float],
        track_grad: bool,
        temporal_gradient_mode: TemporalGradientMode = "full",
        potential_branch: PotentialBranch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay one reparameterized ULA belief update from explicit noise.

        ``temporal_gradient_mode="full"`` retains the graph carried by the
        previous time step.  Mode ``"tbptt_1"`` detaches only that incoming
        belief state; all ULA steps *within* the current environment time step
        remain differentiable.  Environment observations and realised actions
        are treated as replay inputs by the caller rather than differentiated
        through an environment model.
        """

        if temporal_gradient_mode not in {"full", "tbptt_1"}:
            raise ValueError("temporal_gradient_mode must be 'full' or 'tbptt_1'")
        if initial_noise.ndim != 3:
            raise ValueError("initial_noise must have shape [batch, K, state_dim]")
        expected_initial_shape = (
            observation.shape[0],
            initial_noise.shape[1],
            self.state_dim,
        )
        if tuple(initial_noise.shape) != expected_initial_shape:
            raise ValueError(f"initial_noise must have shape {expected_initial_shape}")
        if langevin_noise.ndim != 4:
            raise ValueError("langevin_noise must have shape [batch, L, K, state_dim]")
        expected_noise_tail = (initial_noise.shape[1], self.state_dim)
        if (
            langevin_noise.shape[0] != observation.shape[0]
            or tuple(langevin_noise.shape[2:]) != expected_noise_tail
        ):
            raise ValueError(
                "langevin_noise batch, particle, and state dimensions must match initial_noise"
            )
        for name, value in (
            ("initial_noise", initial_noise),
            ("langevin_noise", langevin_noise),
        ):
            if value.device != observation.device or value.dtype != observation.dtype:
                raise ValueError(f"{name} must share observation dtype and device")

        num_steps = int(langevin_noise.shape[1])
        steps = _langevin_step_sizes(num_steps, step_size)
        incoming_particles = (
            previous_particles.detach()
            if temporal_gradient_mode == "tbptt_1"
            else previous_particles
        )
        # Noise is an exogenous replay input.  Detaching it prevents callers
        # from accidentally optimising the sampled random numbers themselves.
        particles = initial_noise.detach()
        increments = langevin_noise.detach()
        self._validate_context(
            particles,
            observation,
            incoming_particles,
            previous_action,
            initial,
        )
        if potential_branch is None:
            potential_branch = self.classify_potential_branch(initial)
        elif potential_branch not in {"initial", "recursive", "mixed"}:
            raise ValueError("potential_branch must be initial, recursive, or mixed")

        if self.langevin_warm_start:
            if incoming_particles.shape[1] != particles.shape[1]:
                raise ValueError(
                    "langevin_warm_start needs matching previous and current particle counts"
                )
            particles = torch.where(initial.view(-1, 1, 1), particles, incoming_particles)

        if self.langevin_log_temperature is None:
            step_temperatures = None
        else:
            if self.langevin_log_temperature.shape[0] != len(steps):
                raise ValueError(
                    "the learned temperature schedule length must equal the Langevin step count"
                )
            step_temperatures = self.temperature_schedule()

        if self.langevin_log_step_size is None:
            step_alphas = None
        else:
            if self.langevin_log_step_size.shape[0] != len(steps):
                raise ValueError(
                    "the learned step-size schedule length must equal the Langevin step count"
                )
            step_alphas = self.step_size_schedule()

        with torch.set_grad_enabled(track_grad):
            for step_index, alpha in enumerate(steps):
                score = self._score_at_for_branch(
                    particles,
                    observation,
                    incoming_particles,
                    previous_action,
                    initial,
                    potential_branch,
                    create_graph=track_grad,
                    detach_query=not track_grad,
                )
                alpha_value = alpha if step_alphas is None else step_alphas[step_index]
                if step_alphas is None and step_temperatures is None:
                    noise_scale = math.sqrt(2.0 * alpha * self.langevin_temperature)
                else:
                    temperature_value = (
                        self.langevin_temperature
                        if step_temperatures is None
                        else step_temperatures[step_index]
                    )
                    noise_scale = torch.sqrt(2.0 * alpha_value * temperature_value)
                particles = (
                    particles
                    + alpha_value * score
                    + noise_scale * increments[:, step_index]
                )
                if self.particle_clip > 0:
                    particles = particles.clamp(-self.particle_clip, self.particle_clip)
                if not track_grad:
                    particles = particles.detach()

            scores = self._score_at_for_branch(
                particles,
                observation,
                incoming_particles,
                previous_action,
                initial,
                potential_branch,
                create_graph=track_grad,
                detach_query=not track_grad,
            )
        if track_grad:
            return particles, scores
        return particles.detach(), scores.detach()

    def sample_particles(
        self,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
        *,
        num_particles: int,
        num_steps: int,
        step_size: float | Sequence[float],
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample particles with the unadjusted Langevin algorithm (ULA)."""

        steps = _langevin_step_sizes(num_steps, step_size)
        initial_noise, langevin_noise = self.draw_particle_noise(
            observation,
            num_particles=num_particles,
            num_steps=num_steps,
            generator=generator,
        )
        return self.particles_from_noise(
            observation,
            previous_particles,
            previous_action,
            initial,
            initial_noise=initial_noise,
            langevin_noise=langevin_noise,
            step_size=steps,
            track_grad=False,
        )
