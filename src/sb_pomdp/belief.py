"""Energy-defined belief scores and unadjusted Langevin particle sampling."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from .config import validate_belief_acceleration
from .networks import NETWORK_KINDS, make_mlp, make_network, matched_kan_hidden_dims

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


_ANCHOR_LOG_SCALE_MIN = -4.0
_ANCHOR_LOG_SCALE_MAX = 2.0


def _zero_final_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Module:
    """MLP whose last linear layer starts at zero (an identity-preserving proposal)."""

    network = make_mlp(input_dim, list(hidden_dims), output_dim)
    for layer in reversed(list(network.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            break
    return network


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


def _mlp_value_and_input_gradient(
    network: nn.Sequential, inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar Linear/SiLU MLP and its input gradient, using ordinary forward ops.

    The reverse chain rule keeps the activations and weights in the outer graph;
    no Jacobian is materialised and no inner autograd invocation is needed.
    """

    value = inputs
    activations = []
    for layer in network:
        activations.append(value)
        value = layer(value)
    gradient = torch.ones_like(value)
    for layer, activation in zip(reversed(network), reversed(activations)):
        if isinstance(layer, nn.Linear):
            gradient = gradient @ layer.weight
        elif isinstance(layer, nn.SiLU):
            sigmoid = activation.sigmoid()
            gradient = gradient * (sigmoid * (1.0 + activation * (1.0 - sigmoid)))
        else:
            raise TypeError("analytic score requires a scalar Linear/SiLU MLP")
    return value.squeeze(-1), gradient


# Bounds for the learned per-step log-temperature schedule; they stop the PPO
# gradient from freezing the chain entirely or re-inflating it without limit.
_LOG_TEMPERATURE_MIN = math.log(1e-3)
_LOG_TEMPERATURE_MAX = math.log(4.0)

# Bounds for the learned per-step log-step-size (drift) schedule; they keep the
# discretised chain away from a frozen or numerically unstable regime.
_LOG_STEP_SIZE_MIN = math.log(1e-4)
_LOG_STEP_SIZE_MAX = math.log(0.5)

# Bounds for the learned per-step log-noise-scale schedule ``beta_l`` (the
# coefficient of the Gaussian increment when it is decoupled from the drift).
_LOG_NOISE_SCALE_MIN = math.log(1e-3)
_LOG_NOISE_SCALE_MAX = math.log(2.0)
_SCHEDULE_BOUNDS = ("clamp", "sigmoid")
PARTICLE_SAMPLERS = ("ula", "svgd", "smi")
SCORE_PARAMETERIZATIONS = ("energy", "direct", "direct_joint", "direct_flat")
# "branch": t = 0 has its own initial network (+ optional observation anchor);
# "prior": t = 0 is a recursive step over K learned prior particles and a zero action.
INITIAL_BELIEFS = ("branch", "prior")
# Width of the previous-belief summary c(x) that the joint direct score network reads.
_JOINT_CONTEXT_DIM = 8
# SMI component scales s_k are kept in [1e-3, 2] (same range as the ULA noise scale).
_LOG_SMI_SCALE_MIN = math.log(1e-3)
_LOG_SMI_SCALE_MAX = math.log(2.0)


def _rbf_stein_direction(particles: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    """SVGD direction ``phi(x_i) = mean_j [k(x_j, x_i) drive_j + grad_{x_j} k(x_j, x_i)]``.

    ``k(x, y) = exp(-|x - y|^2 / h)`` with the median heuristic ``h = med |x_i - x_j|^2 /
    log(K + 1)`` per batch row (Liu & Wang, 2016), detached so the bandwidth is a constant of
    the step.  ``particles`` and ``drive`` are ``[batch, K, state_dim]``.
    """

    difference = particles[:, :, None, :] - particles[:, None, :, :]  # x_i - x_j
    squared = difference.square().sum(dim=-1)
    num_particles = particles.shape[1]
    with torch.no_grad():
        if num_particles > 1:
            # Fixed-size gather, same row-major order as the original boolean
            # mask. Device arange avoids a host-to-device copy during capture.
            rows = torch.arange(num_particles, device=particles.device)[:, None]
            columns = torch.arange(num_particles - 1, device=particles.device)[None, :]
            indices = (rows * num_particles + columns + (columns >= rows)).flatten()
            median = squared.detach().flatten(1).index_select(1, indices).median(dim=-1).values
        else:
            median = torch.ones(particles.shape[0], device=particles.device, dtype=particles.dtype)
        bandwidth = (median / math.log(num_particles + 1.0)).clamp_min(1e-6)
    kernel = torch.exp(-squared / bandwidth[:, None, None])
    attraction = torch.einsum("bij,bjd->bid", kernel, drive)
    repulsion = (2.0 / bandwidth[:, None, None]) * (kernel[..., None] * difference).sum(dim=2)
    return (attraction + repulsion) / num_particles


def _mixture_entropy_gradient(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    noise: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``-grad log q(z_j)`` w.r.t. ``(theta_j, log s_j)`` for ``q = mean_l N(theta_l, diag(s_l^2))``.

    ``z_j = theta_j + s_j * eps_j`` are the reparameterised draws of each component, so the
    gradient runs through the draw and through every mixture component (the SMI entropy term).
    """

    with torch.enable_grad():
        theta = means if means.requires_grad else means.detach().requires_grad_(True)
        log_s = log_scales if log_scales.requires_grad else log_scales.detach().requires_grad_(True)
        draws = theta + log_s.exp() * noise.detach()
        standardized = (draws[:, :, None, :] - theta[:, None, :, :]) / log_s.exp()[:, None, :, :]
        component = -0.5 * standardized.square().sum(dim=-1) - log_s.sum(dim=-1)[:, None, :]
        log_mixture = torch.logsumexp(component, dim=-1)
        grad_theta, grad_log_s = torch.autograd.grad(
            log_mixture.sum(), (theta, log_s), create_graph=create_graph
        )
    if not create_graph:
        grad_theta, grad_log_s = grad_theta.detach(), grad_log_s.detach()
    return -grad_theta, -grad_log_s


def _analytic_mixture_entropy_gradient(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    noise: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form SMI entropy derivative through draws AND mixture components."""

    with torch.set_grad_enabled(create_graph):
        scale = log_scales.exp()
        displacement = scale * noise.detach()
        draws = means + displacement
        difference = draws[:, :, None, :] - means[:, None, :, :]
        standardized = difference / scale[:, None, :, :]
        component = -0.5 * standardized.square().sum(-1) - log_scales.sum(-1)[:, None, :]
        # Same stable normalization as logsumexp's backward in the reference.
        responsibility = (component - component.logsumexp(-1, keepdim=True)).exp()
        weighted = responsibility[..., None] * standardized / scale[:, None, :, :]
        draw_score = -weighted.sum(dim=2)
        grad_means = draw_score + weighted.sum(dim=1)
        grad_log_scales = draw_score * displacement + (
            responsibility[..., None] * (standardized.square() - 1.0)
        ).sum(dim=1)
    return -grad_means, -grad_log_scales
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
    argument of the update.  ``langevin_noise_scale`` decouples the Gaussian
    increment from the drift: the chain becomes ``x + alpha_l * score +
    beta_l * xi`` with ``beta_l`` the given constant (the coupled sampler has
    ``beta = sqrt(2 alpha tau)``), and ``langevin_noise_scale_learnable``
    makes the per-step ``log beta_l`` a trained parameter initialised there.
    ``langevin_schedule_bound`` selects how the
    learned schedules are kept inside their ranges (``clamp`` = hard
    projection with a dead gradient outside, ``sigmoid`` = smooth bound with
    a live gradient everywhere).  ``langevin_warm_start`` initialises the chain
    of a recursive step at the previous particles instead of fresh noise, which
    turns the particle positions themselves into a memory carrier.
    ``transition_proposal`` (warm start only) first moves those previous
    particles by a learned drift ``g(s_{t-1}, a_{t-1})`` - the explicit
    prediction step of a particle filter - so the chain starts where the state
    plausibly went instead of where it was; ``observation_anchor`` replaces the
    standard-normal chain start of an *initial* step by a learned proposal
    ``mu(o_0) + sigma(o_0) * xi``.  Both are zero-initialised, so a fresh model
    is bit-identical to the plain cold/warm sampler, and both are trained only
    through the policy/auxiliary gradients that flow into the chain (no state).

    ``initial_belief`` selects how the first step of an episode is scored.
    ``"branch"`` (the original model) gives t = 0 its own network: the initial
    potential ``g(o_0, x) - |x|^2 / 2`` (energy) or the score ``g(o_0, x) - x``
    (direct), with the observation anchor as its optional chain start.  ``"prior"``
    removes that branch: an initial row is scored by the *recursive* networks,
    exactly like every later step, with ``K`` learned prior particles
    ``x_prior`` (``num_particles x state_dim``, standard-normal initialised) in
    place of the previous particles and a zero previous action.  The chain of
    an initial row then starts where a recursive row starts (noise, the learned
    proposal, or - with a warm start - the prior particles themselves), so the
    initial belief lives in the same input domain and on the same scale as the
    recursive ones by construction; the anchor cannot be combined with it.
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
        langevin_noise_scale: float | None = None,
        langevin_noise_scale_learnable: bool = False,
        energy_network_kind: str = "mlp",
        kan_grid_size: int = 8,
        kan_spline_order: int = 3,
        kan_grid_range: float = 3.0,
        kan_match_parameters: bool = True,
        transition_proposal: bool = False,
        observation_anchor: bool = False,
        proposal_hidden: Sequence[int] = (64,),
        langevin_proposal: str = "noise",
        langevin_uniform_half_width: float = 3.0 ** 0.5,
        langevin_initial_noise: str = "normal",
        observation_anchor_steps: str = "initial",
        num_particles: int | None = None,
        particle_sampler: str = "ula",
        smi_component_scale: float = 0.1,
        score_parameterization: str = "energy",
        direct_score_bound: float | None = 10.0,
        initial_score_prior: float = 1.0,
        score_implementation: str = "autograd",
        belief_compile: str = "off",
        belief_cuda_graphs: bool = False,
        initial_belief: str = "branch",
    ) -> None:
        super().__init__()
        if initial_belief not in INITIAL_BELIEFS:
            raise ValueError("initial_belief must be branch or prior")
        self.initial_belief = str(initial_belief)
        if self.initial_belief == "prior":
            if observation_anchor:
                raise ValueError(
                    "initial_belief=prior scores the initial step with the recursive networks; "
                    "the observation anchor of the initial branch does not apply"
                )
            if num_particles is None or int(num_particles) <= 0:
                raise ValueError("initial_belief=prior needs the particle count (num_particles) at construction")
        if score_implementation not in {"autograd", "analytic"}:
            raise ValueError("score_implementation must be autograd or analytic")
        if score_implementation == "analytic" and energy_network_kind != "mlp":
            raise ValueError("analytic score requires energy_network_kind=mlp; use autograd for kan")
        self.score_implementation = str(score_implementation)
        if belief_compile not in {"off", "inductor"}:
            raise ValueError("belief_compile must be off or inductor")
        if belief_compile == "inductor" and score_implementation != "analytic":
            raise ValueError("belief_compile=inductor requires score_implementation=analytic")
        validate_belief_acceleration(belief_compile, belief_cuda_graphs)
        self.belief_compile = belief_compile
        self.belief_cuda_graphs = belief_cuda_graphs
        self._compiled_chains: dict[tuple, object] = {}
        self._compile_unavailable = False
        self._cuda_graph_chains: dict[tuple, object] = {}
        self._cuda_graph_cpu_warned = False
        if score_parameterization not in SCORE_PARAMETERIZATIONS:
            raise ValueError("score_parameterization must be energy, direct, direct_joint or direct_flat")
        # ``direct_flat``: no structure at all - one MLP maps the concatenation
        # (o, x, a, x_{t-1,1}, ..., x_{t-1,K}) of the query with every previous particle in its
        # stored order to the score (no attention, no permutation invariance, no pairwise terms).
        # The input width fixes the particle count K at construction.
        # ``direct``: the three networks output the score itself (no autograd through a scalar
        # potential): g(o, x) - x at t = 0, and f(o, x, a) + sum_j softmax_j(k_j) v_j at t >= 1 with
        # (k_j, v_j) = transition(x, x_{t-1,j}, a), the attention analogue of the gradient of the
        # log-mean-exp transition energy.  The field is not constrained to be conservative.
        # ``direct_joint`` does not add the two terms: the transition network returns attention
        # logits and context features (k_j, h_j), c(x) = sum_j softmax_j(k_j) h_j summarises the
        # previous belief around x, and one network maps (o, x, a, c(x)) to the score.
        self.score_parameterization = str(score_parameterization)
        # A SiLU energy network has a bounded gradient; a network that outputs the score grows
        # linearly in |x| and makes the transport expand without limit, so its outputs are bounded
        # smoothly by B * tanh(raw / B) (the -x restoring term of the t = 0 base stays unbounded).
        # ``None`` disables the bound (raw network outputs are the score); ``"log"`` applies the
        # slowly growing, unbounded squashing x / log(e + x^2) (identity slope at 0, ~ x / (2 ln|x|)
        # for large |x|) instead of the tanh bound.
        if isinstance(direct_score_bound, str):
            if direct_score_bound != "log":
                raise ValueError("direct_score_bound must be a positive number, None, or 'log'")
            self.direct_score_bound: float | str | None = "log"
        elif direct_score_bound is None:
            self.direct_score_bound = None
        else:
            if not float(direct_score_bound) > 0.0:
                raise ValueError("direct_score_bound must be positive or None")
            self.direct_score_bound = float(direct_score_bound)
        # Weight of the standard-normal prior term of the initial (t = 0) potential: the direct
        # score is g(o, x) - p * x and the energy potential is g(o, x) - p * |x|^2 / 2.  Every
        # earlier run used p = 1; p = 0 removes the restoring term.
        if not float(initial_score_prior) >= 0.0:
            raise ValueError("initial_score_prior must be non-negative")
        self.initial_score_prior = float(initial_score_prior)
        if particle_sampler not in PARTICLE_SAMPLERS:
            raise ValueError("particle_sampler must be ula, svgd or smi")
        self.particle_sampler = str(particle_sampler)
        if self.particle_sampler != "ula" and (
            langevin_noise_scale is not None or langevin_noise_scale_learnable or langevin_temperature_learnable
        ):
            raise ValueError("svgd and smi are deterministic transports; the ULA noise settings do not apply")
        if not float(smi_component_scale) > 0.0:
            raise ValueError("smi_component_scale must be positive")
        self.smi_component_scale = float(smi_component_scale)
        # SMI particles are Gaussian components (theta_k, log s_k); the encoder reads the
        # per-dimension log scale next to the score, so its per-particle feature is twice as wide.
        self.score_feature_dim = (2 if self.particle_sampler == "smi" else 1) * state_dim
        if energy_network_kind not in NETWORK_KINDS:
            raise ValueError("energy_network_kind must be mlp or kan")
        self.energy_network_kind = str(energy_network_kind)
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
        if langevin_noise_scale is not None and not float(langevin_noise_scale) > 0.0:
            raise ValueError("langevin_noise_scale must be positive")
        self.langevin_noise_scale = (
            None if langevin_noise_scale is None else float(langevin_noise_scale)
        )
        if langevin_noise_scale_learnable:
            if self.langevin_noise_scale is None:
                raise ValueError(
                    "a learnable noise-scale schedule needs a positive langevin_noise_scale"
                )
            if langevin_temperature_learnable:
                raise ValueError(
                    "langevin_noise_scale_learnable and langevin_temperature_learnable "
                    "both parameterise the chain noise; choose one"
                )
            if langevin_steps is None or int(langevin_steps) <= 0:
                raise ValueError(
                    "a learnable noise-scale schedule needs a positive langevin_steps"
                )
            self.langevin_log_noise_scale: nn.Parameter | None = nn.Parameter(
                torch.full(
                    (int(langevin_steps),),
                    _unbound_log_schedule(
                        math.log(self.langevin_noise_scale),
                        _LOG_NOISE_SCALE_MIN,
                        _LOG_NOISE_SCALE_MAX,
                        self.langevin_schedule_bound,
                    ),
                )
            )
        else:
            self.langevin_log_noise_scale = None
        self.kan_match_parameters = bool(kan_match_parameters)
        self.energy_hidden_dims: dict[str, list[int]] = {}

        def energy_network(name: str, input_dim: int, output_dim: int = 1) -> nn.Module:
            widths = list(hidden_dims)
            if self.energy_network_kind == "kan" and self.kan_match_parameters:
                # Same depth and (near-)same parameter budget as the MLP the
                # configuration describes; only the edge function class differs.
                widths = matched_kan_hidden_dims(
                    input_dim,
                    hidden_dims,
                    output_dim,
                    int(kan_grid_size) + int(kan_spline_order),
                )
            self.energy_hidden_dims[name] = widths
            return make_network(
                self.energy_network_kind,
                input_dim,
                widths,
                output_dim,
                kan_grid_size=int(kan_grid_size),
                kan_spline_order=int(kan_spline_order),
                kan_grid_range=float(kan_grid_range),
            )

        if transition_proposal and not self.langevin_warm_start:
            raise ValueError("transition_proposal requires langevin_warm_start")
        self.transition_proposal: nn.Module | None = (
            _zero_final_mlp(state_dim + action_feature_dim, proposal_hidden, state_dim)
            if transition_proposal
            else None
        )
        self.observation_anchor: nn.Module | None = (
            _zero_final_mlp(observation_dim, proposal_hidden, 2 * state_dim)
            if observation_anchor
            else None
        )
        # Learned chain start for the steps that would otherwise begin at standard-normal
        # noise (every recursive step, and an initial step without the anchor):
        #   "affine"  : xi -> mu + exp(log sigma) * xi with one global (mu, log sigma),
        #   "network" : per particle, (mu_k, log sigma_k) = h(x^{k}_{t-1}, a_{t-1}, o_t), zero-initialised,
        #   "uniform" : xi -> (2 Phi(xi) - 1) * w, i.e. U(-w, w) per coordinate (fixed, no parameters;
        #               w = sqrt(3) keeps the unit variance of the standard-normal start).
        # Affine / network start bit-identical to the plain sampler; a warm start already fixes the
        # chain start at the previous particles, so it is exclusive with every proposal.
        if langevin_proposal not in {"noise", "affine", "network", "uniform"}:
            raise ValueError("langevin_proposal must be noise, affine, network, or uniform")
        if langevin_proposal != "noise" and self.langevin_warm_start:
            raise ValueError("langevin_proposal affine/network/uniform cannot be combined with langevin_warm_start")
        self.langevin_proposal = str(langevin_proposal)
        if float(langevin_uniform_half_width) <= 0.0:
            raise ValueError("langevin_uniform_half_width must be positive")
        self.langevin_uniform_half_width = float(langevin_uniform_half_width)
        # Base noise of the t = 0 rows (before the observation anchor): "normal" = N(0, I) as in
        # every earlier run, "uniform" = U(-w, w) per coordinate with the same half width w.  It is
        # independent of langevin_proposal (which only shapes the recursive rows when the anchor
        # is on) and of the warm start (whose t = 0 rows still begin at noise).
        if langevin_initial_noise not in {"normal", "uniform"}:
            raise ValueError("langevin_initial_noise must be normal or uniform")
        self.langevin_initial_noise = str(langevin_initial_noise)
        # Which rows the observation anchor shifts and scales: "initial" = only t = 0 (every earlier
        # run), "all" = every step, i.e. the chain of step t starts at mu(o_t) + sigma(o_t) * noise.
        # The recursive rows keep their own base noise (N(0, I), or U(-w, w) with the uniform
        # proposal); a learned proposal or a warm start already owns those rows, so they are exclusive.
        if observation_anchor_steps not in {"initial", "all"}:
            raise ValueError("observation_anchor_steps must be initial or all")
        if observation_anchor_steps == "all":
            if not observation_anchor:
                raise ValueError("observation_anchor_steps=all needs the observation anchor")
            if langevin_proposal not in {"noise", "uniform"} or self.langevin_warm_start:
                raise ValueError(
                    "observation_anchor_steps=all cannot be combined with an affine / network proposal or a warm start"
                )
        self.observation_anchor_steps = str(observation_anchor_steps)
        self.proposal_mean: nn.Parameter | None = None
        self.proposal_log_scale: nn.Parameter | None = None
        self.proposal_network: nn.Module | None = None
        if langevin_proposal == "affine":
            self.proposal_mean = nn.Parameter(torch.zeros(state_dim))
            self.proposal_log_scale = nn.Parameter(torch.zeros(state_dim))
        elif langevin_proposal == "network":
            self.proposal_network = _zero_final_mlp(
                state_dim + action_feature_dim + observation_dim, proposal_hidden, 2 * state_dim
            )
        direct = self.score_parameterization != "energy"
        joint = self.score_parameterization == "direct_joint"
        flat = self.score_parameterization == "direct_flat"
        self.initial_energy: nn.Module | None
        self.prior_particles: nn.Parameter | None
        if self.initial_belief == "prior":
            # The initial step is a recursive step whose "previous belief" is this learned cloud.
            self.initial_energy = None
            self.prior_particles = nn.Parameter(torch.randn(int(num_particles), state_dim))
        else:
            self.initial_energy = energy_network(
                "initial", observation_dim + state_dim, state_dim if direct else 1
            )
            self.prior_particles = None
        self.transition_energy: nn.Module | None
        if flat:
            if num_particles is None or int(num_particles) <= 0:
                raise ValueError("direct_flat needs the particle count (num_particles) at construction")
            self.num_particles = int(num_particles)
            self.observation_energy = energy_network(
                "observation",
                observation_dim + state_dim + action_feature_dim + self.num_particles * state_dim,
                state_dim,
            )
            self.transition_energy = None
        else:
            self.num_particles = None if num_particles is None else int(num_particles)
            self.observation_energy = energy_network(
                "observation",
                observation_dim + state_dim + action_feature_dim + (_JOINT_CONTEXT_DIM if joint else 0),
                state_dim if direct else 1,
            )
            self.transition_energy = energy_network(
                "transition",
                2 * state_dim + action_feature_dim,
                1 + (_JOINT_CONTEXT_DIM if joint else state_dim) if direct else 1,
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

    def _resolve_context(
        self,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Substitute the learned prior for the previous belief of initial rows (prior mode).

        Returns the context unchanged in branch mode.  In prior mode every initial row reads
        the prior particles and a zero previous action, and the returned ``initial`` flags are
        all false: the row is a recursive step from then on (branch classification, chain
        start, warm start, and transition proposal included).
        """

        if self.prior_particles is None:
            return previous_particles, previous_action, initial
        if previous_particles.shape[1] != self.prior_particles.shape[0]:
            raise ValueError(
                f"initial_belief=prior holds {self.prior_particles.shape[0]} prior particles, "
                f"got {previous_particles.shape[1]} previous particles"
            )
        mask = initial.bool()
        prior = self.prior_particles.to(previous_particles.dtype)[None, :, :].expand_as(previous_particles)
        previous_particles = torch.where(mask[:, None, None], prior, previous_particles)
        previous_action = torch.where(mask[:, None], torch.zeros_like(previous_action), previous_action)
        return previous_particles, previous_action, torch.zeros_like(mask)

    def classify_potential_branch(self, initial: torch.Tensor) -> PotentialBranch:
        """Classify one vectorised environment step with one device synchronisation.

        ULA evaluates the score repeatedly for the same environment context.  The
        caller can classify that context once outside the Langevin loop (and
        outside activation-checkpoint recomputation), then reuse the returned
        Python value for every score evaluation.  In prior mode every step is
        recursive and no synchronisation is needed.
        """

        if initial.ndim != 1:
            raise ValueError("initial must have shape [batch]")
        if self.prior_particles is not None:
            return "recursive"
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
        if self.prior_particles is not None:
            return ("recursive",) * int(initial.shape[0])
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
        return initial_value - 0.5 * self.initial_score_prior * query.square().sum(dim=-1)

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

    def _direct_score_for_branch(
        self,
        query: torch.Tensor,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
        potential_branch: PotentialBranch,
    ) -> torch.Tensor:
        """The directly parameterised score field (``score_parameterization="direct"``)."""

        num_query = query.shape[1]
        bound = self.direct_score_bound
        prior = self.initial_score_prior

        def bounded(raw: torch.Tensor) -> torch.Tensor:
            if bound is None:
                return raw
            if bound == "log":
                return raw / torch.log(math.e + raw.square())
            return bound * torch.tanh(raw / bound)

        def initial_score() -> torch.Tensor:
            obs = observation[:, None, :].expand(-1, num_query, -1)
            return bounded(self.initial_energy(torch.cat((obs, query), dim=-1))) - prior * query

        def recursive_score() -> torch.Tensor:
            num_previous = previous_particles.shape[1]
            obs = observation[:, None, :].expand(-1, num_query, -1)
            action = previous_action[:, None, :].expand(-1, num_query, -1)
            if self.score_parameterization == "direct_flat":
                if num_previous != self.num_particles:
                    raise ValueError(
                        f"direct_flat was built for {self.num_particles} previous particles, got {num_previous}"
                    )
                flat_previous = previous_particles.reshape(previous_particles.shape[0], 1, -1).expand(
                    -1, num_query, -1
                )
                return bounded(self.observation_energy(torch.cat((obs, query, action, flat_previous), dim=-1)))
            pairs = torch.cat(
                (
                    query[:, :, None, :].expand(-1, -1, num_previous, -1),
                    previous_particles[:, None, :, :].expand(-1, num_query, -1, -1),
                    previous_action[:, None, None, :].expand(-1, num_query, num_previous, -1),
                ),
                dim=-1,
            )
            transition = self.transition_energy(pairs)
            weights = torch.softmax(transition[..., 0], dim=-1)
            if self.score_parameterization == "direct_joint":
                context = (weights[..., None] * transition[..., 1:]).sum(dim=2)
                return bounded(self.observation_energy(torch.cat((obs, query, action, context), dim=-1)))
            field = bounded(self.observation_energy(torch.cat((obs, query, action), dim=-1)))
            return field + (weights[..., None] * bounded(transition[..., 1:])).sum(dim=2)

        if potential_branch == "initial":
            return initial_score()
        if potential_branch == "recursive":
            return recursive_score()
        return torch.where(initial[:, None, None].bool(), initial_score(), recursive_score())

    def log_potential(
        self,
        query: torch.Tensor,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the unnormalised scalar potential at all query particles."""

        if self.score_parameterization != "energy":
            raise ValueError("a directly parameterised score field has no scalar potential")

        self._validate_context(query, observation, previous_particles, previous_action, initial)
        previous_particles, previous_action, initial = self._resolve_context(
            previous_particles, previous_action, initial
        )
        potential_branch = self.classify_potential_branch(initial)
        return self._log_potential_for_branch(
            query,
            observation,
            previous_particles,
            previous_action,
            initial,
            potential_branch,
        )

    def _analytic_score_for_branch(
        self,
        query: torch.Tensor,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
        potential_branch: PotentialBranch,
    ) -> torch.Tensor:
        num_query = query.shape[1]
        obs = observation[:, None, :].expand(-1, num_query, -1)

        def initial_score() -> torch.Tensor:
            _, gradient = _mlp_value_and_input_gradient(
                self.initial_energy, torch.cat((obs, query), dim=-1)
            )
            return gradient[..., self.observation_dim:] - self.initial_score_prior * query

        def recursive_score() -> torch.Tensor:
            action = previous_action[:, None, :].expand(-1, num_query, -1)
            _, observation_gradient = _mlp_value_and_input_gradient(
                self.observation_energy, torch.cat((obs, query, action), dim=-1)
            )
            num_previous = previous_particles.shape[1]
            pairs = torch.cat((
                query[:, :, None, :].expand(-1, -1, num_previous, -1),
                previous_particles[:, None, :, :].expand(-1, num_query, -1, -1),
                previous_action[:, None, None, :].expand(-1, num_query, num_previous, -1),
            ), dim=-1)
            value, transition_gradient = _mlp_value_and_input_gradient(self.transition_energy, pairs)
            weights = (value - value.logsumexp(dim=-1, keepdim=True)).exp()
            return (
                observation_gradient[..., self.observation_dim:self.observation_dim + self.state_dim]
                + (weights[..., None] * transition_gradient[..., :self.state_dim]).sum(dim=2)
            )

        if potential_branch == "initial":
            return initial_score()
        if potential_branch == "recursive":
            return recursive_score()
        return torch.where(initial[:, None, None].bool(), initial_score(), recursive_score())

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

        if self.score_parameterization != "energy":
            query = particles.detach() if detach_query else particles
            with torch.set_grad_enabled(create_graph or torch.is_grad_enabled()):
                score = self._direct_score_for_branch(
                    query, observation, previous_particles, previous_action, initial, potential_branch
                )
                if self.score_clip > 0:
                    score = score.clamp(-self.score_clip, self.score_clip)
            return score if create_graph else score.detach()
        if self.score_implementation == "analytic":
            query = particles.detach() if detach_query else particles
            with torch.set_grad_enabled(create_graph):
                score = self._analytic_score_for_branch(
                    query, observation, previous_particles, previous_action, initial, potential_branch
                )
                if self.score_clip > 0:
                    score = score.clamp(-self.score_clip, self.score_clip)
            return score if create_graph else score.detach()
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
        previous_particles, previous_action, initial = self._resolve_context(
            previous_particles, previous_action, initial
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

    def noise_scale_schedule(self) -> torch.Tensor | None:
        """Effective per-step ``beta_l`` (bounded, exponentiated) or ``None``."""

        if self.langevin_log_noise_scale is None:
            return None
        return _bound_log_schedule(
            self.langevin_log_noise_scale,
            _LOG_NOISE_SCALE_MIN,
            _LOG_NOISE_SCALE_MAX,
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
        need_scores: bool = True,
        stop_before_step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Replay one reparameterized ULA belief update from explicit noise.

        ``temporal_gradient_mode="full"`` retains the graph carried by the
        previous time step.  Mode ``"tbptt_1"`` detaches only that incoming
        belief state; all ULA steps *within* the current environment time step
        remain differentiable.  Environment observations and realised actions
        are treated as replay inputs by the caller rather than differentiated
        through an environment model.

        ``need_scores=False`` skips only the final readout and returns ``None``
        for scores. All transport scores (including SMI scale updates) remain.
        """

        # Also guard direct flag changes after construction, before compilation
        # or CUDA work can build a graph with missing parameter gradients.
        validate_belief_acceleration(self.belief_compile, self.belief_cuda_graphs)
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
        if stop_before_step is not None:
            if track_grad or not 0 <= stop_before_step < num_steps:
                raise ValueError("stop_before_step requires detached replay and a stage in 0..L-1")
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
        # Prior mode: the initial rows read the learned prior particles (gradient flows into
        # them through the substitution, also under tbptt_1) and become recursive rows.
        incoming_particles, previous_action, initial = self._resolve_context(
            incoming_particles, previous_action, initial
        )
        if self.prior_particles is not None:
            potential_branch = "recursive"
        elif potential_branch is None:
            potential_branch = self.classify_potential_branch(initial)
        elif potential_branch not in {"initial", "recursive", "mixed"}:
            raise ValueError("potential_branch must be initial, recursive, or mixed")

        if self.langevin_initial_noise == "uniform":
            # Phi(xi) is exactly U(0, 1) for standard-normal xi: the stored noise keeps its role,
            # so the PPO replay reproduces the same chain start.
            flat = (2.0 * torch.special.ndtr(particles) - 1.0) * self.langevin_uniform_half_width
            particles = torch.where(initial.view(-1, 1, 1), flat, particles)
        if self.observation_anchor is not None:
            anchor = self.observation_anchor(observation)
            anchor_mean, anchor_log_scale = anchor.chunk(2, dim=-1)
            anchor_log_scale = anchor_log_scale.clamp(
                _ANCHOR_LOG_SCALE_MIN, _ANCHOR_LOG_SCALE_MAX
            )
            anchored = (
                anchor_mean[:, None, :] + anchor_log_scale.exp()[:, None, :] * particles
            )
            particles = torch.where(initial.view(-1, 1, 1), anchored, particles)
        if self.langevin_proposal != "noise":
            noise = initial_noise.detach()
            if self.langevin_proposal == "uniform":
                # Phi(xi) is exactly U(0, 1) for standard-normal xi, so the replayed noise keeps its
                # role (the same stored noise reproduces the same chain start in the PPO replay).
                proposal_mean = torch.zeros((), dtype=noise.dtype, device=noise.device)
                proposal_log_scale = torch.zeros((), dtype=noise.dtype, device=noise.device)
                noise = (2.0 * torch.special.ndtr(noise) - 1.0) * self.langevin_uniform_half_width
            elif self.proposal_network is not None:
                num_particles = noise.shape[1]
                proposal_input = torch.cat(
                    (
                        incoming_particles,
                        previous_action[:, None, :].expand(-1, num_particles, -1),
                        observation[:, None, :].expand(-1, num_particles, -1),
                    ),
                    dim=-1,
                )
                proposal_mean, proposal_log_scale = self.proposal_network(proposal_input).chunk(2, dim=-1)
            else:
                assert self.proposal_mean is not None and self.proposal_log_scale is not None
                proposal_mean, proposal_log_scale = self.proposal_mean, self.proposal_log_scale
            proposal_log_scale = proposal_log_scale.clamp(_ANCHOR_LOG_SCALE_MIN, _ANCHOR_LOG_SCALE_MAX)
            proposed = proposal_mean + proposal_log_scale.exp() * noise
            if self.observation_anchor is not None:
                # the anchor keeps the initial rows; the learned proposal takes the recursive rows
                particles = torch.where(initial.view(-1, 1, 1), particles, proposed)
            else:
                particles = proposed
        if self.observation_anchor is not None and self.observation_anchor_steps == "all":
            # the recursive rows hold their base noise here; the same anchor moves them as well
            particles = torch.where(
                initial.view(-1, 1, 1),
                particles,
                anchor_mean[:, None, :] + anchor_log_scale.exp()[:, None, :] * particles,
            )
        if self.langevin_warm_start:
            if incoming_particles.shape[1] != particles.shape[1]:
                raise ValueError(
                    "langevin_warm_start needs matching previous and current particle counts"
                )
            warm = incoming_particles
            if self.transition_proposal is not None:
                action_tiled = previous_action[:, None, :].expand(
                    -1, incoming_particles.shape[1], -1
                )
                warm = warm + self.transition_proposal(
                    torch.cat((incoming_particles, action_tiled), dim=-1)
                )
            particles = torch.where(initial.view(-1, 1, 1), particles, warm)

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

        if self.langevin_log_noise_scale is None:
            step_betas = None
        else:
            if self.langevin_log_noise_scale.shape[0] != len(steps):
                raise ValueError(
                    "the learned noise-scale schedule length must equal the Langevin step count"
                )
            step_betas = self.noise_scale_schedule()

        chain_args = (particles, observation, incoming_particles, previous_action, initial,
                      increments, tuple(steps), step_alphas, step_temperatures, step_betas,
                      potential_branch, track_grad, need_scores)
        with torch.set_grad_enabled(track_grad):
            if stop_before_step is not None:
                # Auxiliary curl replay never enters compilation/CUDA Graph capture.
                # Keep the full schedule when replaying a prefix of the transport.
                return self._transport_chain(*chain_args, stop_before_step=stop_before_step)
            if self.belief_compile == "inductor" and not self._compile_unavailable:
                def tensor_signature(value: torch.Tensor | None) -> tuple | None:
                    if value is None:
                        return None
                    return (tuple(value.shape), tuple(value.stride()), value.dtype,
                            value.device, value.requires_grad)

                key = (potential_branch, need_scores, tuple(steps), track_grad,
                       self.training, torch.is_autocast_enabled(),
                       torch.is_autocast_cpu_enabled(),
                       *(tensor_signature(value) for value in chain_args[:6]),
                       *(tensor_signature(value) for value in chain_args[7:10]),
                       tuple(parameter.requires_grad for parameter in self.parameters()))
                chain = self._compiled_chains.get(key)
                first_call = chain is None
                if chain is None:
                    logging.getLogger(__name__).info("Compiling belief transport for %s", key)
                    chain = torch.compile(self._transport_chain, fullgraph=True, dynamic=False,
                                          options={"triton.cudagraphs": False})
                    self._compiled_chains[key] = chain
                try:
                    # Several branch/shape variants share the same Python code object.
                    with torch._dynamo.config.patch(cache_size_limit=128, accumulated_cache_size_limit=512):
                        if first_call:
                            # AOT traces a joint forward/backward on its first call.
                            # PyTorch 2.3.1 otherwise lets an enclosing activation
                            # checkpoint intercept those *tracing* tensors. Compile
                            # once with private hooks, then execute normally so the
                            # real saved tensors still belong to the checkpoint.
                            with torch.autograd.graph.saved_tensors_hooks(
                                lambda value: value.detach(), lambda value: value,
                            ):
                                chain(*chain_args)
                        return self._execute_transport(chain, chain_args)
                except Exception as exc:
                    # A CPU installation may lack an inductor-compatible C++ compiler.
                    # Do not mask other failures, especially CUDA failures.
                    if observation.device.type != "cpu" or "InvalidCxxCompiler" not in str(type(exc)) + str(exc):
                        raise
                    warnings.warn("Inductor C++ compiler unavailable; using eager belief transport", RuntimeWarning)
                    self._compile_unavailable = True
            return self._execute_transport(self._transport_chain, chain_args)

    def _execute_transport(self, function, chain_args):
        if not self.belief_cuda_graphs:
            return function(*chain_args)
        device = chain_args[0].device
        if device.type != "cuda":
            if not self._cuda_graph_cpu_warned:
                warnings.warn("belief_cuda_graphs is a no-op on CPU", RuntimeWarning)
                self._cuda_graph_cpu_warned = True
            return function(*chain_args)
        if self.score_implementation != "analytic":
            raise ValueError("belief_cuda_graphs requires score_implementation=analytic on CUDA")
        from .cuda_graphs import CudaGraphStep

        tensor_indices = tuple(i for i, value in enumerate(chain_args) if isinstance(value, torch.Tensor))
        inputs = tuple(chain_args[i] for i in tensor_indices)
        parameters = tuple(self.parameters())
        key = (
            tuple((i, value) for i, value in enumerate(chain_args) if i not in tensor_indices),
            tuple((value.shape, value.stride(), value.dtype, value.device, value.requires_grad) for value in inputs),
            tuple((value.data_ptr(), value.requires_grad) for value in parameters),
            self.training, self.belief_compile, torch.is_autocast_enabled(),
            torch.get_autocast_gpu_dtype(),
            torch.cuda.current_stream(device).cuda_stream,
        )
        graph = self._cuda_graph_chains.get(key)
        if graph is None:
            # Do not retain the first call's live temporal graph in this closure.
            template = tuple(None if i in tensor_indices else value for i, value in enumerate(chain_args))

            def tensor_function(*values):
                arguments = list(template)
                for index, value in zip(tensor_indices, values):
                    arguments[index] = value
                particles, scores = function(*arguments)
                return (particles,) if scores is None else (particles, scores)

            logging.getLogger(__name__).info("Capturing belief CUDA transport for %s", key)
            # Capture's one-time warmup must not become part of an enclosing
            # activation checkpoint's saved-tensor accounting. Runtime nodes
            # below still use the caller's normal hooks and version checks.
            with torch.cuda.device(device), torch.autograd.graph.saved_tensors_hooks(
                lambda value: value.detach(), lambda value: value,
            ):
                graph = CudaGraphStep(tensor_function, inputs, self, track_grad=chain_args[-2])
            self._cuda_graph_chains[key] = graph
        outputs = graph(*inputs)
        return outputs[0], outputs[1] if chain_args[-1] else None

    def _transport_chain(
        self, particles: torch.Tensor, observation: torch.Tensor,
        incoming_particles: torch.Tensor, previous_action: torch.Tensor, initial: torch.Tensor,
        increments: torch.Tensor, steps: tuple[float, ...], step_alphas: torch.Tensor | None,
        step_temperatures: torch.Tensor | None, step_betas: torch.Tensor | None,
        potential_branch: PotentialBranch, track_grad: bool, need_scores: bool,
        *, stop_before_step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        log_scales = (
            torch.full_like(particles, math.log(self.smi_component_scale))
            if self.particle_sampler == "smi"
            else None
        )
        for step_index, alpha in enumerate(steps):
            if step_index == stop_before_step:
                return particles.detach(), None
            alpha_value = alpha if step_alphas is None else step_alphas[step_index]
            if self.particle_sampler != "ula":
                particles, log_scales = self._stein_step(
                    particles,
                    log_scales,
                    increments[:, step_index],
                    alpha_value,
                    observation,
                    incoming_particles,
                    previous_action,
                    initial,
                    potential_branch,
                    track_grad=track_grad,
                )
                continue
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
            if step_betas is not None:
                noise_scale = step_betas[step_index]
            elif self.langevin_noise_scale is not None:
                noise_scale = self.langevin_noise_scale
            elif step_alphas is None and step_temperatures is None:
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

        scores = None
        if need_scores:
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
            if log_scales is not None:
                scores = torch.cat((scores, log_scales), dim=-1)
        if track_grad:
            return particles, scores
        return particles.detach(), None if scores is None else scores.detach()

    def _stein_step(
        self,
        particles: torch.Tensor,
        log_scales: torch.Tensor | None,
        noise: torch.Tensor,
        alpha: float | torch.Tensor,
        observation: torch.Tensor,
        previous_particles: torch.Tensor,
        previous_action: torch.Tensor,
        initial: torch.Tensor,
        potential_branch: PotentialBranch,
        *,
        track_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One SVGD or SMI transport step ``x <- x + alpha * phi(x)`` of the K particles.

        SVGD drives the kernelised update with the learned score at the particles and uses no
        noise.  SMI (Stein mixture inference) treats particle k as the Gaussian component
        ``N(theta_k, diag(s_k^2))`` and transports ``(theta_k, log s_k)`` jointly: the drive is the
        reparameterised gradient of the mixture ELBO ``E_{q_k}[log p(z) - log q(z)]`` with
        ``z_k = theta_k + s_k * eps_k`` and ``eps`` the saved step noise, so replay stays exact,
        and the kernel acts on the concatenated component parameters.
        """

        def score(query: torch.Tensor) -> torch.Tensor:
            return self._score_at_for_branch(
                query,
                observation,
                previous_particles,
                previous_action,
                initial,
                potential_branch,
                create_graph=track_grad,
                detach_query=not track_grad,
            )

        if self.particle_sampler == "svgd":
            updated = particles + alpha * _rbf_stein_direction(particles, score(particles))
            new_log_scales = None
        else:
            assert log_scales is not None
            scale = log_scales.exp()
            sample_score = score(particles + scale * noise)
            entropy_gradient = (
                _analytic_mixture_entropy_gradient
                if self.score_implementation == "analytic"
                else _mixture_entropy_gradient
            )
            entropy_theta, entropy_log_scale = entropy_gradient(
                particles, log_scales, noise, create_graph=track_grad
            )
            drive = torch.cat(
                (sample_score + entropy_theta, sample_score * noise * scale + entropy_log_scale), dim=-1
            )
            parameters = torch.cat((particles, log_scales), dim=-1)
            moved = parameters + alpha * _rbf_stein_direction(parameters, drive)
            updated, new_log_scales = moved.split(self.state_dim, dim=-1)
            new_log_scales = new_log_scales.clamp(_LOG_SMI_SCALE_MIN, _LOG_SMI_SCALE_MAX)
        if self.particle_clip > 0:
            updated = updated.clamp(-self.particle_clip, self.particle_clip)
        if not track_grad:
            updated = updated.detach()
            new_log_scales = None if new_log_scales is None else new_log_scales.detach()
        return updated, new_log_scales

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
