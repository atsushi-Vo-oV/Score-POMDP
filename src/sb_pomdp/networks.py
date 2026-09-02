"""Neural building blocks used by the score-belief actor--critic."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional


def make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    """Build a small MLP with orthogonal-friendly, smooth activations."""

    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("MLP input and output dimensions must be positive")
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for index, (in_features, out_features) in enumerate(pairwise(dims)):
        layer = nn.Linear(in_features, out_features)
        gain = 1.0 if index == len(dims) - 2 else 2**0.5
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if index != len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class BeliefSetEncoder(nn.Module):
    """Encode an unordered set of ``(particle, score)`` pairs.

    No positional embedding is used and the final reduction is a mean, so the
    returned representation is invariant to a permutation of particle tokens.
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.state_dim = state_dim
        self.token_projection = nn.Sequential(
            nn.Linear(2 * state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, particles: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if particles.shape != scores.shape:
            raise ValueError("particles and scores must have identical shapes")
        if particles.ndim != 3 or particles.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected [batch, particles, {self.state_dim}], got {tuple(particles.shape)}"
            )
        tokens = self.token_projection(torch.cat((particles, scores), dim=-1))
        encoded = self.transformer(tokens)
        return self.output_norm(encoded.mean(dim=1))


class DeepSetsBeliefEncoder(nn.Module):
    """Encode particle-score pairs with a Deep Sets mean-pooling network.

    Each pair is processed independently by an element network, the resulting
    features are averaged, and a pooling network maps the pooled feature to the
    common actor--critic condition dimension.  The mean (rather than a sum)
    keeps the feature scale stable when the particle-count ablation changes.
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or d_model <= 0 or feedforward_dim <= 0:
            raise ValueError("Deep Sets dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.state_dim = state_dim
        self.element_encoder = nn.Sequential(
            nn.Linear(2 * state_dim, feedforward_dim),
            nn.LayerNorm(feedforward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.SiLU(),
        )
        self.pooled_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, feedforward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, particles: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if particles.shape != scores.shape:
            raise ValueError("particles and scores must have identical shapes")
        if particles.ndim != 3 or particles.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected [batch, particles, {self.state_dim}], got {tuple(particles.shape)}"
            )
        tokens = self.element_encoder(torch.cat((particles, scores), dim=-1))
        return self.pooled_encoder(tokens.mean(dim=1))


class AlphaPoolBeliefEncoder(nn.Module):
    """Mean-pool per-particle trunk features with a strictly linear tail.

    Each pooled output coordinate is the Monte Carlo estimate of a linear
    functional ``<psi_j, b>`` of the belief, so any downstream *linear* map of
    the output is itself a belief-linear functional (the alpha-vector reading).
    No post-pooling network is applied on purpose: convexity in the belief may
    only be introduced by an explicit max/log-sum-exp head.  ``use_scores``
    optionally appends the score vectors to the trunk input; the default keeps
    particle positions only, which removes the observation-feature shortcut
    channel through the score field.
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int,
        feedforward_dim: int,
        *,
        use_scores: bool = False,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or d_model <= 0 or feedforward_dim <= 0:
            raise ValueError("alpha-pool dimensions must be positive")
        self.state_dim = state_dim
        self.use_scores = bool(use_scores)
        input_dim = (2 if self.use_scores else 1) * state_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, feedforward_dim),
            nn.LayerNorm(feedforward_dim),
            nn.SiLU(),
            nn.Linear(feedforward_dim, d_model),
            nn.SiLU(),
        )

    def forward(self, particles: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if particles.shape != scores.shape:
            raise ValueError("particles and scores must have identical shapes")
        if particles.ndim != 3 or particles.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected [batch, particles, {self.state_dim}], got {tuple(particles.shape)}"
            )
        inputs = (
            torch.cat((particles, scores), dim=-1) if self.use_scores else particles
        )
        return self.trunk(inputs).mean(dim=1)


class KANLinear(nn.Module):
    """Kolmogorov-Arnold layer: a learnable B-spline function on every edge.

    Efficient-KAN formulation (Liu et al. 2024, "KAN: Kolmogorov-Arnold
    Networks"): each edge ``i -> j`` carries ``phi_ij(x) = w_b * silu(x) +
    w_s * sum_m c_ijm B_m(x)`` with ``B_m`` cubic B-spline bases on a fixed
    grid over ``[-grid_range, grid_range]``; the unit output is the sum over
    incoming edges.  Outside the grid only the smooth ``silu`` base term is
    active, so the layer degrades gracefully instead of extrapolating splines.
    The map is piecewise polynomial in its input, hence its input gradient
    (needed for ULA scores) and second derivative (needed for the reparameterised
    ``full`` belief gradient) are both well defined and computed by autograd.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: float = 3.0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("KAN layer dimensions must be positive")
        if grid_size <= 0 or spline_order <= 0:
            raise ValueError("KAN grid_size and spline_order must be positive")
        if not grid_range > 0.0:
            raise ValueError("KAN grid_range must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        spacing = 2.0 * float(grid_range) / self.grid_size
        knots = (
            torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1)
            * spacing
            - float(grid_range)
        )
        self.register_buffer("grid", knots)
        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, self.grid_size + self.spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.ones(self.out_features, self.in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.spline_weight, std=0.1 / (self.in_features**0.5))

    @property
    def num_bases(self) -> int:
        return self.grid_size + self.spline_order

    def b_splines(self, inputs: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor bases: ``[batch, in_features, grid_size + spline_order]``."""

        if inputs.ndim != 2 or inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected inputs [batch, {self.in_features}], got {tuple(inputs.shape)}"
            )
        grid = self.grid.to(inputs.dtype)
        x = inputs[:, :, None]
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(inputs.dtype)
        for order in range(1, self.spline_order + 1):
            left = (x - grid[: -(order + 1)]) / (grid[order:-1] - grid[: -(order + 1)])
            right = (grid[order + 1 :] - x) / (grid[order + 1 :] - grid[1:-order])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        leading = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features)
        base = functional.linear(functional.silu(flat), self.base_weight)
        bases = self.b_splines(flat).reshape(flat.shape[0], -1)
        scaled = (self.spline_weight * self.spline_scaler[:, :, None]).reshape(
            self.out_features, -1
        )
        spline = functional.linear(bases, scaled)
        return (base + spline).reshape(*leading, self.out_features)


def make_kan(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    grid_size: int = 8,
    spline_order: int = 3,
    grid_range: float = 3.0,
) -> nn.Sequential:
    """Stack KAN layers; the spline edges are the nonlinearity, so no activations."""

    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("KAN input and output dimensions must be positive")
    dims = [input_dim, *hidden_dims, output_dim]
    return nn.Sequential(
        *(
            KANLinear(
                in_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
            for in_features, out_features in pairwise(dims)
        )
    )


NETWORK_KINDS = ("mlp", "kan")


def count_mlp_parameters(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> int:
    dims = [input_dim, *hidden_dims, output_dim]
    return sum(i * o + o for i, o in pairwise(dims))


def count_kan_parameters(
    input_dim: int, hidden_dims: Sequence[int], output_dim: int, num_bases: int
) -> int:
    """Spline coefficients + base weight + spline scaler per edge."""

    dims = [input_dim, *hidden_dims, output_dim]
    return sum(i * o * (num_bases + 2) for i, o in pairwise(dims))


def matched_kan_hidden_dims(
    input_dim: int,
    mlp_hidden_dims: Sequence[int],
    output_dim: int,
    num_bases: int,
) -> list[int]:
    """Uniform KAN widths (same depth) whose parameter count is closest to the MLP's.

    A KAN edge carries ``num_bases + 2`` parameters where an MLP edge carries
    one, so a KAN of the same widths is an order of magnitude larger.  This
    keeps the ablation honest by default: same depth, same budget, only the
    edge function class differs.
    """

    if not mlp_hidden_dims:
        return []
    target = count_mlp_parameters(input_dim, mlp_hidden_dims, output_dim)
    depth = len(mlp_hidden_dims)
    best_width, best_gap = 1, None
    for width in range(1, max(mlp_hidden_dims) + 1):
        gap = abs(count_kan_parameters(input_dim, [width] * depth, output_dim, num_bases) - target)
        if best_gap is None or gap < best_gap:
            best_width, best_gap = width, gap
        elif count_kan_parameters(input_dim, [width] * depth, output_dim, num_bases) > target:
            break
    return [best_width] * depth


def make_network(
    kind: str,
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    kan_grid_size: int = 8,
    kan_spline_order: int = 3,
    kan_grid_range: float = 3.0,
) -> nn.Sequential:
    """Build an MLP or a KAN of the same layer widths."""

    if kind == "mlp":
        return make_mlp(input_dim, hidden_dims, output_dim)
    if kind == "kan":
        return make_kan(
            input_dim,
            hidden_dims,
            output_dim,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            grid_range=kan_grid_range,
        )
    raise ValueError(f"unsupported network kind: {kind!r} (expected mlp or kan)")


class ObservationPredictor(nn.Module):
    """Heteroscedastic next-observation density from particles and the action.

    Predictive-sufficiency auxiliary supervision for the belief: maximising
    ``log (1/K) sum_k N(o_{t+1}; mu(s_k, a_t), sigma(s_k, a_t))`` forces the
    particle set to carry exactly the information needed to predict future
    observations - a sufficient statistic in the predictive-state sense -
    using nothing but the agent's own observation/action stream.  The
    predictor reads particle positions only (never score vectors), so the
    observation-feature shortcut through the encoder cannot satisfy this
    loss; the gradient lands on the belief chain.  The head is auxiliary:
    policy and value never consume its output.
    """

    _LOG_STD_MIN = -4.0
    _LOG_STD_MAX = 3.0

    def __init__(
        self,
        state_dim: int,
        action_feature_dim: int,
        observation_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_feature_dim < 0 or observation_dim <= 0:
            raise ValueError("observation predictor dimensions must be positive")
        self.state_dim = state_dim
        self.action_feature_dim = action_feature_dim
        self.observation_dim = observation_dim
        self.network = make_mlp(
            state_dim + action_feature_dim,
            hidden_dims,
            2 * observation_dim,
        )

    def log_likelihood(
        self,
        particles: torch.Tensor,
        action_features: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture log-likelihood ``log (1/K) sum_k p(o' | s_k, a)`` per batch row."""

        if particles.ndim != 3 or particles.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected particles [batch, K, {self.state_dim}], got {tuple(particles.shape)}"
            )
        batch, num_particles = particles.shape[0], particles.shape[1]
        if action_features.shape != (batch, self.action_feature_dim):
            raise ValueError("action features must have shape [batch, action_feature_dim]")
        if next_observation.shape != (batch, self.observation_dim):
            raise ValueError("next observation must have shape [batch, observation_dim]")
        expanded_actions = action_features[:, None, :].expand(
            batch, num_particles, self.action_feature_dim
        )
        outputs = self.network(torch.cat((particles, expanded_actions), dim=-1))
        mean, log_std = outputs.chunk(2, dim=-1)
        log_std = log_std.clamp(self._LOG_STD_MIN, self._LOG_STD_MAX)
        target = next_observation[:, None, :]
        per_dim = (
            -0.5 * ((target - mean) / log_std.exp()).square()
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        )
        per_particle = per_dim.sum(dim=-1)
        return torch.logsumexp(per_particle, dim=1) - math.log(num_particles)


class AlphaLSEHead(nn.Module):
    """Tempered log-sum-exp over linear pieces: a learned PWLC belief functional.

    Fed with mean-pooled trunk features ``Psi(b)``, every piece
    ``W[o, m] . Psi(b) + c[o, m]`` is a linear functional of the belief and the
    tempered log-sum-exp over the ``m`` axis is its smooth maximum, so each
    output realises ``max_m <alpha_{o,m}, b>`` in the alpha-vector sense.
    ``temperature -> 0`` recovers the hard maximum; the smooth version keeps a
    gradient on every piece, avoiding the dead-piece failure of hard max.
    """

    def __init__(
        self,
        condition_dim: int,
        outputs: int,
        pieces: int,
        temperature: float,
        *,
        init_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if condition_dim <= 0 or outputs <= 0 or pieces <= 0:
            raise ValueError("alpha head dimensions must be positive")
        if not float(temperature) > 0.0:
            raise ValueError("alpha head temperature must be positive")
        self.outputs = int(outputs)
        self.pieces = int(pieces)
        self.temperature = float(temperature)
        self.linear = nn.Linear(condition_dim, self.outputs * self.pieces)
        nn.init.orthogonal_(self.linear.weight, gain=init_gain)
        nn.init.zeros_(self.linear.bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        values = self.linear(condition)
        values = values.view(*values.shape[:-1], self.outputs, self.pieces)
        return self.temperature * torch.logsumexp(
            values / self.temperature, dim=-1
        )
