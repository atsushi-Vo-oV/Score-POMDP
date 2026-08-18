"""Neural building blocks used by the score-belief actor--critic."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn


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
