"""A non-finite gradient skips the optimizer step instead of killing the run.

Three production arms (core4 Light-Dark PF seed 13, pendbal v1lf alpha seed 11, pendbal
v4lf alpha seed 12) died with ``clip_grad_norm_(error_if_nonfinite=True)`` raising on a
single NaN minibatch.  ``clip_gradients_or_skip`` clears such gradients and reports
``finite=False`` so the callers skip the step.
"""

from __future__ import annotations

import math

import torch

from sb_pomdp.ppo import clip_gradients_or_skip


def _layer_with_gradients(weight_value: float, bias_value: float) -> torch.nn.Linear:
    layer = torch.nn.Linear(3, 1)
    layer.weight.grad = torch.full_like(layer.weight, weight_value)
    layer.bias.grad = torch.full_like(layer.bias, bias_value)
    return layer


def test_finite_gradients_are_clipped_and_kept() -> None:
    layer = _layer_with_gradients(10.0, 0.0)
    total_norm, finite = clip_gradients_or_skip(layer.parameters(), 1.0)
    assert finite
    assert math.isclose(float(total_norm), 10.0 * math.sqrt(3.0), rel_tol=1e-5)
    assert layer.weight.grad is not None and layer.bias.grad is not None
    clipped = torch.sqrt(layer.weight.grad.square().sum() + layer.bias.grad.square().sum())
    assert math.isclose(float(clipped), 1.0, rel_tol=1e-4)


def test_small_finite_gradients_are_untouched() -> None:
    layer = _layer_with_gradients(0.1, 0.1)
    before = layer.weight.grad.clone()
    _, finite = clip_gradients_or_skip(layer.parameters(), 1.0)
    assert finite
    assert torch.equal(layer.weight.grad, before)


def test_nan_gradients_are_cleared_and_reported() -> None:
    layer = _layer_with_gradients(float("nan"), 1.0)
    total_norm, finite = clip_gradients_or_skip(layer.parameters(), 1.0)
    assert not finite
    assert not torch.isfinite(total_norm)
    assert layer.weight.grad is None and layer.bias.grad is None


def test_infinite_gradients_are_cleared_and_reported() -> None:
    layer = _layer_with_gradients(float("inf"), 1.0)
    total_norm, finite = clip_gradients_or_skip(layer.parameters(), 1.0)
    assert not finite
    assert not torch.isfinite(total_norm)
    assert layer.weight.grad is None and layer.bias.grad is None


def test_skipped_step_leaves_parameters_unchanged() -> None:
    layer = _layer_with_gradients(float("nan"), 1.0)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1.0)
    before = layer.weight.detach().clone()
    _, finite = clip_gradients_or_skip(layer.parameters(), 1.0)
    if finite:  # pragma: no cover - the helper must report the NaN
        optimizer.step()
    assert torch.equal(layer.weight.detach(), before)
