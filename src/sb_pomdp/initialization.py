"""Opt-in, weights-only campaign initialization (never a training resume)."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import torch

from .config import normalize_legacy_resolved_config
from .policies import DiffusionPolicy


# Tensor keys/shapes are checked as well. These distinguish shape-compatible
# architectures, e.g. different attention head counts or diffusion noise floors.
_STRUCTURE_KEYS = (
    "num_particles", "langevin_steps", "belief_latent_dim", "energy_hidden",
    "d_model", "num_heads", "num_transformer_layers", "transformer_ff_dim",
    "policy_hidden", "diffusion_steps", "diffusion_beta_start", "diffusion_beta_end",
    "diffusion_min_std", "encoder_kind", "continuous_policy_kind", "policy_head_kind",
    "particle_sampler", "score_parameterization", "energy_network_kind",
    "head_network_kind", "trunk_network_kind", "critic_kind", "value_encoder",
    "policy_observation_input", "encoder_use_scores", "prediction_head",
    "initial_belief", "aux_predictor_kind", "kan_grid_size", "kan_spline_order",
    "basis_order", "basis_projection_dim", "langevin_schedule_bound",
)


def _finite_state(state: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(state, Mapping) or any(
        not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all())
        for tensor in state.values()
    ):
        raise ValueError("initialization state must contain only finite tensors")


def _compatible_state(module: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Validate before load_state_dict, which can otherwise partially mutate a module."""
    current = module.state_dict()
    if current.keys() != state.keys() or any(
        current[key].shape != value.shape or current[key].dtype != value.dtype
        for key, value in state.items()
    ):
        raise ValueError("initialization model tensors are incompatible")


def load_initial_weights(model: torch.nn.Module, path: str | Path, model_config: dict) -> None:
    """Load just model tensors; do not inspect or restore optimizer, progress or RNG."""
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload or "config" not in payload:
        raise ValueError("init_from_checkpoint needs a checkpoint with model and config")
    source = normalize_legacy_resolved_config(payload["config"])["model"]
    target = normalize_legacy_resolved_config({"model": model_config})["model"]
    mismatches = [key for key in _STRUCTURE_KEYS if source.get(key) != target.get(key)]
    if mismatches:
        raise ValueError(f"initial checkpoint model structure differs: {', '.join(mismatches)}")
    _finite_state(payload["model"])
    _compatible_state(model, payload["model"])
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError, KeyError) as exc:
        raise ValueError("initial checkpoint model tensors are incompatible") from exc


def load_bc_actor(policy: DiffusionPolicy, path: str | Path) -> None:
    """Load the followup_bc format, with an explicit normalization convention.

    The November 10 exporter folds normalization BEFORE saving state_dict. Its
    historical payloads lack a flag and must therefore be treated as folded.
    New, *unfolded* exports must set meta.normalization_folded=false.
    """
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not all(
        key in payload for key in ("state_dict", "meta", "xm", "xs")
    ):
        raise ValueError("actor_init_path needs state_dict, meta, xm and xs")
    meta = payload["meta"]
    if not isinstance(meta, Mapping):
        raise ValueError("BC actor meta must be a mapping")
    if meta.get("steps", policy.num_steps) != policy.num_steps or meta.get("min_std", policy.min_std) != policy.min_std:
        raise ValueError("BC actor diffusion steps/min_std do not match")
    folded = meta.get("normalization_folded", True)
    if not isinstance(folded, bool):
        raise ValueError("BC actor normalization_folded must be boolean")
    state = payload["state_dict"]
    _finite_state(state)
    _compatible_state(policy, state)
    # A strict load must not silently change the environment bounds or noise schedule.
    for key, value in policy.named_buffers():
        if key not in state or not torch.equal(state[key].cpu(), value.cpu()):
            raise ValueError(f"BC actor buffer differs: {key}")
    first = policy.denoiser[0]
    if not isinstance(first, torch.nn.Linear):
        raise ValueError("BC actor normalization requires a linear first denoiser layer")
    condition_dim = first.in_features - policy.action_dim - policy.time_embedding.dimension
    xm, xs = payload["xm"], payload["xs"]
    if not all(isinstance(t, torch.Tensor) and t.shape == (condition_dim,) and bool(torch.isfinite(t).all()) for t in (xm, xs)) or not bool((xs > 0).all()):
        raise ValueError("BC actor normalization statistics have invalid shape or values")
    if not folded:
        state = {key: value.clone() for key, value in state.items()}
        weight = state["denoiser.0.weight"][:, :condition_dim].clone()
        state["denoiser.0.weight"][:, :condition_dim] = weight / xs
        state["denoiser.0.bias"] -= weight @ (xm / xs)
        _finite_state(state)
    try:
        policy.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, KeyError) as exc:
        raise ValueError("BC actor tensors are incompatible") from exc


def initialize_model(model: torch.nn.Module, model_config: dict, *, resume: bool) -> None:
    """Fresh construction -> checkpoint -> BC override -> fresh critic -> freeze.

    Resume bypasses every external asset and reset. Freezing is re-applied before
    optimizer construction to reproduce its parameter groups.
    """
    heads = getattr(model, "heads", model)
    if not resume:
        value_head = getattr(heads, "value_head", None)
        fresh_value = (
            {key: value.clone() for key, value in value_head.state_dict().items()}
            if model_config.get("reset_value_head", False) and value_head is not None else None
        )
        if model_config.get("init_from_checkpoint"):
            load_initial_weights(model, model_config["init_from_checkpoint"], model_config)
        if model_config.get("actor_init_path"):
            policy = getattr(heads, "diffusion_policy", None)
            if not isinstance(policy, DiffusionPolicy):
                raise ValueError("actor_init_path requires a diffusion policy")
            load_bc_actor(policy, model_config["actor_init_path"])
        if model_config.get("reset_value_head", False):
            if fresh_value is None:
                raise ValueError("reset_value_head requires a state-value head")
            value_head.load_state_dict(fresh_value)
    for prefix in model_config.get("freeze_modules", []):
        matched = [p for name, p in model.named_parameters() if name.startswith(prefix)]
        if not matched:
            raise ValueError(f"freeze_modules prefix matches no parameters: {prefix!r}")
        for parameter in matched:
            parameter.requires_grad_(False)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("freeze_modules leaves no trainable parameters")


@contextmanager
def critic_only(model: torch.nn.Module, enabled: bool):
    """Temporarily restrict a baseline PPO optimizer step to the value head."""
    if not enabled:
        yield
        return
    parameters = list(model.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    value_parameters = set(getattr(model, "heads", model).value_head.parameters())
    try:
        for parameter in parameters:
            parameter.requires_grad_(parameter.requires_grad and parameter in value_parameters)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
