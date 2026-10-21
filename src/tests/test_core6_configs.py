"""core6 = the core4 score transformer on the multimodal tasks with the ULA drift and noise decoupled.

b02    : alpha 0.2, beta 0.2 fixed (the chain noise of core4, ten times its drift).
blearn : per-step alpha_l and beta_l learned from 0.2 / 0.2 with a sigmoid bound and a schedule rate of 0.025.
b02wd  : b02 with AdamW weight decay 0.01 on the network weights only.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, load_config, validate_config
from sb_pomdp.envs import make_env
from sb_pomdp.train import build_optimizer, langevin_schedule_summary, model_for_environment

TASKS = ["mirror", "mirror2", "tps"]
EXPECTED_DIFFERENCES = {
    "b02": {"model.langevin_step_size", "model.langevin_noise_scale"},
    "blearn": {
        "model.langevin_step_size",
        "model.langevin_noise_scale",
        "model.langevin_step_size_learnable",
        "model.langevin_noise_scale_learnable",
        "model.langevin_schedule_bound",
        "ppo.langevin_schedule_lr_multiplier",
    },
    "b02wd": {"model.langevin_step_size", "model.langevin_noise_scale", "ppo.weight_decay"},
}


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


@pytest.mark.parametrize("variant", ["b02", "blearn", "b02wd"])
@pytest.mark.parametrize("task", TASKS)
def test_core6_arm_differs_from_core4_only_in_the_chain_coefficients(task: str, variant: str) -> None:
    core4 = _flatten(load_config(f"config/core4_{task}.json").to_dict())
    core6 = _flatten(load_config(f"config/core6_{task}_{variant}.json").to_dict())
    changed = {key for key in core4.keys() | core6.keys() if core4.get(key) != core6.get(key)}
    assert changed == EXPECTED_DIFFERENCES[variant] | {"comparison.methods", "experiment.label"}
    assert core6["comparison.methods"] == ["score_transformer"]
    assert core6["experiment.label"] == f"core6-{task}-{variant}"
    assert (core6["model.langevin_step_size"], core6["model.langevin_noise_scale"]) == (0.2, 0.2)
    assert core4["model.langevin_noise_scale"] is None  # core4 keeps the coupled sqrt(2 alpha tau) noise
    if variant == "blearn":
        assert core6["model.langevin_step_size_learnable"] and core6["model.langevin_noise_scale_learnable"]
        assert core6["ppo.learning_rate"] * core6["ppo.langevin_schedule_lr_multiplier"] == pytest.approx(0.025)
    if variant == "b02wd":
        assert core6["ppo.weight_decay"] == 0.01


def _model(name: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    resolved = load_config(name).to_dict()
    env = make_env("mountain_hike", resolved["environment"]["tasks"]["mountain_hike"], seed=0)
    return model_for_environment(env, dict(resolved["model"]), torch.device("cpu")), resolved


def test_core6_learned_schedules_start_at_their_fixed_values_and_share_the_schedule_rate() -> None:
    model, resolved = _model("config/core6_mirror_blearn.json")
    summary = langevin_schedule_summary(model)
    assert set(summary) == {"step_size", "noise_scale"}
    for name in ("step_size", "noise_scale"):
        assert summary[name] == pytest.approx((0.2, 0.2, 0.2), rel=1e-4)
    optimizer = build_optimizer(model, resolved["ppo"])
    assert isinstance(optimizer, torch.optim.Adam) and not isinstance(optimizer, torch.optim.AdamW)
    assert sorted(group["lr"] for group in optimizer.param_groups) == pytest.approx(
        sorted([resolved["ppo"]["learning_rate"], 0.025])
    )


def test_core6_weight_decay_uses_adamw_on_the_weights_only() -> None:
    model, resolved = _model("config/core6_tps_b02wd.json")
    assert langevin_schedule_summary(model) == {}
    optimizer = build_optimizer(model, resolved["ppo"])
    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.01]
    assert sum(p.numel() for group in optimizer.param_groups for p in group["params"]) == sum(
        p.numel() for p in model.parameters()
    )
    learned, _ = _model("config/core6_mirror_blearn.json")
    learned_ppo = dict(resolved["ppo"], weight_decay=0.01, langevin_schedule_lr_multiplier=100.0)
    groups = build_optimizer(learned, learned_ppo).param_groups
    assert [group["weight_decay"] for group in groups] == [0.01, 0.0]


def _belief(**kwargs: Any) -> EnergyBelief:
    torch.manual_seed(0)
    return EnergyBelief(2, 2, 2, [8], langevin_warm_start=True, langevin_steps=3, langevin_step_size=0.2, **kwargs)


def _chain(belief: EnergyBelief, alpha: float) -> torch.Tensor:
    torch.manual_seed(1)
    observation = torch.randn(1, 2)
    previous = torch.randn(1, 4, 2)
    action = torch.zeros(1, 2)
    initial = torch.zeros(1, dtype=torch.bool)
    initial_noise, step_noise = belief.draw_particle_noise(observation, num_particles=4, num_steps=3,
                                                           generator=torch.Generator().manual_seed(2))
    particles, _ = belief.particles_from_noise(observation, previous, action, initial, initial_noise=initial_noise,
                                               langevin_noise=step_noise, step_size=alpha, track_grad=False)
    return particles


def test_decoupled_noise_equals_the_coupled_sampler_at_beta_sqrt_2_alpha_tau() -> None:
    coupled = _belief()
    decoupled = _belief(langevin_noise_scale=math.sqrt(2 * 0.2))
    decoupled.load_state_dict(coupled.state_dict())
    assert torch.allclose(_chain(coupled, 0.2), _chain(decoupled, 0.2), atol=1e-6)
    narrower = _belief(langevin_noise_scale=0.2)
    narrower.load_state_dict(coupled.state_dict())
    assert not torch.allclose(_chain(coupled, 0.2), _chain(narrower, 0.2), atol=1e-3)


def test_learned_noise_scale_is_bounded_and_receives_gradient() -> None:
    belief = _belief(langevin_noise_scale=0.2, langevin_noise_scale_learnable=True, langevin_schedule_bound="sigmoid")
    assert torch.allclose(belief.noise_scale_schedule(), torch.full((3,), 0.2), rtol=1e-5)
    with torch.no_grad():
        belief.langevin_log_noise_scale.fill_(50.0)
    assert float(belief.noise_scale_schedule().max()) <= 2.0 + 1e-6
    belief = _belief(langevin_noise_scale=0.2, langevin_noise_scale_learnable=True, langevin_schedule_bound="sigmoid")
    observation, previous, action = torch.randn(1, 2), torch.randn(1, 4, 2), torch.zeros(1, 2)
    initial_noise, step_noise = belief.draw_particle_noise(observation, num_particles=4, num_steps=3)
    particles, _ = belief.particles_from_noise(observation, previous, action, torch.zeros(1, dtype=torch.bool),
                                               initial_noise=initial_noise, langevin_noise=step_noise, step_size=0.2,
                                               track_grad=True)
    particles.square().sum().backward()
    assert belief.langevin_log_noise_scale.grad is not None and belief.langevin_log_noise_scale.grad.abs().sum() > 0


def test_config_rejects_learned_noise_scale_without_a_value_or_with_a_learned_temperature() -> None:
    resolved = load_config("config/core6_mirror_b02.json").to_dict()
    resolved["model"]["langevin_noise_scale_learnable"] = True
    resolved["model"]["langevin_noise_scale"] = None
    with pytest.raises(ConfigError):
        validate_config(resolved)
    resolved["model"]["langevin_noise_scale"] = 0.2
    resolved["model"]["langevin_temperature_learnable"] = True
    with pytest.raises(ConfigError):
        validate_config(resolved)
    resolved["model"]["langevin_temperature_learnable"] = False
    resolved["ppo"]["weight_decay"] = -1.0
    with pytest.raises(ConfigError):
        validate_config(resolved)
