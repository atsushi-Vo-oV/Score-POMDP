"""Separate learning rate for the learned Langevin schedules."""

from __future__ import annotations

import pytest
import torch

from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.policies import ScoreBeliefActorCritic
from sb_pomdp.train import build_optimizer, langevin_schedule_parameter_names


def _model(**flags: object) -> ScoreBeliefActorCritic:
    config = load_config("config/local.json").with_overrides(
        {f"model.{key}": value for key, value in flags.items()}
    )
    torch.manual_seed(0)
    return ScoreBeliefActorCritic(
        observation_dim=3,
        state_dim=2,
        action_kind="discrete",
        action_feature_dim=2,
        discrete_actions=2,
        action_low=None,
        action_high=None,
        model_config=dict(config["model"]),
    )


def _ppo(**overrides: object) -> dict[str, object]:
    return dict(load_config("config/local.json")["ppo"], **overrides)


def test_default_multiplier_keeps_the_single_group_optimizer() -> None:
    model = _model(
        langevin_step_size_learnable=True, langevin_temperature_learnable=True
    )
    optimizer = build_optimizer(model, _ppo())
    assert len(optimizer.param_groups) == 1
    assert len(optimizer.param_groups[0]["params"]) == len(list(model.parameters()))


def test_no_learned_schedule_ignores_the_multiplier() -> None:
    model = _model()
    assert langevin_schedule_parameter_names(model) == []
    optimizer = build_optimizer(model, _ppo(langevin_schedule_lr_multiplier=100.0))
    assert len(optimizer.param_groups) == 1


def test_multiplier_isolates_exactly_the_schedule_parameters() -> None:
    model = _model(
        langevin_step_size_learnable=True, langevin_temperature_learnable=True
    )
    names = langevin_schedule_parameter_names(model)
    assert sorted(names) == [
        "belief.langevin_log_step_size",
        "belief.langevin_log_temperature",
    ]
    ppo = _ppo(langevin_schedule_lr_multiplier=100.0)
    optimizer = build_optimizer(model, ppo)
    assert len(optimizer.param_groups) == 2
    main, schedule = optimizer.param_groups
    assert main["lr"] == pytest.approx(float(ppo["learning_rate"]))
    assert schedule["lr"] == pytest.approx(100.0 * float(ppo["learning_rate"]))
    schedule_ids = {id(parameter) for parameter in schedule["params"]}
    assert schedule_ids == {
        id(model.belief.langevin_log_step_size),
        id(model.belief.langevin_log_temperature),
    }
    assert len(main["params"]) + len(schedule["params"]) == len(
        list(model.parameters())
    )


def test_schedule_moves_at_the_scaled_rate() -> None:
    model = _model(langevin_step_size_learnable=True)
    ppo = _ppo(learning_rate=1e-3, langevin_schedule_lr_multiplier=50.0)
    optimizer = build_optimizer(model, ppo)
    before = model.belief.langevin_log_step_size.detach().clone()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    moved = (before - model.belief.langevin_log_step_size.detach()).abs()
    # Adam's first step moves every coordinate by ~lr regardless of gradient scale.
    assert torch.allclose(moved, torch.full_like(moved, 50.0 * 1e-3), rtol=1e-3)


def test_config_validates_the_multiplier() -> None:
    config = load_config("config/local.json")
    assert config.to_dict()["ppo"]["langevin_schedule_lr_multiplier"] == 1.0
    with pytest.raises(ConfigError):
        config.with_overrides({"ppo.langevin_schedule_lr_multiplier": 0.0})
    with pytest.raises(ConfigError):
        config.with_overrides({"ppo.langevin_schedule_lr_multiplier": -3})
    accepted = config.with_overrides({"ppo.langevin_schedule_lr_multiplier": 100})
    assert accepted["ppo"]["langevin_schedule_lr_multiplier"] == pytest.approx(100.0)
