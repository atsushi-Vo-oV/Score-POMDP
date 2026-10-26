"""core6.1 = core6 blearn with the alpha_l / beta_l schedule learning rate cut from 0.025 to 0.0025."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from sb_pomdp.config import load_config
from sb_pomdp.envs import make_env
from sb_pomdp.train import build_optimizer, model_for_environment

TASKS = ["mirror", "mirror2", "tps"]


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


@pytest.mark.parametrize("task", TASKS)
def test_core61_arm_differs_from_core6_blearn_only_in_the_schedule_rate(task: str) -> None:
    core6 = _flatten(load_config(f"config/core6_{task}_blearn.json").to_dict())
    core61 = _flatten(load_config(f"config/core61_{task}_blslow.json").to_dict())
    changed = {key for key in core6.keys() | core61.keys() if core6.get(key) != core61.get(key)}
    assert changed == {"ppo.langevin_schedule_lr_multiplier", "experiment.label"}
    assert core61["experiment.label"] == f"core61-{task}-blslow"
    assert core61["ppo.learning_rate"] * core61["ppo.langevin_schedule_lr_multiplier"] == pytest.approx(0.0025)


def test_core61_schedule_group_runs_at_one_tenth_of_core6() -> None:
    resolved = load_config("config/core61_mirror_blslow.json").to_dict()
    env = make_env("mountain_hike", resolved["environment"]["tasks"]["mountain_hike"], seed=0)
    model = model_for_environment(env, dict(resolved["model"]), torch.device("cpu"))
    rates = sorted(group["lr"] for group in build_optimizer(model, resolved["ppo"]).param_groups)
    assert rates == pytest.approx(sorted([resolved["ppo"]["learning_rate"], 0.0025]))
