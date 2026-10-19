"""core5 = core4 plus two score-transformer arms that change only the ULA step size alpha.

a02    : alpha 0.2 fixed.
alearn : per-step alpha_l learned from 0.02 with a sigmoid bound and a schedule learning rate of 0.025.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from sb_pomdp.config import load_config
from sb_pomdp.envs import make_env
from sb_pomdp.train import build_optimizer, langevin_schedule_summary, model_for_environment

TASKS = ["mh", "ld2d", "pendbal", "mirror", "mirror2", "tps"]
TASK_NAMES = {
    "mh": "mountain_hike",
    "ld2d": "light_dark",
    "pendbal": "masked_pendulum",
    "mirror": "mountain_hike",
    "mirror2": "mountain_hike",
    "tps": "mountain_hike",
}
EXPECTED_DIFFERENCES = {
    "a02": {"model.langevin_step_size"},
    "alearn": {
        "model.langevin_step_size_learnable",
        "model.langevin_schedule_bound",
        "ppo.langevin_schedule_lr_multiplier",
    },
}


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


@pytest.mark.parametrize("variant", ["a02", "alearn"])
@pytest.mark.parametrize("task", TASKS)
def test_core5_arm_differs_from_core4_only_in_the_step_size(task: str, variant: str) -> None:
    core4 = _flatten(load_config(f"config/core4_{task}.json").to_dict())
    core5 = _flatten(load_config(f"config/core5_{task}_{variant}.json").to_dict())
    changed = {key for key in core4.keys() | core5.keys() if core4.get(key) != core5.get(key)}
    assert changed == EXPECTED_DIFFERENCES[variant] | {"comparison.methods", "experiment.label"}
    assert core5["comparison.methods"] == ["score_transformer"]
    assert core5["experiment.label"] == f"core5-{task}-{variant}"
    assert core5["experiment.seeds"] == [10, 11, 12, 13, 14]
    # the core4 k16f budget is untouched
    assert (core5["model.num_particles"], core5["model.langevin_steps"], core5["model.energy_hidden"]) == (16, 12, [8])
    assert core5["model.langevin_temperature"] == 1.0 and core5["model.langevin_temperature_learnable"] is False
    if variant == "a02":
        assert core5["model.langevin_step_size"] == 0.2
        assert core5["model.langevin_step_size_learnable"] is False
    else:
        assert core5["model.langevin_step_size"] == 0.02  # initial value = core4
        assert core5["model.langevin_step_size_learnable"] is True
        assert core5["model.langevin_schedule_bound"] == "sigmoid"
        schedule_lr = core5["ppo.learning_rate"] * core5["ppo.langevin_schedule_lr_multiplier"]
        assert schedule_lr == pytest.approx(0.025)


def test_core5_tps_keeps_the_core4_update_count() -> None:
    for name in ("config/core4_tps.json", "config/core5_tps_a02.json", "config/core5_tps_alearn.json"):
        ppo = load_config(name).to_dict()["ppo"]
        assert ppo["total_steps"] // (ppo["num_envs"] * ppo["rollout_steps"]) == 100


def _model(name: str, task: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    resolved = load_config(name).to_dict()
    env = make_env(TASK_NAMES[task], resolved["environment"]["tasks"][TASK_NAMES[task]], seed=0)
    return model_for_environment(env, dict(resolved["model"]), torch.device("cpu")), resolved


@pytest.mark.parametrize("task", ["mh", "pendbal"])
def test_core5_learned_step_size_starts_at_core4_and_gets_its_own_rate(task: str) -> None:
    model, resolved = _model(f"config/core5_{task}_alearn.json", task)
    summary = langevin_schedule_summary(model)
    assert set(summary) == {"step_size"}
    mean, low, high = summary["step_size"]
    assert mean == pytest.approx(0.02, rel=1e-4) and low == pytest.approx(0.02, rel=1e-4) and high == pytest.approx(0.02, rel=1e-4)
    optimizer = build_optimizer(model, resolved["ppo"])
    rates = sorted(group["lr"] for group in optimizer.param_groups)
    assert rates == pytest.approx(sorted([resolved["ppo"]["learning_rate"], 0.025]))


def test_core5_fixed_step_size_logs_no_schedule() -> None:
    model, _ = _model("config/core5_mh_a02.json", "mh")
    assert langevin_schedule_summary(model) == {}
