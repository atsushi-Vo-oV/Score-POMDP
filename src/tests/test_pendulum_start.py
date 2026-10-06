"""Masked pendulum start distribution: swing-up defaults stay bit-identical, balancing variant via ranges."""

from __future__ import annotations

import numpy as np
import pytest

from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.envs import MaskedPendulumEnv, make_env


def test_defaults_are_the_gym_swing_up_start_and_bit_identical() -> None:
    env = MaskedPendulumEnv(seed=3)
    assert env.initial_angle_range == pytest.approx(np.pi)
    assert env.initial_speed_range == pytest.approx(1.0)
    obs, info = env.reset(seed=3)
    rng = np.random.default_rng(3)
    expected = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.0, 1.0)])
    assert np.array_equal(info["state"], expected)


def test_balancing_variant_starts_near_upright() -> None:
    env = make_env("masked_pendulum", {"initial_angle_range": 0.5, "initial_speed_range": 1.0}, seed=0)
    states = np.stack([env.reset()[1]["state"] for _ in range(300)])
    assert np.all(np.abs(states[:, 0]) <= 0.5) and np.all(np.abs(states[:, 1]) <= 1.0)
    assert np.abs(states[:, 0]).max() > 0.4 and np.abs(states[:, 1]).max() > 0.8
    with pytest.raises(ValueError):
        MaskedPendulumEnv(initial_angle_range=-0.1)
    with pytest.raises(ValueError):
        MaskedPendulumEnv(initial_angle_range=4.0)
    with pytest.raises(ValueError):
        MaskedPendulumEnv(initial_speed_range=float("nan"))


def test_config_layer_fills_and_validates_the_start_keys() -> None:
    swing_up = load_config("config/pendulum_v1.json").to_dict()["environment"]["tasks"]["masked_pendulum"]
    assert swing_up["initial_angle_range"] == pytest.approx(np.pi)
    assert swing_up["initial_speed_range"] == 1.0
    balance = load_config("config/pendulum_balance_v1.json")
    task = balance.to_dict()["environment"]["tasks"]["masked_pendulum"]
    assert task["initial_angle_range"] == 0.5 and task["initial_speed_range"] == 1.0
    env = make_env("masked_pendulum", task, seed=0)
    assert env.initial_angle_range == 0.5
    with pytest.raises(ConfigError):
        balance.with_overrides({"environment.tasks.masked_pendulum.initial_angle_range": 4.0})
    with pytest.raises(ConfigError):
        balance.with_overrides({"environment.tasks.masked_pendulum.initial_speed_range": -1.0})


@pytest.mark.parametrize("name", ["v1lf", "v2lf", "v3lf", "v4lf", "v5lf", "v6lf", "v2", "v4", "v5"])
def test_pendulum2_configs_carry_the_solving_ppo_settings(name: str) -> None:
    resolved = load_config(f"config/pendulum2_{name}.json").to_dict()
    ppo = resolved["ppo"]
    assert (ppo["gamma"], ppo["learning_rate"], ppo["epochs"], ppo["minibatch_size"]) == (0.9, 0.001, 10, 128)
    assert ppo["entropy_coef"] == 0.0 and ppo["value_target_transform"] == "none" and ppo["gae_lambda"] == 0.9
    model = resolved["model"]
    if name.endswith("lf"):
        assert (model["num_particles"], model["langevin_steps"], model["energy_hidden"]) == (4, 2, [32, 32])
    else:
        assert model["num_particles"] == 16 and model["langevin_warm_start"] is True
    assert resolved["experiment"]["tasks"] == ["masked_pendulum"]
