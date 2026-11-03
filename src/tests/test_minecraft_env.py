"""MinecraftWalkEnv with the in-process fake backend (no Minecraft, no VAE)."""
from __future__ import annotations

import numpy as np
import pytest

from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.envs import make_env
from sb_pomdp.minecraft_env import NUM_ACTIONS, MinecraftWalkEnv


def _env(seed=3, horizon=20):
    return make_env("minecraft_walk", {"horizon": horizon, "observation_noise_std": 0.0, "backend": "fake"}, seed=seed)


def test_fake_backend_runs_an_episode_with_heading_progress_reward():
    env = _env()
    assert isinstance(env, MinecraftWalkEnv)
    obs, info = env.reset(seed=3)
    assert obs.shape == (48,) and obs.dtype == np.float32
    assert info["state"].shape == (6,) and info["progress"] == 0.0
    total = 0.0
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(1)  # forward
        total += reward
    assert truncated and not terminated
    assert total > 0.0 and abs(total - info["progress"]) < 1e-6
    oracle = env.oracle_features()
    assert oracle.shape == (4,) and oracle[0] == pytest.approx(info["progress"] / 10.0) and abs(oracle[1]) < 1e-6
    with pytest.raises(RuntimeError):
        env.step(1)


def test_turning_changes_heading_feature_and_observation_depends_on_pose():
    env = _env(seed=5, horizon=50)
    obs0, _ = env.reset(seed=5)
    for _ in range(6):
        obs, _, _, _, _ = env.step(6)  # turn right
    cos, sin = env.oracle_features()[2:]
    assert cos < 0.999 and abs(sin) > 0.1
    assert not np.allclose(obs0, obs)
    assert env.action_spec.n == NUM_ACTIONS


def test_state_dict_round_trip_forces_a_fresh_episode():
    env = _env(seed=7)
    env.reset(seed=7)
    env.step(1)
    other = _env(seed=8)
    other.load_state_dict(env.state_dict())
    with pytest.raises(RuntimeError):
        other.step(1)
    obs, info = other.reset()
    assert obs.shape == (48,) and info["elapsed_steps"] == 0


def test_config_schema_fills_minecraft_defaults_and_validates():
    base = load_config("config/trysvgdlight_tps_tf-svgd.json")
    data = base.to_dict()
    data["environment"]["tasks"]["minecraft_walk"] = {"horizon": 200, "observation_noise_std": 0.0, "backend": "fake"}
    data["experiment"]["tasks"] = ["minecraft_walk"]
    from sb_pomdp.config import ExperimentConfig

    config = ExperimentConfig(data)
    task = config.to_dict()["environment"]["tasks"]["minecraft_walk"]
    assert task["action_repeat"] == 4 and task["latent_dim"] == 32 and task["backend"] == "fake"
    data["environment"]["tasks"]["minecraft_walk"]["backend"] = "unity"
    with pytest.raises(ConfigError):
        ExperimentConfig(data)
    env = make_env("minecraft_walk", task, seed=1)
    assert env.observation_dim == 48


def test_minerl_backend_needs_paths():
    with pytest.raises(ValueError):
        MinecraftWalkEnv(horizon=10, backend="minerl")
