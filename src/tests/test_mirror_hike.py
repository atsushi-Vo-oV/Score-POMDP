"""Mirror-world Mountain Hike: a persistent four-modal position belief that matters for control."""

from __future__ import annotations

import numpy as np
import pytest

from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.envs import MountainHikeEnv, make_env, mountain_hike_terrain

MIRROR = {
    "mirror_symmetry": True,
    "start_distribution": "mirror_corners",
    "terrain_ridges": [0],
    "signpost_width": 2.0,
    "signpost_noise_std": 0.0,
    "observation_noise_std": 0.0,
    "transition_std": 0.0,
}
FRAMES = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]


def _place(env: MountainHikeEnv, mirror, state) -> None:
    env._mirror = np.array(mirror, dtype=np.float64)
    env._state = np.array(state, dtype=np.float64)
    env._elapsed_steps = 0
    env._needs_reset = False


def test_ridge_subset_keeps_ridge_one_and_flattens_the_others() -> None:
    assert float(mountain_hike_terrain(0.8, 0.5, ridges=[0])) == pytest.approx(
        float(mountain_hike_terrain(0.8, 0.5))
    )
    assert float(mountain_hike_terrain(0.0, 0.0)) > -1.0  # ridge 2 passes through the origin
    assert float(mountain_hike_terrain(0.0, 0.0, ridges=[0])) < -4.0  # removed: the origin is deep
    assert float(mountain_hike_terrain(0.8, -0.5, ridges=[0])) < -4.0  # mirror image of the ridge
    with pytest.raises(ValueError):
        mountain_hike_terrain(0.0, 0.0, ridges=[3])
    env = make_env("mountain_hike", {"terrain_ridges": [0]}, seed=0)
    assert env.terrain_reward(np.array([0.0, 0.0])) < -4.0
    assert env.terrain_reward(np.array([8.0, 5.0])) > -0.6


def test_mirror_frame_is_invisible_to_body_frame_dynamics_and_observations() -> None:
    env = MountainHikeEnv(**MIRROR, seed=0)
    assert env.observation_dim == 4
    env.reset()
    increments = []
    for frame in FRAMES:
        _place(env, frame, [3.0, -2.0])
        obs, reward, terminated, truncated, info = env.step(np.array([0.3, 0.4]))
        assert obs.shape == (4,)
        # the world moves by m * a, the agent sees m * s: the body-frame increment is frame-free
        assert np.allclose(info["state"], [3.0 + 0.3 * frame[0], -2.0 + 0.4 * frame[1]])
        assert np.allclose(obs[:2], np.array(frame) * info["state"])
        increments.append(obs[:2] - np.array(frame) * np.array([3.0, -2.0]))
        assert np.array_equal(info["mirror"], np.array(frame))
    assert all(np.allclose(increment, [0.3, 0.4]) for increment in increments)


def test_rewards_differ_between_mirror_worlds_at_the_same_body_position() -> None:
    env = MountainHikeEnv(**MIRROR, seed=0)
    body = np.array([8.0, 5.0])
    rewards = {frame: env.terrain_reward(np.array(frame) * body) for frame in FRAMES}
    assert rewards[(1.0, 1.0)] > max(r for f, r in rewards.items() if f != (1.0, 1.0)) + 2.0


def test_signpost_reveals_the_frame_only_near_the_origin() -> None:
    env = MountainHikeEnv(**MIRROR, seed=0)
    env.reset()
    _place(env, (-1.0, 1.0), [0.0, 0.0])
    obs, *_ = env.step(np.zeros(2))
    assert np.allclose(obs[2:], [-1.0, 1.0], atol=1e-6)
    _place(env, (-1.0, 1.0), [6.0, 6.0])
    obs, *_ = env.step(np.zeros(2))
    assert np.all(np.abs(obs[2:]) < 1e-3)
    assert env.signpost_signal(np.array([2.0, 0.0])) == pytest.approx(np.exp(-0.5))


def test_start_corners_are_uniform_and_independent_of_the_frame() -> None:
    env = MountainHikeEnv(**MIRROR, seed=11)
    body_corners, agreements = set(), 0
    for _ in range(400):
        obs, info = env.reset()
        assert np.all(np.abs(np.abs(info["state"]) - 8.5) < 5.0)
        body_corners.add(tuple(np.sign(obs[:2]).astype(int)))
        agreements += int(np.array_equal(np.sign(info["state"]), info["mirror"]))
    assert len(body_corners) == 4
    assert 0.15 < agreements / 400 < 0.35  # P(frame == start corner signs) = 1/4
    uniform = MountainHikeEnv(**{**MIRROR, "start_distribution": "uniform"}, seed=2)
    states = np.stack([uniform.reset()[1]["state"] for _ in range(200)])
    assert np.all(np.abs(states) <= 10.0) and np.abs(states).mean() < 7.0


def test_legacy_configuration_is_unchanged() -> None:
    env = make_env("mountain_hike", None, seed=0)
    assert env.observation_dim == 2 and env.mirror_symmetry is False
    obs, info = env.reset()
    assert obs.shape == (2,) and "mirror" not in info and "mirror" not in env.state_dict()
    assert env.terrain_ridges is None
    reference = MountainHikeEnv(seed=0)
    assert np.array_equal(reference.reset()[0], obs)


def test_invalid_mirror_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        MountainHikeEnv(mirror_symmetry=True)  # the fixed start corner would reveal the frame
    with pytest.raises(ValueError):
        MountainHikeEnv(mirror_symmetry=True, start_distribution="uniform", observation_symmetry="abs")
    with pytest.raises(ValueError):
        MountainHikeEnv(start_distribution="corner")
    with pytest.raises(ValueError):
        MountainHikeEnv(terrain_ridges=[])
    with pytest.raises(ValueError):
        MountainHikeEnv(terrain_ridges=[0, 0])
    with pytest.raises(ValueError):
        MountainHikeEnv(signpost_width=0.0)
    with pytest.raises(ValueError):
        MountainHikeEnv(signpost_noise_std=-1.0)


def test_state_dict_round_trip_restores_the_mirror_frame() -> None:
    task = {**MIRROR, "observation_noise_std": 3.0, "signpost_noise_std": 0.2}
    env = make_env("mountain_hike", task, seed=5)
    env.reset()
    snapshot = env.state_dict()
    assert "mirror" in snapshot
    clone = make_env("mountain_hike", task, seed=6)
    clone.reset()
    clone.load_state_dict(snapshot)
    assert np.array_equal(clone._mirror, env._mirror)
    action = np.array([0.2, -0.3])
    assert np.array_equal(env.step(action)[0], clone.step(action)[0])
    with pytest.raises(ValueError):
        clone.load_state_dict({**snapshot, "mirror": np.array([2.0, 1.0])})
    with pytest.raises(ValueError):
        clone.load_state_dict({k: v for k, v in snapshot.items() if k != "mirror"})
    plain = make_env("mountain_hike", None, seed=1)
    plain.reset()
    with pytest.raises(ValueError):
        plain.load_state_dict({**plain.state_dict(), "mirror": np.ones(2)})


def test_config_validates_and_fills_the_mirror_keys() -> None:
    config = load_config("config/mountain_hike_v2.json")
    resolved = config.to_dict()["environment"]["tasks"]["mountain_hike"]
    assert resolved["mirror_symmetry"] is False
    assert resolved["start_distribution"] == "normal"
    assert resolved["terrain_ridges"] == [0, 1, 2]
    assert resolved["signpost_width"] == 2.0 and resolved["signpost_noise_std"] == 0.2
    prefix = "environment.tasks.mountain_hike."
    accepted = config.with_overrides(
        {
            prefix + "mirror_symmetry": True,
            prefix + "start_distribution": "mirror_corners",
            prefix + "terrain_ridges": [0],
        }
    )
    task = accepted.to_dict()["environment"]["tasks"]["mountain_hike"]
    env = make_env("mountain_hike", task, seed=0)
    assert env.observation_dim == 4 and env.terrain_ridges == (0,)
    with pytest.raises(ConfigError):
        config.with_overrides({prefix + "mirror_symmetry": True})
    with pytest.raises(ConfigError):
        config.with_overrides({prefix + "terrain_ridges": [3]})
    with pytest.raises(ConfigError):
        config.with_overrides({prefix + "terrain_ridges": []})
    with pytest.raises(ConfigError):
        config.with_overrides({prefix + "start_distribution": "corner"})
    with pytest.raises(ConfigError):
        config.with_overrides({prefix + "signpost_width": 0.0})
    with pytest.raises(ConfigError):
        config.with_overrides(
            {
                prefix + "mirror_symmetry": True,
                prefix + "start_distribution": "uniform",
                prefix + "observation_symmetry": "abs",
            }
        )


@pytest.mark.parametrize("design", ["v1", "v2", "v3", "v4", "v5", "v6"])
def test_mirror_grid_configs_load_and_build_the_task(design: str) -> None:
    config = load_config(f"config/mirror_{design}.json")
    task = config.to_dict()["environment"]["tasks"]["mountain_hike"]
    assert task["mirror_symmetry"] is True and task["start_distribution"] == "mirror_corners"
    assert task["terrain_ridges"] == [0] and task["horizon"] == 100
    env = make_env("mountain_hike", task, seed=0)
    obs, info = env.reset()
    assert obs.shape == (4,) and "mirror" in info
