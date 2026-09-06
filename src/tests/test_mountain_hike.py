"""Mountain Hike (DVRL, Igl et al. 2018) environment fidelity tests."""

from __future__ import annotations

import numpy as np
import pytest

from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.envs import MountainHikeEnv, make_env, mountain_hike_terrain


def _reference_terrain(x: float, y: float, hill_height: float = 4.0, power: float = 4.0) -> float:
    """Independent transcription of DeathValleyEnv.get_reward (mlab.bivariate_normal)."""

    means = [(0.5, 0.5), (0.0, 0.2), (-0.375, -0.5)]
    sigmas = [(0.75, 0.1), (0.1, 0.75), (0.75, 0.1)]
    factors = [1.0, 0.8, 0.55]
    z = 0.0
    for (mx, my), (sx, sy), factor in zip(means, sigmas, factors):
        density = np.exp(-0.5 * (((x - mx) / sx) ** 2 + ((y - my) / sy) ** 2)) / (
            2.0 * np.pi * sx * sy
        )
        z = max(z, factor * density / 3.0)
    z = 1.0 - (1.0 - z) ** power
    return z * hill_height - (0.5 + hill_height) + (x + y) / 4.0


def test_terrain_matches_the_reference_formula_and_has_the_expected_shape() -> None:
    for x, y in ((0.5, 0.5), (0.0, 0.2), (-0.375, -0.5), (-0.85, -0.85), (0.7, 0.5), (0.9, -0.9)):
        assert float(mountain_hike_terrain(x, y)) == pytest.approx(_reference_terrain(x, y), rel=1e-9)
    # The start corner is deep (about -4.9), the ridge tops are near -0.3.
    assert float(mountain_hike_terrain(-0.85, -0.85)) < -4.5
    assert float(mountain_hike_terrain(0.5, 0.5)) > -0.4
    assert float(mountain_hike_terrain(0.7, 0.5)) > float(mountain_hike_terrain(-0.85, -0.85)) + 4.0


def test_factory_defaults_follow_the_dvrl_configuration() -> None:
    env = make_env("mountain_hike", None, seed=0)
    assert isinstance(env, MountainHikeEnv)
    assert env.horizon == 75
    assert env.observation_noise_std == pytest.approx(3.0)
    assert env.transition_std == pytest.approx(0.25)
    assert env.max_action == pytest.approx(0.5)
    assert env.start_std == pytest.approx(1.0)
    assert env.action_cost == pytest.approx(0.01)
    assert env.action_spec.is_continuous and env.action_spec.shape == (2,)
    env2 = make_env("DeathValley-v0", {"observation_noise_std": 1.5, "horizon": 10}, seed=1)
    assert env2.observation_noise_std == pytest.approx(1.5) and env2.horizon == 10


def test_reset_observation_and_step_dynamics() -> None:
    env = make_env("mountain_hike", {"observation_noise_std": 0.0, "transition_std": 0.0}, seed=3)
    obs, info = env.reset(seed=3)
    assert obs.shape == (2,) and np.allclose(obs, info["state"])
    assert np.linalg.norm(info["state"] - np.array([-8.5, -8.5])) < 5.0
    env._state = np.array([-8.5, -8.5])  # inside the box (the start prior can leave it)
    before = env._state.copy()
    # Norm clipping: a request of norm 1.0 is scaled to 0.5, penalty uses the requested norm.
    obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5]))
    moved = info["state"] - before
    assert np.linalg.norm(moved) == pytest.approx(0.5, abs=1e-9)
    expected = env.terrain_reward(info["state"]) - 0.01 * np.linalg.norm([0.5, 0.5])
    assert reward == pytest.approx(expected, rel=1e-9)
    assert not terminated and not truncated
    assert info["terrain_reward"] == pytest.approx(env.terrain_reward(info["state"]))


def test_outside_box_costs_a_flat_penalty_and_does_not_terminate() -> None:
    env = make_env("mountain_hike", {"observation_noise_std": 0.0, "transition_std": 0.0}, seed=0)
    env.reset(seed=0)
    env._state = np.array([-10.4, 0.0])
    _obs, reward, terminated, _truncated, info = env.step(np.array([-0.5, 0.0]))
    assert info["outside_box"]
    assert reward == pytest.approx(-1.5 * 4.0 - 0.01 * 0.5)
    assert not terminated


def test_goal_is_inert_by_default_and_pays_when_configured() -> None:
    quiet = {"observation_noise_std": 0.0, "transition_std": 0.0}
    env = make_env("mountain_hike", quiet, seed=0)
    env.reset(seed=0)
    env._state = np.array([7.0, 5.0])
    _, reward, terminated, _, _ = env.step(np.zeros(2))
    assert reward == pytest.approx(env.terrain_reward(np.array([7.0, 5.0])))
    assert not terminated
    paying = make_env("mountain_hike", dict(quiet, goal_reward=10.0, goal_end=True), seed=0)
    paying.reset(seed=0)
    paying._state = np.array([7.0, 5.0])
    _, reward, terminated, _, _ = paying.step(np.zeros(2))
    assert reward == pytest.approx(10.0) and terminated


def test_horizon_truncates_and_noise_is_seeded() -> None:
    env = make_env("mountain_hike", {"horizon": 3}, seed=5)
    env.reset(seed=5)
    flags = [env.step(np.zeros(2))[3] for _ in range(3)]
    assert flags == [False, False, True]
    a = make_env("mountain_hike", None, seed=9)
    b = make_env("mountain_hike", None, seed=9)
    oa, _ = a.reset(seed=9)
    ob, _ = b.reset(seed=9)
    assert np.allclose(oa, ob)
    sa = a.step(np.array([0.3, -0.2]))
    sb = b.step(np.array([0.3, -0.2]))
    assert np.allclose(sa[0], sb[0]) and sa[1] == pytest.approx(sb[1])
    assert a.oracle_features().shape == (2,) and np.all(np.abs(a.oracle_features()) < 1.0)


def test_state_dict_round_trip_restores_the_generator() -> None:
    env = make_env("mountain_hike", None, seed=11)
    env.reset(seed=11)
    env.step(np.array([0.1, 0.1]))
    snapshot = env.state_dict()
    clone = make_env("mountain_hike", None, seed=0)
    clone.load_state_dict(snapshot)
    a = env.step(np.array([0.2, -0.1]))
    b = clone.step(np.array([0.2, -0.1]))
    assert np.allclose(a[0], b[0]) and a[1] == pytest.approx(b[1])


def test_config_accepts_a_mountain_hike_task_block() -> None:
    config = load_config("config/local.json")
    accepted = config.with_overrides(
        {
            "experiment.tasks": ["mountain_hike"],
            "environment.tasks": {
                "mountain_hike": {
                    "horizon": 75,
                    "observation_noise_std": 3.0,
                    "transition_std": 0.25,
                    "goal_end": False,
                }
            },
        }
    )
    assert accepted["environment"]["tasks"]["mountain_hike"]["horizon"] == 75
    with pytest.raises(ConfigError):
        config.with_overrides(
            {
                "experiment.tasks": ["mountain_hike"],
                "environment.tasks": {
                    "mountain_hike": {"horizon": 75, "observation_noise_std": 3.0, "max_action": 0.0}
                },
            }
        )
    with pytest.raises(ConfigError):
        config.with_overrides(
            {
                "experiment.tasks": ["mountain_hike"],
                "environment.tasks": {
                    "mountain_hike": {"horizon": 75, "observation_noise_std": 3.0, "unknown_key": 1}
                },
            }
        )


def test_teleport_relocates_uniformly_and_is_off_by_default() -> None:
    import numpy as np

    from sb_pomdp.envs import MountainHikeEnv

    plain = MountainHikeEnv(observation_noise_std=0.0, transition_std=0.0)
    plain.reset(seed=3)
    before = plain._state.copy()
    _, _, _, _, info = plain.step(np.array([0.1, 0.0]))
    assert info["teleported"] is False
    assert np.allclose(plain._state, before + np.array([0.1, 0.0]))

    jumpy = MountainHikeEnv(observation_noise_std=0.0, transition_std=0.0, teleport_probability=0.999)
    jumpy.reset(seed=3)
    states = []
    for _ in range(20):
        _, _, _, _, info = jumpy.step(np.array([0.0, 0.0]))
        assert info["teleported"] is True
        states.append(jumpy._state.copy())
    states = np.stack(states)
    assert np.all(np.abs(states) <= 10.0) and states.std(axis=0).min() > 2.0
    with pytest.raises(ValueError):
        MountainHikeEnv(teleport_probability=1.0)


def test_teleport_probability_is_a_validated_task_key() -> None:
    from sb_pomdp.config import ConfigError, load_config

    config = load_config("config/mountain_hike.json")
    assert config.to_dict()["environment"]["tasks"]["mountain_hike"]["teleport_probability"] == 0.0
    accepted = config.with_overrides({"environment.tasks.mountain_hike.teleport_probability": 0.03})
    assert accepted.to_dict()["environment"]["tasks"]["mountain_hike"]["teleport_probability"] == 0.03
    with pytest.raises(ConfigError):
        config.with_overrides({"environment.tasks.mountain_hike.teleport_probability": 1.5})
