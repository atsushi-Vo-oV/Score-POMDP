"""Tests for the dependency-free masked control environments."""

from __future__ import annotations

import numpy as np
import pytest

from sb_pomdp.envs import (
    ActionSpec,
    LightDarkNDEnv,
    MaskedCartPoleEnv,
    MaskedMountainCarContinuousEnv,
    MaskedPendulumEnv,
    make_env,
)


def test_discrete_action_spec_validates_and_one_hot_encodes() -> None:
    spec = ActionSpec.discrete(3)

    assert spec.kind == "discrete"
    assert spec.shape == ()
    assert spec.feature_dim == 3
    assert spec.validate(np.int64(2)) == 2
    np.testing.assert_array_equal(
        spec.encode(np.array([0, 2, 1])),
        np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
    )
    assert spec.contains(1)
    assert not spec.contains(True)
    assert not spec.contains(3)
    with pytest.raises(ValueError):
        spec.encode(np.array([0.5]))


def test_continuous_action_spec_validates_and_normalizes() -> None:
    spec = ActionSpec.continuous(-2.0, 2.0)

    assert spec.kind == "continuous"
    assert spec.shape == (1,)
    assert spec.feature_dim == 1
    np.testing.assert_array_equal(spec.low, np.array([-2.0], dtype=np.float32))
    np.testing.assert_array_equal(spec.high, np.array([2.0], dtype=np.float32))
    np.testing.assert_array_equal(spec.validate(1.0), np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(
        spec.encode(np.array([[-2.0], [0.0], [2.0]], dtype=np.float32)),
        np.array([[-1.0], [0.0], [1.0]], dtype=np.float32),
    )
    assert not spec.contains([2.1])
    assert not spec.contains([np.nan])


@pytest.mark.parametrize(
    ("name", "expected_type", "state_dim", "observation_dim", "action"),
    [
        ("masked_cartpole", MaskedCartPoleEnv, 4, 2, 0),
        ("Pendulum-v1", MaskedPendulumEnv, 2, 2, [0.0]),
        (
            "masked-mountain-car-continuous",
            MaskedMountainCarContinuousEnv,
            2,
            1,
            np.array([0.0]),
        ),
    ],
)
def test_factory_and_common_step_api(
    name: str,
    expected_type: type,
    state_dim: int,
    observation_dim: int,
    action: object,
) -> None:
    env = make_env(name, {"horizon": 5, "observation_noise_std": 0.0}, seed=7)
    observation, info = env.reset()

    assert isinstance(env, expected_type)
    assert env.state_dim == state_dim
    assert env.observation_dim == observation_dim
    assert env.horizon == 5
    assert observation.shape == (observation_dim,)
    assert observation.dtype == np.float32
    assert info["state"].shape == (state_dim,)
    result = env.step(action)
    assert len(result) == 5
    next_observation, reward, terminated, truncated, next_info = result
    assert next_observation.shape == (observation_dim,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert next_info["elapsed_steps"] == 1


def test_factory_constructs_configurable_nd_light_dark() -> None:
    env = make_env(
        "Light-Dark-ND",
        {"horizon": 6, "observation_noise_std": 0.05, "dimension": 5},
        seed=7,
    )
    observation, info = env.reset()

    assert isinstance(env, LightDarkNDEnv)
    assert env.state_dim == env.observation_dim == env.oracle_dim == 5
    assert env.action_spec.shape == (5,)
    assert observation.shape == (5,)
    assert info["state"].shape == (5,)
    assert len(env.step(np.zeros(5, dtype=np.float32))) == 5


@pytest.mark.parametrize(
    "constructor",
    [
        MaskedCartPoleEnv,
        MaskedPendulumEnv,
        MaskedMountainCarContinuousEnv,
        LightDarkNDEnv,
    ],
)
def test_reset_seed_reproduces_state_and_noisy_observation(constructor: type) -> None:
    env = constructor(observation_noise_std=0.2)

    observation_a, info_a = env.reset(seed=1234)
    action = 0 if env.action_spec.is_discrete else np.zeros(env.action_spec.shape)
    env.step(action)
    observation_b, info_b = env.reset(seed=1234)

    np.testing.assert_array_equal(observation_a, observation_b)
    np.testing.assert_array_equal(info_a["state"], info_b["state"])


def test_info_and_state_properties_are_defensive_copies() -> None:
    env = MaskedCartPoleEnv(observation_noise_std=0.0)
    _, info = env.reset(seed=11)
    expected_state = env.state

    info["state"][:] = 999.0
    public_state = env.state
    public_state[:] = -999.0

    np.testing.assert_array_equal(env.state, expected_state)


def test_cartpole_observation_masks_velocities_and_is_normalized() -> None:
    env = MaskedCartPoleEnv(observation_noise_std=0.0)
    observation, info = env.reset(seed=2)
    state = info["state"]

    np.testing.assert_allclose(
        observation,
        [state[0] / env.x_threshold, state[2] / env.theta_threshold_radians],
        rtol=1e-6,
    )


def test_cartpole_uses_standard_euler_dynamics() -> None:
    env = MaskedCartPoleEnv(observation_noise_std=0.0)
    env.reset(seed=0)
    env._state = np.zeros(4, dtype=np.float64)

    _, reward, terminated, truncated, info = env.step(1)

    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert info["state"][0] == pytest.approx(0.0)
    assert info["state"][1] > 0.0
    assert info["state"][2] == pytest.approx(0.0)
    assert info["state"][3] < 0.0


def test_pendulum_observation_and_known_transition() -> None:
    env = MaskedPendulumEnv(observation_noise_std=0.0)
    env.reset(seed=0)
    env._state = np.zeros(2, dtype=np.float64)

    observation, reward, terminated, truncated, info = env.step([2.0])

    np.testing.assert_allclose(info["state"], [0.015, 0.3], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        observation,
        [np.cos(info["state"][0]), np.sin(info["state"][0])],
        rtol=1e-6,
    )
    assert reward == pytest.approx(-0.004)
    assert not terminated
    assert not truncated


def test_mountain_car_observation_and_known_transition() -> None:
    env = MaskedMountainCarContinuousEnv(observation_noise_std=0.0)
    env.reset(seed=0)
    env._state = np.array([-0.5, 0.0], dtype=np.float64)
    expected_velocity = 0.0015 - 0.0025 * np.cos(-1.5)
    expected_position = -0.5 + expected_velocity

    observation, reward, terminated, truncated, info = env.step([1.0])

    np.testing.assert_allclose(info["state"], [expected_position, expected_velocity])
    expected_observation = 2.0 * (expected_position + 1.2) / 1.8 - 1.0
    np.testing.assert_allclose(observation, [expected_observation], rtol=1e-6)
    assert reward == pytest.approx(-0.1)
    assert not terminated
    assert not truncated


def test_light_dark_nd_observation_transition_and_quadratic_reward() -> None:
    env = LightDarkNDEnv(dimension=3, observation_noise_std=0.0)
    env.reset(seed=0)
    env._state = np.array([4.0, 1.0, -2.0], dtype=np.float64)

    observation, reward, terminated, truncated, info = env.step([1.0, -1.0, 1.0])

    np.testing.assert_allclose(info["state"], [5.0, 0.0, -1.0])
    np.testing.assert_allclose(observation, info["state"])
    assert reward == pytest.approx(-12.0)
    assert info["effective_observation_std"] == pytest.approx(0.0)
    assert info["distance_to_goal"] == pytest.approx(np.sqrt(26.0))
    assert not terminated
    assert not truncated


def test_light_dark_default_reward_is_the_pure_quadratic_cost_sequence() -> None:
    """Lock the campaign reward: no terminal bonus, cost on the pre-transition state."""

    environment = make_env(
        "light_dark",
        {"horizon": 30, "observation_noise_std": 0.05, "dimension": 5},
        seed=4242,
    )
    _, info = environment.reset(seed=4242)
    actions = np.random.default_rng(7).uniform(-1.0, 1.0, size=(30, 5)).astype(np.float32)

    state = info["state"]
    rewards: list[float] = []
    terminated = truncated = False
    for action in actions:
        expected = -(0.5 * float(np.dot(state, state)) + 0.5 * float(np.dot(action, action)))
        _, reward, terminated, truncated, info = environment.step(action)
        assert reward == expected
        assert not terminated
        rewards.append(reward)
        state = info["state"]

    assert truncated
    assert len(rewards) == 30
    # Regression anchors recorded from the implementation that predates
    # ``goal_bonus``; the defaults must stay bit-identical to the campaign.
    assert rewards[:3] == [
        -7.470412862739259,
        -9.483487681116621,
        -10.046177127631426,
    ]
    assert rewards[-1] == -22.715076660727107


def test_light_dark_goal_bonus_is_added_only_on_the_terminating_step() -> None:
    plain = LightDarkNDEnv(dimension=2, horizon=6, observation_noise_std=0.0)
    rewarded = LightDarkNDEnv(
        dimension=2,
        horizon=6,
        observation_noise_std=0.0,
        goal_bonus=50.0,
    )
    for environment in (plain, rewarded):
        environment.reset(seed=1)
        environment._state = np.array([2.0, 0.0], dtype=np.float64)

    approach_plain = plain.step([-1.0, 0.0])
    approach_rewarded = rewarded.step([-1.0, 0.0])
    arrival_plain = plain.step([-1.0, 0.0])
    arrival_rewarded = rewarded.step([-1.0, 0.0])

    assert plain.goal_bonus == 0.0
    assert approach_plain[1] == approach_rewarded[1] == pytest.approx(-2.5)
    assert not approach_plain[2] and not approach_rewarded[2]
    assert arrival_plain[2] and arrival_rewarded[2]
    assert arrival_plain[1] == pytest.approx(-1.0)
    assert arrival_rewarded[1] == pytest.approx(-1.0 + 50.0)
    # The bonus is pure reward shaping: dynamics, observations and the RNG
    # stream are untouched.
    np.testing.assert_array_equal(approach_plain[0], approach_rewarded[0])
    np.testing.assert_array_equal(arrival_plain[4]["state"], arrival_rewarded[4]["state"])


def test_light_dark_goal_bonus_is_not_paid_on_horizon_truncation() -> None:
    environment = make_env(
        "light_dark",
        {
            "horizon": 1,
            "observation_noise_std": 0.0,
            "dimension": 2,
            "goal_bonus": 50.0,
        },
        seed=1,
    )
    environment.reset(seed=1)
    environment._state = np.array([2.0, 0.0], dtype=np.float64)

    _, reward, terminated, truncated, _ = environment.step([-1.0, 0.0])

    assert environment.goal_bonus == 50.0
    assert reward == pytest.approx(-2.5)
    assert not terminated
    assert truncated


def test_light_dark_observations_are_brightest_on_light_hyperplane() -> None:
    env = LightDarkNDEnv(dimension=4, observation_noise_std=0.2)
    env.reset(seed=0)
    env._state = np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float64)
    bright_std = env.observation_std()
    env._state = np.zeros(4, dtype=np.float64)
    dark_std = env.observation_std()

    assert bright_std == pytest.approx(0.2)
    assert dark_std == pytest.approx(np.sqrt(12.5 + 0.2**2))
    assert dark_std > bright_std


def test_light_dark_goal_termination_and_nd_action_validation() -> None:
    env = LightDarkNDEnv(dimension=4, goal_radius=0.25)
    env.reset(seed=0)
    env._state = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float64)

    _, _, terminated, truncated, _ = env.step(np.zeros(4))

    assert terminated
    assert not truncated
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4))

    env.reset(seed=1)
    with pytest.raises(ValueError, match="shape"):
        env.step([0.0])


def test_horizon_truncates_and_requires_reset() -> None:
    env = MaskedPendulumEnv(horizon=2, observation_noise_std=0.0)
    env.reset(seed=5)

    assert env.step([0.0])[3] is False
    assert env.step([0.0])[3] is True
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0.0])


@pytest.mark.parametrize(
    ("name", "config", "action"),
    [
        ("masked_cartpole", {"horizon": 20, "observation_noise_std": 0.2}, 1),
        ("masked_pendulum", {"horizon": 20, "observation_noise_std": 0.2}, [0.1]),
        (
            "masked_mountain_car_continuous",
            {"horizon": 20, "observation_noise_std": 0.2},
            [0.1],
        ),
        (
            "light_dark",
            {"horizon": 20, "observation_noise_std": 0.2, "dimension": 4},
            [0.1, 0.0, -0.1, 0.0],
        ),
    ],
)
def test_environment_state_dict_restores_exact_transition_and_generator(
    name: str,
    config: dict[str, object],
    action: object,
) -> None:
    environment = make_env(name, config, seed=17)
    environment.reset(seed=17)
    environment.step(action)
    saved = environment.state_dict()
    expected = environment.step(action)

    restored = make_env(name, config, seed=999)
    restored.load_state_dict(saved)
    actual = restored.step(action)

    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:4] == expected[1:4]
    np.testing.assert_array_equal(actual[4]["state"], expected[4]["state"])
    assert actual[4]["elapsed_steps"] == expected[4]["elapsed_steps"]

    assert isinstance(saved["state"], np.ndarray)
    saved["state"][:] = 999.0
    assert not np.all(restored.state == 999.0)


def test_environment_state_dict_rejects_a_different_simulator_type() -> None:
    source = MaskedCartPoleEnv()
    source.reset(seed=0)
    target = MaskedPendulumEnv()

    with pytest.raises(ValueError, match="type mismatch"):
        target.load_state_dict(source.state_dict())


@pytest.mark.parametrize(
    ("name", "expected_dim"),
    [
        ("masked_cartpole", 4),
        ("masked_pendulum", 3),
        ("masked_mountain_car_continuous", 2),
        ("light_dark", 5),
    ],
)
def test_oracle_features_are_normalized_noise_free_full_state(
    name: str,
    expected_dim: int,
) -> None:
    task_config: dict[str, float | int] = {
        "horizon": 8,
        "observation_noise_std": 10.0,
    }
    if name == "light_dark":
        task_config["dimension"] = 5
    environment = make_env(
        name,
        task_config,
        seed=13,
    )
    environment.reset(seed=13)
    first = environment.oracle_features()
    second = environment.oracle_features()

    assert first.shape == (expected_dim,)
    assert np.isfinite(first).all()
    assert np.max(np.abs(first)) <= 1.0 + 1e-6
    np.testing.assert_array_equal(first, second)


def test_invalid_config_name_and_actions_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown environment"):
        make_env("not-a-task")
    with pytest.raises(ValueError, match="Unsupported environment config"):
        make_env("cartpole", {"typo_noise": 0.1})

    discrete_env = MaskedCartPoleEnv()
    discrete_env.reset(seed=0)
    with pytest.raises(ValueError):
        discrete_env.step(2)

    continuous_env = MaskedPendulumEnv()
    continuous_env.reset(seed=0)
    with pytest.raises(ValueError):
        continuous_env.step([3.0])

    with pytest.raises(ValueError, match="dimension"):
        LightDarkNDEnv(dimension=0)
    with pytest.raises(ValueError, match="dimension"):
        LightDarkNDEnv(dimension=True)
    with pytest.raises(ValueError, match="goal_bonus must be non-negative"):
        LightDarkNDEnv(goal_bonus=-1.0)
    with pytest.raises(ValueError, match="must be finite"):
        LightDarkNDEnv(goal_bonus=float("inf"))


def test_light_dark_noise_gain_scales_the_observation_noise() -> None:
    default = LightDarkNDEnv(dimension=2, seed=0)
    default.reset(seed=0)
    darker = LightDarkNDEnv(dimension=2, noise_gain=5.0, seed=0)
    darker.reset(seed=0)
    # Same state (same seed), ten times the variance slope away from the light.
    np.testing.assert_allclose(darker._state, default._state)
    x1 = float(default._state[0])
    expected_default = np.sqrt(0.5 * (5.0 - x1) ** 2 + 0.05**2)
    expected_darker = np.sqrt(5.0 * (5.0 - x1) ** 2 + 0.05**2)
    assert default.observation_std() == pytest.approx(expected_default)
    assert darker.observation_std() == pytest.approx(expected_darker)


def test_light_dark_terminal_cost_applies_only_at_the_horizon() -> None:
    plain = LightDarkNDEnv(dimension=2, horizon=3, seed=1)
    graded = LightDarkNDEnv(dimension=2, horizon=3, terminal_cost=10.0, seed=1)
    plain.reset(seed=1)
    graded.reset(seed=1)
    action = np.zeros(2)
    for step in range(3):
        _, plain_reward, _, plain_truncated, _ = plain.step(action)
        _, graded_reward, _, graded_truncated, _ = graded.step(action)
        assert plain_truncated == graded_truncated
        if not graded_truncated:
            assert graded_reward == plain_reward
        else:
            final_state = graded._state
            assert graded_reward == pytest.approx(
                plain_reward - 10.0 * float(np.dot(final_state, final_state))
            )


def test_light_dark_process_noise_is_seeded_and_off_by_default() -> None:
    quiet = LightDarkNDEnv(dimension=2, seed=2)
    quiet.reset(seed=2)
    start = quiet._state.copy()
    quiet.step(np.array([0.5, -0.5]))
    np.testing.assert_allclose(quiet._state, start + np.array([0.5, -0.5]))

    noisy_a = LightDarkNDEnv(dimension=2, process_noise_std=0.1, seed=2)
    noisy_b = LightDarkNDEnv(dimension=2, process_noise_std=0.1, seed=2)
    noisy_a.reset(seed=2)
    noisy_b.reset(seed=2)
    noisy_a.step(np.array([0.5, -0.5]))
    noisy_b.step(np.array([0.5, -0.5]))
    np.testing.assert_allclose(noisy_a._state, noisy_b._state)
    assert not np.allclose(noisy_a._state, start + np.array([0.5, -0.5]))


def test_light_dark_rejects_negative_p3o_knobs() -> None:
    for kwargs in (
        {"noise_gain": -0.1},
        {"terminal_cost": -1.0},
        {"process_noise_std": -0.5},
    ):
        with pytest.raises(ValueError, match="non-negative"):
            LightDarkNDEnv(dimension=2, **kwargs)
