"""Car-Flag and Heaven-Hell: the information-gathering POMDPs from the asymmetric-RL literature."""

from __future__ import annotations

import numpy as np
import pytest

from sb_pomdp.config import ConfigError, load_config, validate_config
from sb_pomdp.envs import CarFlagEnv, HeavenHellEnv, make_env

N, S, E, W = 0, 1, 2, 3


def _drive(env: CarFlagEnv, force: float, limit: int = 500) -> tuple[float, int, bool]:
    total, steps = 0.0, 0
    while steps < limit:
        _, reward, terminated, truncated, _ = env.step(np.array([force], dtype=np.float32))
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps, terminated
    raise AssertionError("episode did not end")


def test_car_flag_reveals_the_side_only_near_the_blue_flag() -> None:
    env = CarFlagEnv(horizon=400, seed=0)
    for seed in range(20):
        observation, info = env.reset(seed=seed)
        side = info["state"][2]
        assert side in (-1.0, 1.0)
        assert observation.shape == (3,) and observation[2] == 0.0
        seen = set()
        for _ in range(60):  # full force right passes through [0.3, 0.7]
            observation, _, terminated, _, info = env.step(np.array([1.0], dtype=np.float32))
            position = info["state"][0]
            near = abs(position - env.priest_position) <= env.priest_range
            assert observation[2] == (side if near else 0.0)
            seen.add(float(observation[2]))
            if terminated:
                break
        assert side in seen


def test_car_flag_pays_the_green_flag_and_charges_the_red_one() -> None:
    env = CarFlagEnv(seed=1)
    outcomes = {}
    for seed in range(40):
        _, info = env.reset(seed=seed)
        side = info["state"][2]
        total, steps, terminated = _drive(env, 1.0)
        assert terminated and abs(env.state[0]) >= env.flag_position
        expected = (1.0 if side > 0 else -1.0) - env.step_cost * steps
        assert total == pytest.approx(expected)
        outcomes[side] = True
    assert set(outcomes) == {-1.0, 1.0}  # both layouts occur


def test_car_flag_truncates_at_the_horizon_and_round_trips_its_state() -> None:
    env = CarFlagEnv(horizon=5, seed=2)
    env.reset(seed=3)
    snapshot = env.state_dict()
    total, steps, terminated = _drive(env, 0.0)
    assert (steps, terminated) == (5, False) and total == pytest.approx(-0.05)
    env.load_state_dict(snapshot)
    assert env.state_dict()["elapsed_steps"] == 0


@pytest.mark.parametrize("n,symbols", [(3, 15), (4, 19)])
def test_heaven_hell_matches_the_published_sizes(n: int, symbols: int) -> None:
    env = HeavenHellEnv(corridor_length=n)
    assert env.observation_dim == symbols  # gym-pomdps heavenhell_3 / _4
    assert 2 * env.num_cells == {3: 28, 4: 36}[n]
    assert env.action_spec.is_discrete and env.action_spec.n == 4


def _priest_route(n: int, side: float) -> list[int]:
    return [S] + [E] * n + [W] * n + [N] + [N] * n + ([W] if side < 0 else [E]) * n


@pytest.mark.parametrize("n", [3, 4])
def test_heaven_hell_priest_route_reaches_heaven_in_4n_plus_2_steps(n: int) -> None:
    env = HeavenHellEnv(corridor_length=n)
    for seed in range(10):
        observation, info = env.reset(seed=seed)
        side = info["state"][2]
        rewards = []
        for step, action in enumerate(_priest_route(n, side)):
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if step == n:  # at the priest: the symbol is the heaven side
                assert int(np.argmax(observation)) == env.num_cells - 1 + (0 if side < 0 else 1)
        assert terminated and not truncated
        assert rewards[-1] == 1.0 and sum(rewards) == 1.0 and len(rewards) == 4 * n + 2


@pytest.mark.parametrize("n", [3, 4])
def test_heaven_hell_hides_the_side_everywhere_but_the_priest(n: int) -> None:
    left, right = HeavenHellEnv(corridor_length=n), HeavenHellEnv(corridor_length=n)
    for env, side in ((left, -1.0), (right, 1.0)):
        env.reset(seed=0)
        env._state[2] = side
    route = [N] * n + [S] * n + [S] + [E] * n
    for action in route[:-1]:
        a, *_ = left.step(action)
        b, *_ = right.step(action)
        np.testing.assert_array_equal(a, b)
    a, *_ = left.step(route[-1])
    b, *_ = right.step(route[-1])
    assert not np.array_equal(a, b)


def test_heaven_hell_blind_route_lands_in_hell_half_the_time() -> None:
    env = HeavenHellEnv(corridor_length=3)
    totals = []
    for seed in range(200):
        env.reset(seed=seed)
        for action in [N] * 3 + [W] * 3:
            _, reward, terminated, _, _ = env.step(action)
        assert terminated
        totals.append(reward)
    assert set(totals) == {-1.0, 1.0}
    assert abs(np.mean(totals)) < 0.2


def test_heaven_hell_walls_leave_the_agent_in_place() -> None:
    env = HeavenHellEnv(corridor_length=3)
    env.reset(seed=0)
    start = env.state.copy()
    for action in (E, W):
        env.step(action)
        np.testing.assert_array_equal(env.state, start)


def test_make_env_and_config_defaults_for_the_new_tasks() -> None:
    assert isinstance(make_env("car_flag", {"horizon": 160, "observation_noise_std": 0.0}), CarFlagEnv)
    env = make_env("heaven_hell", {"horizon": 100, "observation_noise_std": 0.0, "corridor_length": 4})
    assert isinstance(env, HeavenHellEnv) and env.corridor_length == 4
    for name in ("config/core5_carflag.json", "config/core5_heavenhell3.json", "config/core5_heavenhell4.json"):
        resolved = load_config(name).to_dict()
        (task,) = resolved["experiment"]["tasks"]
        built = make_env(task, resolved["environment"]["tasks"][task], seed=0)
        assert built.state_dim == 3


def test_config_rejects_a_bad_corridor_length() -> None:
    resolved = load_config("config/core5_heavenhell3.json").to_dict()
    resolved["environment"]["tasks"]["heaven_hell"]["corridor_length"] = 0
    with pytest.raises(ConfigError):
        validate_config(resolved)
