"""Memory tasks taken from the literature (experiment 29): bsuite memory length and the POPGym noisy pendulum clip."""

from __future__ import annotations

import numpy as np
import pytest

from sb_pomdp.config import ConfigError, load_config, validate_config
from sb_pomdp.envs import MaskedPendulumEnv, MemoryChainEnv, make_env


@pytest.mark.parametrize("length", [1, 2, 10, 30])
def test_memory_chain_shows_the_context_once_and_pays_only_the_last_action(length: int) -> None:
    env = MemoryChainEnv(horizon=length, seed=0)
    contexts = set()
    for seed in range(20):
        observation, info = env.reset(seed=seed)
        context = float(info["state"][0])
        contexts.add(context)
        assert observation.tolist() == [1.0, 0.0, context]
        answer = 1 if context > 0 else 0
        for step in range(length):
            last = step == length - 1
            action = answer if last else 1 - answer  # earlier actions do not matter
            observation, reward, terminated, truncated, info = env.step(action)
            assert reward == (1.0 if last else 0.0)
            assert (terminated, truncated) == (last, False)
            assert observation[2] == 0.0 and observation[1] == 0.0
            assert observation[0] == pytest.approx(1.0 - (step + 1) / length)
        assert env.needs_reset
    assert contexts == {-1.0, 1.0}


def test_memory_chain_wrong_answer_costs_one_and_a_blind_guess_averages_zero() -> None:
    env = MemoryChainEnv(horizon=5, seed=1)
    totals = []
    for seed in range(400):
        env.reset(seed=seed)
        total = 0.0
        for _ in range(5):
            _, reward, terminated, _, _ = env.step(1)  # always answer "+1"
            total += reward
        assert terminated and total in (-1.0, 1.0)
        totals.append(total)
    assert abs(np.mean(totals)) < 0.15


def test_memory_chain_observations_after_the_first_do_not_depend_on_the_context() -> None:
    plus, minus = MemoryChainEnv(horizon=6), MemoryChainEnv(horizon=6)
    for env, context in ((plus, 1.0), (minus, -1.0)):
        env.reset(seed=0)
        env._state[0] = context
    for _ in range(5):
        a, *_ = plus.step(0)
        b, *_ = minus.step(0)
        np.testing.assert_array_equal(a, b)


def test_memory_chain_round_trips_its_state_and_is_built_by_name() -> None:
    env = make_env("memory_chain", {"horizon": 10, "observation_noise_std": 0.0}, seed=3)
    assert isinstance(env, MemoryChainEnv) and env.action_spec.is_discrete and env.action_spec.n == 2
    assert (env.state_dim, env.observation_dim, env.horizon) == (2, 3, 10)
    env.reset(seed=4)
    env.step(0)
    snapshot = env.state_dict()
    first, *_ = env.step(1)
    env.load_state_dict(snapshot)
    second, *_ = env.step(1)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(env.oracle_features(), env.state.astype(np.float32))


def test_pendulum_observation_clip_is_off_by_default_and_bounds_the_noisy_observation() -> None:
    kwargs = dict(horizon=200, observation_noise_std=0.3, seed=5)
    plain, same, clipped = MaskedPendulumEnv(**kwargs), MaskedPendulumEnv(**kwargs, observation_clip=False), MaskedPendulumEnv(
        **kwargs, observation_clip=True
    )
    rows = {"plain": [], "same": [], "clipped": []}
    for name, env in (("plain", plain), ("same", same), ("clipped", clipped)):
        observation, _ = env.reset(seed=6)
        rows[name].append(observation)
        for _ in range(199):
            observation, *_ = env.step(np.array([0.5], dtype=np.float32))
            rows[name].append(observation)
    plain_rows, same_rows, clipped_rows = (np.stack(rows[name]) for name in ("plain", "same", "clipped"))
    np.testing.assert_array_equal(plain_rows, same_rows)
    assert np.abs(plain_rows).max() > 1.0  # the unclipped noisy observation leaves [-1, 1]
    np.testing.assert_array_equal(clipped_rows, np.clip(plain_rows, -1.0, 1.0))
    assert clipped_rows.dtype == np.float32


def test_pendulum_configs_without_the_clip_key_resolve_to_false() -> None:
    resolved = load_config("config/core4_pendbal.json").to_dict()
    task = resolved["environment"]["tasks"]["masked_pendulum"]
    assert task["observation_clip"] is False
    assert make_env("masked_pendulum", task, seed=0).observation_clip is False
    task["observation_clip"] = "yes"
    with pytest.raises(ConfigError):
        validate_config(resolved)
