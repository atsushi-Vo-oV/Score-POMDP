"""TD3 on the belief models: exact off-policy belief replay from stored noise seeds."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sb_pomdp.config import load_config
from sb_pomdp.envs import make_env
from sb_pomdp.td3 import BeliefAdapter, SequenceReplay, _replay_conditions, _to_env_action
from sb_pomdp.train import belief_latent_dim, model_for_environment
from sb_pomdp.train_baselines import _make_model


def _setup(method: str):
    resolved = load_config("config/trysvgd_tps_tf-svgd.json" if method == "score_transformer"
                           else "config/trysvgd_tps_base.json").to_dict()
    env = make_env("mountain_hike", resolved["environment"]["tasks"]["mountain_hike"], seed=0)
    torch.manual_seed(0)
    if method == "score_transformer":
        model = model_for_environment(env, resolved["model"], torch.device("cpu"))
        latent = belief_latent_dim(env, resolved["model"])
    else:
        model = _make_model(env, method, resolved["model"], resolved["comparison"], torch.device("cpu"))
        latent = None
    adapter = BeliefAdapter(model.eval(), method, resolved["model"], torch.device("cpu"))
    return adapter, env, latent, resolved["environment"]["tasks"]["mountain_hike"]


@pytest.mark.parametrize("method", ["score_transformer", "gru", "particle_filter"])
def test_replayed_conditions_equal_the_online_conditions(method: str) -> None:
    adapter, env, latent, task_config = _setup(method)
    envs = [make_env("mountain_hike", task_config, seed=s) for s in (1, 2)]
    replay = SequenceReplay(2, 64, env.observation_dim, env.action_spec.feature_dim)
    rng = np.random.default_rng(0)
    observation = np.stack([e.reset(seed=3 + i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    previous = np.zeros((2, 2), dtype=np.float32)
    initial = np.ones(2, dtype=bool)
    state = adapter.initial_state(2, latent_dim=latent)
    online = []
    for t in range(30):
        seeds = rng.integers(1 << 62, size=2)
        with torch.no_grad():
            condition, state = adapter.step(state, torch.as_tensor(observation), torch.as_tensor(previous),
                                            torch.as_tensor(initial), seeds, track_grad=False)
        online.append(condition)
        action = rng.uniform(-1, 1, size=(2, 2)).astype(np.float32)
        env_action = _to_env_action(torch.as_tensor(action), env.action_spec)
        steps = [e.step(env_action[i]) for i, e in enumerate(envs)]
        done = np.array([t == 11, False])  # force an episode boundary in stream 0
        replay.add(observation=observation, previous_action=previous, initial=initial, action=action,
                   reward=np.zeros(2, np.float32), terminated=done, truncated=np.zeros(2, bool), seed=seeds)
        next_obs = np.stack([s[0] for s in steps]).astype(np.float32)
        if done[0]:
            next_obs[0] = envs[0].reset(seed=9)[0]
        observation, previous, initial = next_obs, np.where(done[:, None], 0.0, action).astype(np.float32), done
    sample = {"envs": np.array([0, 1, 0]), "starts": np.array([5, 17, 14]),
              "episode_start": np.array([0, 0, 12])}
    sample["index"] = sample["starts"][:, None] + np.arange(5)[None, :]
    with torch.no_grad():
        replayed = _replay_conditions(adapter, replay, sample, latent_dim=latent, device=torch.device("cpu"))
    for column, (env_index, start) in enumerate(zip(sample["envs"], sample["starts"], strict=True)):
        for k in range(5):
            assert torch.allclose(replayed[k, column], online[start + k][env_index], atol=1e-5)


def test_replay_marks_steps_outside_the_episode_invalid() -> None:
    replay = SequenceReplay(1, 16, 2, 2)
    for t in range(10):
        replay.add(observation=np.zeros((1, 2)), previous_action=np.zeros((1, 2)), initial=np.array([t in (0, 4)]),
                   action=np.zeros((1, 2)), reward=np.zeros(1), terminated=np.array([t == 3]),
                   truncated=np.zeros(1, bool), seed=np.zeros(1, np.int64))
    rng = np.random.default_rng(0)
    for _ in range(50):
        sample = replay.sample(rng, 4, 6)
        for row, start in enumerate(sample["starts"]):
            expected = [(start + k < 10) and ((start + k < 4) == (start < 4)) for k in range(7)]
            assert sample["valid"][row].tolist() == expected
