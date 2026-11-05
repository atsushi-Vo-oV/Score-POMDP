"""DiskTagEnv: evade wandering, seeing ghosts that the player observes only as a count in its view cone."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sb_pomdp.config import ConfigError, load_config, normalize_legacy_resolved_config, validate_config
from sb_pomdp.envs import DiskTagEnv, disk_tag_ghost_step, make_env


def _place(env: DiskTagEnv, player, phi, ghosts, headings, modes=None, speed=0.0) -> None:
    ghosts = np.asarray(ghosts, dtype=np.float64)
    count = len(ghosts)
    modes = np.zeros(count) if modes is None else np.asarray(modes, dtype=np.float64)
    block = np.concatenate([ghosts, np.asarray(headings)[:, None], modes[:, None], ghosts], axis=1)
    speeds = np.full(count, env.max_speed)
    env._state = np.concatenate([player, [phi, speed], block.ravel(), speeds]).astype(np.float64)


def test_reset_places_everything_in_the_disk_and_the_player_away_from_the_ghosts() -> None:
    env = DiskTagEnv(seed=0)
    for seed in range(50):
        observation, info = env.reset(seed=seed)
        state = info["state"]
        assert state.shape == (env.state_dim,) == (4 + 7 * env.num_ghosts,)
        assert observation.shape == (6,) and observation[4] == 0.0
        ghosts = env._ghosts()
        assert np.all(np.linalg.norm(ghosts, axis=1) <= env.arena_radius)
        assert np.all(np.linalg.norm(ghosts - state[:2], axis=1) >= env.start_distance)
        assert np.all(env._ghost_block()[:, 3] == 0.0)  # every ghost starts wandering
        speeds = env._speeds() / env.max_speed
        assert np.all((speeds >= env.ghost_speed_min) & (speeds <= env.ghost_speed_max))


def test_the_count_sees_only_ghosts_inside_the_player_cone() -> None:
    env = DiskTagEnv(seed=0)
    env.reset(seed=0)
    ahead, aside, far = [0.3, 0.0], [0.0, 0.3], [0.0, 0.0 + env.view_range + 0.05]
    _place(env, [0.0, 0.0], 0.0, [ahead, aside, far], [0.0, 0.0, 0.0])
    assert env.visible_count() == 1
    _place(env, [0.0, 0.0], math.pi / 2, [ahead, aside, [0.0, 0.5]], [0.0, 0.0, 0.0])
    assert env.visible_count() == 2


def test_a_ghost_pursues_once_it_sees_the_player_and_searches_after_losing_sight() -> None:
    env = DiskTagEnv(seed=0, num_ghosts=1)
    params = env.ghost_params()
    speed = np.array([env.max_speed])
    ghost = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # facing +x, wandering
    seen = disk_tag_ghost_step(ghost, speed, np.array([0.5, 0.0]), np.zeros(1), **params)
    assert seen[0, 3] == 1.0 and seen[0, 0] == pytest.approx(env.max_speed) and seen[0, 4] == 0.5
    behind = disk_tag_ghost_step(ghost, speed, np.array([-0.5, 0.0]), np.zeros(1), **params)
    assert behind[0, 3] == 0.0 and behind[0, 0] == pytest.approx(env.ghost_wander_ratio * env.max_speed)
    # pursuing ghost that cannot see the player runs to the remembered spot, then wanders again
    lost = np.array([[0.0, 0.0, math.pi, 1.0, 0.03, 0.0]])
    arrived = disk_tag_ghost_step(lost, speed, np.array([0.9, 0.0]), np.zeros(1), **params)
    assert arrived[0, 0] == pytest.approx(0.03) and arrived[0, 3] == 0.0


def test_the_ghost_step_is_batched_like_the_simulator() -> None:
    env = DiskTagEnv(seed=3)
    env.reset(seed=4)
    block, speeds = env._ghost_block().copy(), env._speeds().copy()
    player = env.state[:2] + 0.01
    noise = np.random.default_rng(0).standard_normal((5, env.num_ghosts))
    batched = disk_tag_ghost_step(np.broadcast_to(block, (5, *block.shape)), np.broadcast_to(speeds, (5, len(speeds))),
                                  np.broadcast_to(player, (5, 2)), noise, **env.ghost_params())
    for row in range(5):
        single = disk_tag_ghost_step(block, speeds, player, noise[row], **env.ghost_params())
        assert np.allclose(batched[row], single)
    assert np.all(np.linalg.norm(batched[..., 0:2], axis=-1) <= env.arena_radius + 1e-12)


def test_capture_ends_the_episode_and_survival_pays_per_step() -> None:
    env = DiskTagEnv(seed=0, num_ghosts=1)
    env.reset(seed=0)
    _place(env, [0.0, 0.0], 0.0, [[-0.5, 0.0]], [math.pi])  # ghost faces away, wanders off
    _, reward, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert reward == env.step_reward and not terminated
    _place(env, [0.0, 0.0], 0.0, [[env.capture_radius + 0.01, 0.0]], [math.pi])  # facing the player
    _, reward, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert terminated and reward == env.step_reward
    penalised = DiskTagEnv(seed=0, num_ghosts=1, capture_reward=-1.0)
    penalised.reset(seed=0)
    _place(penalised, [0.0, 0.0], 0.0, [[penalised.capture_radius + 0.01, 0.0]], [math.pi])
    _, reward, terminated, _, _ = penalised.step(np.zeros(2, dtype=np.float32))
    assert terminated and reward == pytest.approx(penalised.step_reward - 1.0)


def test_the_player_stays_in_the_disk_and_round_trips_its_state() -> None:
    env = DiskTagEnv(seed=1, horizon=400, num_ghosts=2)
    env.reset(seed=2)
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 0.05], dtype=np.float32))
        assert np.linalg.norm(info["state"][:2]) <= env.arena_radius + 1e-12
        if terminated or truncated:
            break
    env.reset(seed=5)
    env.step(np.array([0.5, -0.5], dtype=np.float32))
    saved = env.state_dict()
    after = env.step(np.array([1.0, 1.0], dtype=np.float32))[0]
    env.load_state_dict(saved)
    assert np.array_equal(env.step(np.array([1.0, 1.0], dtype=np.float32))[0], after)


def test_make_env_and_config_defaults() -> None:
    env = make_env("disk_tag", {"horizon": 200, "observation_noise_std": 0.0, "num_ghosts": 2})
    assert isinstance(env, DiskTagEnv) and env.state_dim == 18 and env.oracle_dim == 13
    resolved = load_config("config/core4_tps.json").to_dict()
    resolved["experiment"]["tasks"] = ["disk_tag"]
    resolved["environment"]["tasks"] = {"disk_tag": {"horizon": 200, "observation_noise_std": 0.0}}
    resolved = normalize_legacy_resolved_config(resolved)
    validate_config(resolved)
    filled = resolved["environment"]["tasks"]["disk_tag"]
    assert filled["num_ghosts"] == 3 and filled["ghost_view_angle_deg"] == 180.0
    filled["num_ghosts"] = 1.5
    with pytest.raises(ConfigError):
        validate_config(resolved)


def test_the_score_belief_latent_size_is_a_model_choice() -> None:
    import torch

    from sb_pomdp.train import belief_latent_dim, model_for_environment

    env = make_env("disk_tag", {"horizon": 200, "observation_noise_std": 0.0}, seed=0)
    resolved = load_config("config/core4_tps.json").to_dict()
    assert belief_latent_dim(env, resolved["model"]) == env.state_dim == 25  # legacy: simulator state size
    model_config = dict(resolved["model"], belief_latent_dim=6, langevin_warm_start=False)
    model = model_for_environment(env, model_config, torch.device("cpu"))
    assert model.belief.state_dim == 6 and model.encoder.state_dim == 6


def test_range_observation_reports_every_ghost_distance_with_growing_noise() -> None:
    env = DiskTagEnv(seed=0, observation_mode="ranges", range_noise_base=0.05, range_noise_scale=0.2)
    assert env.observation_dim == 8
    errors, distances = [], []
    for seed in range(300):
        observation, info = env.reset(seed=seed)
        true = np.linalg.norm(env._ghosts() - info["state"][:2], axis=1)
        errors.append(observation[5:] - true)
        distances.append(true)
    errors, distances = np.concatenate(errors), np.concatenate(distances)
    assert abs(errors.mean()) < 0.03
    near, far = distances < 0.8, distances > 1.2
    assert errors[near].std() < errors[far].std()  # sigma = 0.05 + 0.2 d
    assert errors.std() == pytest.approx(np.sqrt(np.mean((0.05 + 0.2 * distances) ** 2)), rel=0.15)
    count_env = DiskTagEnv(seed=0)
    assert count_env.observation_dim == 6 and count_env.observation_mode == "count"


def test_elliptic_ranges_depend_on_the_heading_and_reduce_to_ranges_at_unit_scales() -> None:
    env = DiskTagEnv(seed=0, observation_mode="elliptic_ranges", range_forward_scale=1.0, range_side_scale=0.5,
                     range_noise_base=0.0, range_noise_scale=0.0)
    assert env.observation_dim == 8
    env.reset(seed=0)
    ahead, aside, diagonal = [0.4, 0.0], [0.0, 0.4], [0.3, 0.3]
    _place(env, [0.0, 0.0], 0.0, [ahead, aside, diagonal], [0.0, 0.0, 0.0])
    observation = env._observation()
    # facing +x: the ghost ahead keeps its Euclidean distance, the one at the side is stretched by 1 / 0.5
    assert observation[5] == pytest.approx(0.4)
    assert observation[6] == pytest.approx(0.8)
    assert observation[7] == pytest.approx(math.sqrt(0.3**2 + (0.3 / 0.5) ** 2))
    # turning the body by 90 degrees swaps the roles: the same ghosts now read 0.8 / 0.4
    _place(env, [0.0, 0.0], math.pi / 2, [ahead, aside, diagonal], [0.0, 0.0, 0.0])
    turned = env._observation()
    assert turned[5] == pytest.approx(0.8) and turned[6] == pytest.approx(0.4)
    # the stretched value is consistent with an ellipse of semi-axes (a r, b r) aligned with the heading
    r = turned[5]
    assert ahead[0] == pytest.approx(env.range_side_scale * r)
    # unit scales recover the Euclidean ranges exactly (same noise draw, same seed)
    iso = DiskTagEnv(seed=0, observation_mode="ranges")
    ell = DiskTagEnv(seed=0, observation_mode="elliptic_ranges")
    assert np.array_equal(iso.reset(seed=3)[0], ell.reset(seed=3)[0])
    with pytest.raises(ValueError):
        DiskTagEnv(seed=0, observation_mode="elliptic_ranges", range_side_scale=0.0)
    resolved = load_config("config/core4_tps.json").to_dict()
    resolved["experiment"]["tasks"] = ["disk_tag"]
    resolved["environment"]["tasks"] = {"disk_tag": {"horizon": 200, "observation_noise_std": 0.0,
                                                     "observation_mode": "elliptic_ranges", "range_side_scale": 0.6}}
    resolved = normalize_legacy_resolved_config(resolved)
    validate_config(resolved)
    filled = resolved["environment"]["tasks"]["disk_tag"]
    assert filled["range_forward_scale"] == 1.0 and filled["range_side_scale"] == 0.6
    filled["range_side_scale"] = 0.0
    with pytest.raises(ConfigError):
        validate_config(resolved)
