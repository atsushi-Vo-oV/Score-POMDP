"""Resuming a Minecraft run restarts the episodes the world cannot continue.

The Minecraft world is not restorable from a checkpoint: ``MinecraftWalkEnv.load_state_dict``
leaves ``needs_reset`` set.  Both trainers must then reset those environments at resume,
drop the interrupted episode from the per-environment accumulators, and carry on from the
committed update.  The fake backend stands in for the server.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.config import ExperimentConfig, load_config
from sb_pomdp.envs import make_env
from sb_pomdp.train import reset_unresumable_environments, train_single
from sb_pomdp.train_baselines import train_baseline_single

TASK = "minecraft_walk"
HORIZON = 20  # longer than two rollouts, so the segment boundary falls inside an episode
ROLLOUT = 4


def _config(tmp_path: Path) -> ExperimentConfig:
    data = load_config("config/local.json").to_dict()
    data["environment"]["tasks"][TASK] = {"horizon": HORIZON, "observation_noise_std": 0.0, "backend": "fake"}
    data["experiment"]["tasks"] = [TASK]
    return ExperimentConfig(data).with_overrides(
        {
            "experiment.output_dir": str(tmp_path / "unused-output-root"),
            "experiment.eval_episodes": 1,
            "experiment.eval_interval_updates": 999,
            "experiment.checkpoint_interval_updates": 999,
            "model.num_particles": 2,
            "model.energy_hidden": [4],
            "model.d_model": 4,
            "model.num_heads": 1,
            "model.transformer_ff_dim": 8,
            "model.policy_hidden": [4],
            "comparison.gru_encoder_hidden": [4],
            "comparison.gru_hidden_dim": 4,
            "comparison.pf_hidden": [4],
            "comparison.pf_particle_dim": 3,
            "ppo.total_steps": 2 * ROLLOUT,
            "ppo.num_envs": 1,
            "ppo.rollout_steps": ROLLOUT,
            "ppo.epochs": 1,
            "ppo.minibatch_size": ROLLOUT,
            "ppo.sequence_microbatch_size": ROLLOUT,
        }
    )


def _artifacts(path: Path, method: str | None = None) -> SeedArtifacts:
    path.mkdir(parents=True)
    return SeedArtifacts(path=path, task=TASK, seed=3, method=method)


def _latest(artifacts: SeedArtifacts) -> dict:
    return torch.load(artifacts.checkpoint_path(tag="latest", create_parent=False), map_location="cpu", weights_only=False)


def test_needs_reset_tracks_the_episode_and_the_unrestorable_world() -> None:
    env = make_env(TASK, {"horizon": HORIZON, "observation_noise_std": 0.0, "backend": "fake"}, seed=3)
    assert env.needs_reset
    env.reset(seed=3)
    assert not env.needs_reset
    env.step(1)
    saved = env.state_dict()
    assert saved["needs_reset"] is False

    other = make_env(TASK, {"horizon": HORIZON, "observation_noise_std": 0.0, "backend": "fake"}, seed=4)
    other.load_state_dict(saved)
    assert other.needs_reset  # the world is gone even though the bookkeeping was restored
    restarted = reset_unresumable_environments([env, other])
    assert [index for index, _ in restarted] == [1]
    assert restarted[0][1].shape == (other.observation_dim,)
    assert not other.needs_reset
    other.step(1)

    # a simulator restores exactly and keeps going
    sim = make_env("masked_cartpole", {"horizon": HORIZON}, seed=3)
    sim.reset(seed=3)
    sim.step(0)
    twin = make_env("masked_cartpole", {"horizon": HORIZON}, seed=5)
    twin.load_state_dict(sim.state_dict())
    assert not twin.needs_reset
    assert reset_unresumable_environments([sim, twin]) == []


def test_score_trainer_resume_restarts_the_minecraft_episode(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = _config(tmp_path)
    artifacts = _artifacts(tmp_path / "s")
    first = train_single(config, task=TASK, seed=3, artifacts=artifacts, segment_updates=1, verbose=False)
    assert first["status"] == "partial" and first["updates"] == 1
    saved = _latest(artifacts)
    assert saved["trainer_state"]["episode_lengths"].tolist() == [ROLLOUT]
    assert len(saved["trainer_state"]["belief_histories"][0]["observations"]) == ROLLOUT

    with caplog.at_level(logging.WARNING, logger="sb_pomdp"):
        resumed = train_single(
            config,
            task=TASK,
            seed=3,
            artifacts=artifacts,
            max_updates=2,
            resume_checkpoint=artifacts.checkpoint_path(tag="latest", create_parent=False),
            evaluate_at_end=False,
            verbose=False,
        )
    assert resumed["status"] == "completed" and resumed["updates"] == 2
    assert any("cannot continue its interrupted episode" in record.getMessage() for record in caplog.records)

    final = _latest(artifacts)
    assert final["update"] == 2
    # the new episode started at the resume point: one rollout long, not two
    assert final["trainer_state"]["episode_lengths"].tolist() == [ROLLOUT]
    assert final["trainer_state"]["environments"][0]["elapsed_steps"] == ROLLOUT
    assert len(final["trainer_state"]["belief_histories"][0]["observations"]) == ROLLOUT
    assert bool(final["trainer_state"]["belief_histories"][0]["initial"][0])
    assert final["trainer_state"]["completed_returns"] == []


@pytest.mark.parametrize("method", ["gru", "rnn", "particle_filter"])
def test_baseline_resume_restarts_the_minecraft_episode(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, method: str
) -> None:
    config = _config(tmp_path)
    artifacts = _artifacts(tmp_path / "b", method)
    first = train_baseline_single(
        config, method=method, task=TASK, seed=3, artifacts=artifacts, segment_updates=1, verbose=False,
        result_dir=f"{method}/{TASK}/seed_003",
    )
    assert first["status"] == "partial" and first["updates"] == 1
    saved = _latest(artifacts)
    assert saved["trainer_state"]["episode_lengths"].tolist() == [ROLLOUT]
    assert len(saved["trainer_state"]["recurrent_histories"][0]["inputs"]) == ROLLOUT

    with caplog.at_level(logging.WARNING, logger="sb_pomdp"):
        resumed = train_baseline_single(
            config,
            method=method,
            task=TASK,
            seed=3,
            artifacts=artifacts,
            max_updates=2,
            resume_checkpoint=artifacts.checkpoint_path(tag="latest", create_parent=False),
            evaluate_at_end=False,
            verbose=False,
            result_dir=f"{method}/{TASK}/seed_003",
        )
    assert resumed["status"] == "completed" and resumed["updates"] == 2
    assert any("cannot continue its interrupted episode" in record.getMessage() for record in caplog.records)

    final = _latest(artifacts)
    assert final["update"] == 2
    assert final["trainer_state"]["episode_lengths"].tolist() == [ROLLOUT]
    assert final["trainer_state"]["environments"][0]["elapsed_steps"] == ROLLOUT
    assert len(final["trainer_state"]["recurrent_histories"][0]["inputs"]) == ROLLOUT
    assert bool(final["trainer_state"]["recurrent_histories"][0]["episode_starts"][0])
    assert final["trainer_state"]["completed_returns"] == []
