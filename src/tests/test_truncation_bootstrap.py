"""Opt-in truncation-aware value bootstrapping (``ppo.bootstrap_on_truncation``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from sb_pomdp import train as train_module
from sb_pomdp import train_baselines as train_baselines_module
from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.config import ExperimentConfig, load_config
from sb_pomdp.envs import make_env
from sb_pomdp.train import (
    SyncEnvironmentBatch,
    model_for_environment,
    train_single,
    truncation_bootstrap_values,
)
from sb_pomdp.train_baselines import train_baseline_single

ROOT = Path(__file__).resolve().parents[2]
TASK = "masked_pendulum"
CARTPOLE = {"horizon": 1, "observation_noise_std": 0.0}


def _smoke_config(tmp_path: Path, *, bootstrap: bool) -> ExperimentConfig:
    """Return the local one-update config with the flag in a chosen state.

    ``config/local.json`` gives ``masked_pendulum`` a horizon of eight and a
    rollout of eight steps, so every environment truncates exactly once inside
    the single update and the bootstrap path is always exercised.
    """

    return load_config(ROOT / "config" / "local.json").with_overrides(
        {
            "experiment.output_dir": str(tmp_path / "unused-output-root"),
            "ppo.bootstrap_on_truncation": bootstrap,
        }
    )


def _artifacts(path: Path, *, method: str | None = None) -> SeedArtifacts:
    # Keep the directory name short so atomic temporary files stay below the
    # legacy Windows MAX_PATH limit used by CI.
    path.mkdir(parents=True)
    return SeedArtifacts(path=path, task=TASK, seed=0, method=method)


def _explode(*arguments: Any, **keywords: Any) -> torch.Tensor:
    raise AssertionError("the default path must not compute a truncation bootstrap")


def test_step_returns_reset_observations_and_keeps_the_discarded_final_one() -> None:
    batch = SyncEnvironmentBatch(
        [make_env("masked_cartpole", CARTPOLE, seed=index) for index in range(2)]
    )
    batch.reset([0, 1])
    reference = make_env("masked_cartpole", CARTPOLE, seed=0)
    reference.reset(seed=0)

    observations, rewards, dones, infos = batch.step(np.zeros(2, dtype=np.int64))

    expected_final, expected_reward, terminated, truncated, _ = reference.step(0)
    expected_reset, _ = reference.reset()
    assert (terminated, truncated) == (False, True)
    # The returned observation array is unchanged: it is the auto-reset one.
    assert dones.tolist() == [True, True]
    assert observations.dtype == np.float32
    np.testing.assert_allclose(observations[0], expected_reset, rtol=0, atol=1e-6)
    np.testing.assert_allclose(rewards[0], expected_reward, rtol=0, atol=1e-6)
    # The genuine post-step observation is preserved beside it.
    for info in infos:
        assert info["truncated"] is True
        assert info["terminated"] is False
    np.testing.assert_allclose(infos[0]["final_observation"], expected_final, rtol=0, atol=1e-6)
    assert infos[0]["final_oracle_features"].shape == (reference.oracle_dim,)
    assert "reset_state" in infos[0]


def test_truncation_bootstrap_values_leave_the_online_belief_untouched() -> None:
    config = load_config(ROOT / "config" / "local.json")
    model_config = config["model"]
    environment = make_env(TASK, config.environment_for_task(TASK), seed=0)
    observation, _ = environment.reset(seed=0)
    model = model_for_environment(environment, model_config, torch.device("cpu"))
    particles = torch.randn(2, int(model_config["num_particles"]), environment.state_dim)
    action_features = torch.randn(2, environment.action_spec.feature_dim)
    final_observation = np.asarray(observation, dtype=np.float32)
    infos = [
        {"truncated": True, "terminated": False, "final_observation": final_observation},
        {"truncated": False, "terminated": True, "final_observation": final_observation},
    ]
    particles_before = particles.clone()
    actions_before = action_features.clone()

    values = truncation_bootstrap_values(
        model,
        model_config,
        infos=infos,
        particles=particles,
        action_features=action_features,
        device=torch.device("cpu"),
    )

    assert values.shape == (2,)
    assert not values.requires_grad
    # A true terminal state never bootstraps, even though its final observation
    # was recorded.
    assert float(values[1]) == 0.0
    assert torch.isfinite(values).all()
    assert torch.equal(particles, particles_before)
    assert torch.equal(action_features, actions_before)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_truncation_bootstrap_values_are_zero_without_a_truncation() -> None:
    config = load_config(ROOT / "config" / "local.json")
    model_config = config["model"]
    environment = make_env(TASK, config.environment_for_task(TASK), seed=0)
    model = model_for_environment(environment, model_config, torch.device("cpu"))
    state = torch.get_rng_state()

    values = truncation_bootstrap_values(
        model,
        model_config,
        infos=[{"truncated": False, "terminated": False}],
        particles=torch.zeros(1, int(model_config["num_particles"]), environment.state_dim),
        action_features=torch.zeros(1, environment.action_spec.feature_dim),
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(values, torch.zeros(1))
    # No truncation means no belief sample, hence no RNG consumption at all.
    assert torch.equal(torch.get_rng_state(), state)


def test_score_trainer_defaults_to_no_truncation_bootstrap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_module, "truncation_bootstrap_values", _explode)
    config = _smoke_config(tmp_path, bootstrap=False)
    assert config["ppo"]["bootstrap_on_truncation"] is False

    summary = train_single(
        config,
        task=TASK,
        seed=0,
        artifacts=_artifacts(tmp_path / "off"),
        max_updates=1,
        verbose=False,
    )

    assert summary["status"] == "completed"
    assert "truncation_bootstrap" not in (tmp_path / "off" / "run.log").read_text(
        encoding="utf-8"
    )


def test_score_trainer_bootstraps_one_update_on_truncation(tmp_path) -> None:
    artifacts = _artifacts(tmp_path / "on")

    summary = train_single(
        _smoke_config(tmp_path, bootstrap=True),
        task=TASK,
        seed=0,
        artifacts=artifacts,
        max_updates=1,
        verbose=False,
    )

    assert summary["status"] == "completed"
    assert np.isfinite(summary["training_return_mean"])
    log_text = (artifacts.path / "run.log").read_text(encoding="utf-8")
    # Horizon eight and an eight-step rollout truncate both environments once.
    assert "truncation_bootstrap update=1 truncations=2 value_mean=" in log_text
    assert "value_mean=n/a" not in log_text
    checkpoint = torch.load(
        artifacts.checkpoint_path(tag="latest", create_parent=False),
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["config"]["ppo"]["bootstrap_on_truncation"] is True
    assert all(torch.isfinite(tensor).all() for tensor in checkpoint["model"].values())


def test_the_flag_changes_the_score_trainer_update(tmp_path) -> None:
    parameters: dict[bool, dict[str, torch.Tensor]] = {}
    for bootstrap in (False, True):
        artifacts = _artifacts(tmp_path / f"m{int(bootstrap)}")
        train_single(
            _smoke_config(tmp_path, bootstrap=bootstrap),
            task=TASK,
            seed=0,
            artifacts=artifacts,
            max_updates=1,
            verbose=False,
        )
        parameters[bootstrap] = torch.load(
            artifacts.checkpoint_path(tag="latest", create_parent=False),
            map_location="cpu",
            weights_only=False,
        )["model"]

    assert any(
        not torch.equal(parameters[False][name], value)
        for name, value in parameters[True].items()
    )


def test_gru_baseline_defaults_to_no_truncation_bootstrap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_baselines_module, "_truncation_bootstrap_values", _explode)

    summary = train_baseline_single(
        _smoke_config(tmp_path, bootstrap=False),
        method="gru",
        task=TASK,
        seed=0,
        artifacts=_artifacts(tmp_path / "b0", method="gru"),
        max_updates=1,
        verbose=False,
        result_dir=f"gru/{TASK}/seed_000",
    )

    assert summary["status"] == "completed"


@pytest.mark.parametrize("method", ("gru", "observation_mlp", "oracle_state"))
def test_baselines_bootstrap_one_update_on_truncation(tmp_path, method: str) -> None:
    artifacts = _artifacts(tmp_path / "b1", method=method)

    summary = train_baseline_single(
        _smoke_config(tmp_path, bootstrap=True),
        method=method,  # type: ignore[arg-type]
        task=TASK,
        seed=0,
        artifacts=artifacts,
        max_updates=1,
        verbose=False,
        result_dir=f"{method}/{TASK}/seed_000",
    )

    assert summary["status"] == "completed"
    log_text = (artifacts.path / "run.log").read_text(encoding="utf-8")
    assert "truncation_bootstrap update=1 truncations=2 value_mean=" in log_text
    assert "value_mean=n/a" not in log_text
