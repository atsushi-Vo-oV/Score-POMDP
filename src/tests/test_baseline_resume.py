"""Segmented resume for the baseline trainer, mirroring ``test_training_resume``."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.compare import _read_final_evaluations
from sb_pomdp.config import ExperimentConfig, load_config
from sb_pomdp.train import METRIC_FIELDS, train_single
from sb_pomdp.train_baselines import (
    BASELINE_CHECKPOINT_FORMAT_VERSION,
    train_baseline_single,
)

TASK = "masked_cartpole"
METHODS = ("observation_mlp", "gru", "oracle_state", "particle_filter")


def _tiny_two_update_config(tmp_path: Path, *, horizon: int = 3) -> ExperimentConfig:
    """Return a two-update baseline config small enough to run twice per test.

    Horizon three against a four-step rollout is deliberate: every update both
    finishes one episode and leaves a one-step episode prefix open, so a segment
    boundary has to carry a partially replayed recurrent history across it.
    """

    return load_config("config/local.json").with_overrides(
        {
            "experiment.output_dir": str(tmp_path / "unused-output-root"),
            "experiment.eval_episodes": 1,
            "experiment.eval_interval_updates": 999,
            "experiment.checkpoint_interval_updates": 999,
            f"environment.tasks.{TASK}.horizon": horizon,
            "model.d_model": 4,
            "model.policy_hidden": [4],
            "model.diffusion_steps": 1,
            "comparison.gru_encoder_hidden": [4],
            "comparison.gru_hidden_dim": 4,
            "comparison.pf_hidden": [4],
            "comparison.pf_particle_dim": 3,
            "ppo.total_steps": 8,
            "ppo.num_envs": 1,
            "ppo.rollout_steps": 4,
            "ppo.epochs": 1,
            "ppo.minibatch_size": 4,
            "ppo.sequence_microbatch_size": 4,
        }
    )


def _artifacts(path: Path, method: str) -> SeedArtifacts:
    # Short directory names keep the atomic temporary files below the legacy
    # Windows MAX_PATH limit used by CI.
    path.mkdir(parents=True)
    return SeedArtifacts(path=path, task=TASK, seed=3, method=method)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _train(
    config: ExperimentConfig,
    artifacts: SeedArtifacts,
    method: str,
    **keywords: object,
) -> dict[str, object]:
    return train_baseline_single(
        config,
        method=method,  # type: ignore[arg-type]
        task=TASK,
        seed=3,
        artifacts=artifacts,
        verbose=False,
        result_dir=f"{method}/{TASK}/seed_003",
        **keywords,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("method", METHODS)
def test_segmented_baseline_resume_matches_uninterrupted_training(
    tmp_path: Path,
    method: str,
) -> None:
    config = _tiny_two_update_config(tmp_path)
    uninterrupted_artifacts = _artifacts(tmp_path / "u", method)
    resumed_artifacts = _artifacts(tmp_path / "r", method)

    uninterrupted = _train(
        config,
        uninterrupted_artifacts,
        method,
        max_updates=2,
        evaluate_at_end=False,
    )
    first_segment = _train(config, resumed_artifacts, method, segment_updates=1)
    first_checkpoint_path = resumed_artifacts.checkpoint_path(tag="latest", create_parent=False)
    first_checkpoint = torch.load(first_checkpoint_path, map_location="cpu", weights_only=False)

    assert first_segment["status"] == "partial"
    assert first_segment["updates"] == 1
    assert not (resumed_artifacts.path / "evaluation.csv").exists()
    assert first_checkpoint["format_version"] == BASELINE_CHECKPOINT_FORMAT_VERSION
    assert first_checkpoint["trainer_state"]["format_version"] == 1
    if method in ("gru", "particle_filter"):
        assert first_checkpoint["trainer_state"]["recurrent_histories"][0]["inputs"]
    else:
        assert first_checkpoint["trainer_state"]["recurrent_histories"] == []

    resumed = _train(
        config,
        resumed_artifacts,
        method,
        max_updates=2,
        resume_checkpoint=first_checkpoint_path,
        evaluate_at_end=False,
    )

    assert uninterrupted["status"] == resumed["status"] == "completed"
    assert uninterrupted["updates"] == resumed["updates"] == 2
    assert resumed["wall_time_seconds"] >= first_segment["wall_time_seconds"]
    assert uninterrupted["episodes_completed"] == resumed["episodes_completed"]

    uninterrupted_checkpoint = torch.load(
        uninterrupted_artifacts.checkpoint_path(tag="latest", create_parent=False),
        map_location="cpu",
        weights_only=False,
    )
    resumed_checkpoint = torch.load(
        resumed_artifacts.checkpoint_path(tag="latest", create_parent=False),
        map_location="cpu",
        weights_only=False,
    )
    assert uninterrupted_checkpoint["update"] == resumed_checkpoint["update"] == 2
    assert uninterrupted_checkpoint["model"].keys() == resumed_checkpoint["model"].keys()
    for name, tensor in uninterrupted_checkpoint["model"].items():
        torch.testing.assert_close(tensor, resumed_checkpoint["model"][name], rtol=0, atol=0)

    timing_fields = {"steps_per_second"}
    expected_metrics = _csv_rows(uninterrupted_artifacts.metrics_path)
    actual_metrics = _csv_rows(resumed_artifacts.metrics_path)
    assert [row["update"] for row in actual_metrics] == ["1", "2"]
    for actual, expected in zip(actual_metrics, expected_metrics):
        assert {field: actual[field] for field in actual if field not in timing_fields} == {
            field: expected[field] for field in expected if field not in timing_fields
        }
    assert _csv_rows(resumed_artifacts.path / "episodes.csv") == _csv_rows(
        uninterrupted_artifacts.path / "episodes.csv"
    )

    with (
        np.load(uninterrupted_artifacts.arrays_path, allow_pickle=False) as expected_arrays,
        np.load(resumed_artifacts.arrays_path, allow_pickle=False) as actual_arrays,
    ):
        for field in set(METRIC_FIELDS) - timing_fields:
            np.testing.assert_allclose(
                actual_arrays[field],
                expected_arrays[field],
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        np.testing.assert_array_equal(
            actual_arrays["episode_returns"], expected_arrays["episode_returns"]
        )
        np.testing.assert_array_equal(
            actual_arrays["episode_lengths"], expected_arrays["episode_lengths"]
        )


def test_segmented_baseline_final_evaluation_matches_an_uninterrupted_run(
    tmp_path: Path,
) -> None:
    """The final update evaluates in both shapes and produces identical readouts."""

    config = _tiny_two_update_config(tmp_path)
    uninterrupted_artifacts = _artifacts(tmp_path / "u", "gru")
    resumed_artifacts = _artifacts(tmp_path / "r", "gru")

    _train(config, uninterrupted_artifacts, "gru")
    _train(config, resumed_artifacts, "gru", segment_updates=1)
    resumed = _train(
        config,
        resumed_artifacts,
        "gru",
        segment_updates=1,
        resume_checkpoint=resumed_artifacts.checkpoint_path(tag="latest", create_parent=False),
    )

    assert resumed["status"] == "completed"
    expected = _read_final_evaluations(uninterrupted_artifacts.path / "evaluation.csv")
    actual = _read_final_evaluations(resumed_artifacts.path / "evaluation.csv")
    assert actual == expected
    assert resumed["evaluation_return_mean"] == expected["deterministic_return_mean"]


def test_baseline_resume_discards_csv_rows_written_after_the_committed_checkpoint(
    tmp_path: Path,
) -> None:
    config = _tiny_two_update_config(tmp_path)
    artifacts = _artifacts(tmp_path / "c", "gru")
    episodes_path = artifacts.path / "episodes.csv"
    evaluation_path = artifacts.path / "evaluation.csv"

    _train(config, artifacts, "gru", segment_updates=1)
    committed_checkpoint = artifacts.checkpoint_path(tag="latest", create_parent=False)
    committed_bytes = committed_checkpoint.read_bytes()

    uninterrupted = _train(
        config,
        artifacts,
        "gru",
        segment_updates=1,
        resume_checkpoint=committed_checkpoint,
    )
    expected_metrics = _csv_rows(artifacts.metrics_path)
    expected_episodes = _csv_rows(episodes_path)
    expected_evaluation = _csv_rows(evaluation_path)

    # Rewind only the checkpoint: this is the state left by a kill between the
    # CSV fsyncs of the final update and the checkpoint save that commits it.
    committed_checkpoint.write_bytes(committed_bytes)
    assert [row["update"] for row in expected_metrics] == ["1", "2"]
    assert [row["update"] for row in expected_evaluation] == ["2", "2"]
    assert any(int(row["global_step"]) > 4 for row in expected_episodes)

    resumed = _train(
        config,
        artifacts,
        "gru",
        segment_updates=1,
        resume_checkpoint=committed_checkpoint,
    )

    assert resumed["status"] == uninterrupted["status"] == "completed"
    assert resumed["updates"] == uninterrupted["updates"] == 2
    assert _csv_rows(episodes_path) == expected_episodes
    assert _csv_rows(evaluation_path) == expected_evaluation
    assert _read_final_evaluations(evaluation_path)["deterministic_mode"] == "greedy"
    assert [row["update"] for row in _csv_rows(artifacts.metrics_path)] == ["1", "2"]
    assert not list(artifacts.path.glob("._*.tmp"))


def test_baseline_status_is_partial_until_the_configured_updates_are_reached(
    tmp_path: Path,
) -> None:
    config = _tiny_two_update_config(tmp_path)
    artifacts = _artifacts(tmp_path / "s", "observation_mlp")

    first = _train(config, artifacts, "observation_mlp", segment_updates=1)
    assert first["status"] == "partial"
    assert first["updates"] == 1

    capped = _train(
        config,
        _artifacts(tmp_path / "m", "observation_mlp"),
        "observation_mlp",
        max_updates=1,
    )
    assert capped["status"] == "partial"

    second = _train(
        config,
        artifacts,
        "observation_mlp",
        segment_updates=1,
        resume_checkpoint=artifacts.checkpoint_path(tag="latest", create_parent=False),
    )
    assert second["status"] == "completed"
    assert second["updates"] == 2


def test_baseline_resume_rejects_legacy_checkpoint_without_trainer_state(
    tmp_path: Path,
) -> None:
    config = _tiny_two_update_config(tmp_path)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"model": {}, "optimizer": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="lacks trainer_state"):
        _train(
            config,
            _artifacts(tmp_path / "l", "gru"),
            "gru",
            max_updates=2,
            resume_checkpoint=checkpoint_path,
        )


def test_the_two_trainers_refuse_each_other_checkpoints(tmp_path: Path) -> None:
    config = _tiny_two_update_config(tmp_path)
    baseline_artifacts = _artifacts(tmp_path / "b", "observation_mlp")
    score_artifacts = SeedArtifacts(path=tmp_path / "s", task=TASK, seed=3)
    score_artifacts.path.mkdir(parents=True)

    _train(config, baseline_artifacts, "observation_mlp", segment_updates=1)
    train_single(
        config,
        task=TASK,
        seed=3,
        artifacts=score_artifacts,
        segment_updates=1,
        verbose=False,
        method="score_transformer",
    )
    baseline_checkpoint = baseline_artifacts.checkpoint_path(tag="latest", create_parent=False)
    score_checkpoint = score_artifacts.checkpoint_path(tag="latest", create_parent=False)

    with pytest.raises(ValueError, match="different checkpoint format"):
        _train(
            config,
            _artifacts(tmp_path / "bs", "observation_mlp"),
            "observation_mlp",
            max_updates=2,
            resume_checkpoint=score_checkpoint,
        )
    score_attempt = tmp_path / "sb"
    score_attempt.mkdir()
    with pytest.raises(ValueError, match="different checkpoint format"):
        train_single(
            config,
            task=TASK,
            seed=3,
            artifacts=SeedArtifacts(path=score_attempt, task=TASK, seed=3),
            max_updates=2,
            resume_checkpoint=baseline_checkpoint,
            verbose=False,
            method="score_transformer",
        )
