from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.compare import _read_final_evaluations
from sb_pomdp.config import load_config
from sb_pomdp.train import (
    METRIC_FIELDS,
    _checkpoint_rollback_guidance,
    _truncate_uncommitted_csv_rows,
    train_single,
)


def _tiny_two_update_config(tmp_path: Path, *, horizon: int = 20):
    return load_config("config/local.json").with_overrides(
        {
            "experiment.output_dir": str(tmp_path / "unused-output-root"),
            "experiment.eval_interval_updates": 999,
            "experiment.checkpoint_interval_updates": 999,
            "environment.tasks.masked_cartpole.horizon": horizon,
            "model.num_particles": 2,
            "model.energy_hidden": [4],
            "model.d_model": 4,
            "model.num_heads": 1,
            "model.transformer_ff_dim": 8,
            "model.policy_hidden": [4],
            "ppo.total_steps": 8,
            "ppo.num_envs": 1,
            "ppo.rollout_steps": 4,
            "ppo.minibatch_size": 4,
            "ppo.sequence_microbatch_size": 4,
        }
    )


def _artifacts(path: Path) -> SeedArtifacts:
    path.mkdir()
    return SeedArtifacts(path=path, task="masked_cartpole", seed=3)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_update_boundary_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    config = _tiny_two_update_config(tmp_path)
    uninterrupted_artifacts = _artifacts(tmp_path / "uninterrupted")
    first_segment_artifacts = _artifacts(tmp_path / "first-segment")
    resumed_artifacts = _artifacts(tmp_path / "resumed")

    uninterrupted = train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=uninterrupted_artifacts,
        max_updates=2,
        evaluate_at_end=False,
        verbose=False,
    )
    first_segment = train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=first_segment_artifacts,
        segment_updates=1,
        verbose=False,
    )
    first_checkpoint_path = first_segment_artifacts.checkpoint_path(
        tag="latest", create_parent=False
    )
    first_checkpoint = torch.load(first_checkpoint_path, map_location="cpu", weights_only=False)

    assert first_segment["status"] == "partial"
    assert first_segment["updates"] == 1
    assert not (first_segment_artifacts.path / "evaluation.csv").exists()
    assert first_checkpoint["trainer_state"]["belief_histories"][0]["observations"]

    resumed = train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=resumed_artifacts,
        max_updates=2,
        resume_checkpoint=first_checkpoint_path,
        evaluate_at_end=False,
        verbose=False,
    )

    assert uninterrupted["status"] == resumed["status"] == "completed"
    assert uninterrupted["updates"] == resumed["updates"] == 2
    assert resumed["wall_time_seconds"] >= first_segment["wall_time_seconds"]

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
    for name, tensor in uninterrupted_checkpoint["model"].items():
        torch.testing.assert_close(tensor, resumed_checkpoint["model"][name], rtol=0, atol=0)

    timing_fields = {"steps_per_second"}
    with (
        np.load(uninterrupted_artifacts.arrays_path, allow_pickle=False) as expected,
        np.load(resumed_artifacts.arrays_path, allow_pickle=False) as actual,
    ):
        for field in set(METRIC_FIELDS) - timing_fields:
            np.testing.assert_allclose(actual[field], expected[field], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(actual["episode_returns"], expected["episode_returns"])
        np.testing.assert_array_equal(actual["episode_lengths"], expected["episode_lengths"])

    with resumed_artifacts.metrics_path.open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_truncation_bootstrap_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    """The opt-in bootstrap adds no unsaved state, so a segment boundary is exact.

    Horizon four equals the rollout length, so every update truncates and draws
    its extra ULA noise from the global stream.  That stream is part of the
    checkpoint, which is what keeps the resumed segment bit-identical.
    """

    config = _tiny_two_update_config(tmp_path, horizon=4).with_overrides(
        {"ppo.bootstrap_on_truncation": True}
    )
    uninterrupted_artifacts = _artifacts(tmp_path / "uninterrupted")
    resumed_artifacts = _artifacts(tmp_path / "resumed")

    train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=uninterrupted_artifacts,
        max_updates=2,
        evaluate_at_end=False,
        verbose=False,
    )
    train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=resumed_artifacts,
        segment_updates=1,
        verbose=False,
    )
    resumed = train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=resumed_artifacts,
        max_updates=2,
        resume_checkpoint=resumed_artifacts.checkpoint_path(tag="latest", create_parent=False),
        evaluate_at_end=False,
        verbose=False,
    )

    assert resumed["status"] == "completed"
    assert "truncation_bootstrap update=2 truncations=1" in (
        resumed_artifacts.path / "run.log"
    ).read_text(encoding="utf-8")
    uninterrupted_model = torch.load(
        uninterrupted_artifacts.checkpoint_path(tag="latest", create_parent=False),
        map_location="cpu",
        weights_only=False,
    )["model"]
    resumed_model = torch.load(
        resumed_artifacts.checkpoint_path(tag="latest", create_parent=False),
        map_location="cpu",
        weights_only=False,
    )["model"]
    for name, tensor in uninterrupted_model.items():
        torch.testing.assert_close(tensor, resumed_model[name], rtol=0, atol=0)


def test_resume_discards_csv_rows_written_after_the_committed_checkpoint(
    tmp_path: Path,
) -> None:
    config = _tiny_two_update_config(tmp_path, horizon=4)
    artifacts = _artifacts(tmp_path / "chained-shard")
    episodes_path = artifacts.path / "episodes.csv"
    evaluation_path = artifacts.path / "evaluation.csv"

    train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=artifacts,
        segment_updates=1,
        verbose=False,
    )
    committed_checkpoint = artifacts.checkpoint_path(tag="latest", create_parent=False)
    committed_bytes = committed_checkpoint.read_bytes()

    # Play the final update forward exactly as an uninterrupted chain would.
    uninterrupted = train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=artifacts,
        segment_updates=1,
        resume_checkpoint=committed_checkpoint,
        verbose=False,
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

    resumed = train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=artifacts,
        segment_updates=1,
        resume_checkpoint=committed_checkpoint,
        verbose=False,
    )

    assert resumed["status"] == uninterrupted["status"] == "completed"
    assert resumed["updates"] == uninterrupted["updates"] == 2
    assert _csv_rows(episodes_path) == expected_episodes
    assert _csv_rows(evaluation_path) == expected_evaluation
    final_evaluations = _read_final_evaluations(evaluation_path)
    assert final_evaluations["deterministic_mode"] == "greedy"
    assert final_evaluations["deterministic_return_mean"] == float(
        expected_evaluation[0]["mean_return"]
    )
    assert final_evaluations["stochastic_return_mean"] == float(
        expected_evaluation[1]["mean_return"]
    )

    timing_fields = {"steps_per_second"}
    resumed_metrics = _csv_rows(artifacts.metrics_path)
    assert [row["update"] for row in resumed_metrics] == ["1", "2"]
    for actual, expected in zip(resumed_metrics, expected_metrics):
        assert {field: actual[field] for field in actual if field not in timing_fields} == {
            field: expected[field] for field in expected if field not in timing_fields
        }
    assert not list(artifacts.path.glob("._*.tmp"))


def test_uncommitted_row_truncation_fails_fast_on_an_invalid_progress_column(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path / "corrupt")
    artifacts.metrics_path.write_text(
        "update,global_step\r\nlatest,4\r\n",
        encoding="utf-8",
        newline="",
    )

    with pytest.raises(ValueError, match="instead of an integer"):
        _truncate_uncommitted_csv_rows(artifacts, update=1, global_step=4)


def test_rejected_nonfinite_checkpoint_state_names_the_rollback_targets(
    tmp_path: Path,
) -> None:
    config = _tiny_two_update_config(tmp_path)
    artifacts = _artifacts(tmp_path / "corrupted-shard")

    train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=artifacts,
        segment_updates=1,
        verbose=False,
    )
    committed = artifacts.checkpoint_path(tag="latest", create_parent=False)
    payload = torch.load(committed, map_location="cpu", weights_only=False)
    payload["trainer_state"]["belief_context"]["observation"][0, 0] = float("nan")
    torch.save(payload, committed)

    with pytest.raises(ValueError) as failure:
        train_single(
            config,
            task="masked_cartpole",
            seed=3,
            artifacts=artifacts,
            max_updates=2,
            resume_checkpoint=committed,
            verbose=False,
        )

    message = str(failure.value)
    assert "belief_context.observation contains non-finite values" in message
    assert "resume instead from an older checkpoint in" in message
    assert "step_000000004.pt" in message
    assert "latest.pt" not in message


def test_rollback_guidance_lists_newest_steps_first_and_reports_an_empty_directory(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    rejected = checkpoints / "latest.pt"
    rejected.write_bytes(b"rejected")

    assert "holds no other checkpoint" in _checkpoint_rollback_guidance(rejected)

    for step in range(1, 8):
        (checkpoints / f"step_{step * 4:09d}.pt").write_bytes(b"checkpoint")
    (checkpoints / "best.pt").write_bytes(b"checkpoint")
    guidance = _checkpoint_rollback_guidance(rejected)

    assert "step_000000028.pt, step_000000024.pt, step_000000020.pt" in guidance
    assert "(newest 5 of 8)" in guidance
    assert "step_000000004.pt" not in guidance
    assert "latest.pt" not in guidance


def test_resume_rejects_legacy_checkpoint_without_trainer_state(tmp_path: Path) -> None:
    config = _tiny_two_update_config(tmp_path)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"model": {}, "optimizer": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="lacks trainer_state"):
        train_single(
            config,
            task="masked_cartpole",
            seed=3,
            artifacts=_artifacts(tmp_path / "legacy-attempt"),
            max_updates=2,
            resume_checkpoint=checkpoint_path,
            verbose=False,
        )


@pytest.mark.parametrize("section", ("model", "optimizer"))
def test_resume_rejects_nonfinite_model_or_optimizer_state(
    tmp_path: Path,
    section: str,
) -> None:
    config = _tiny_two_update_config(tmp_path)
    artifacts = _artifacts(tmp_path / f"nonfinite-{section}")
    train_single(
        config,
        task="masked_cartpole",
        seed=3,
        artifacts=artifacts,
        segment_updates=1,
        verbose=False,
        method="score_transformer",
    )
    checkpoint_path = artifacts.checkpoint_path(tag="latest", create_parent=False)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if section == "model":
        tensor = next(
            value for value in payload["model"].values() if value.is_floating_point()
        )
    else:
        tensor = next(
            value
            for state in payload["optimizer"]["state"].values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        )
    tensor.reshape(-1)[0] = float("nan")
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match=rf"checkpoint {section}.*non-finite") as failure:
        train_single(
            config,
            task="masked_cartpole",
            seed=3,
            artifacts=artifacts,
            segment_updates=1,
            resume_checkpoint=checkpoint_path,
            verbose=False,
            method="score_transformer",
        )
    assert "resume instead from an older checkpoint" in str(failure.value)
