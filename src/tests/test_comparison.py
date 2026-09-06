from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from sb_pomdp.aggregate import _expected_coverage, _validate_coverage
from sb_pomdp.compare import (
    COMPARISON_FIELDS,
    _aggregate,
    _eligible,
    _method_config,
    _selection,
    run_comparisons,
)
from sb_pomdp.compare import (
    main as comparison_main,
)
from sb_pomdp.config import ConfigError, ExperimentConfig, load_config
from sb_pomdp.fetch_results import _verify_staged_result


def _segmented_shard_config(output_dir: Path) -> ExperimentConfig:
    """Return a two-update score_transformer config for segmented-resume tests."""

    return load_config("config/local.json").with_overrides(
        {
            "experiment.output_dir": str(output_dir),
            "experiment.eval_episodes": 1,
            "experiment.eval_interval_updates": 999,
            "experiment.checkpoint_interval_updates": 999,
            "environment.tasks.masked_cartpole.horizon": 20,
            "model.num_particles": 2,
            "model.langevin_steps": 1,
            "model.energy_hidden": [4],
            "model.d_model": 4,
            "model.num_heads": 1,
            "model.num_transformer_layers": 1,
            "model.transformer_ff_dim": 8,
            "model.policy_hidden": [4],
            "model.diffusion_steps": 1,
            "ppo.total_steps": 8,
            "ppo.num_envs": 1,
            "ppo.rollout_steps": 4,
            "ppo.epochs": 1,
            "ppo.minibatch_size": 4,
            "ppo.sequence_microbatch_size": 4,
        }
    )


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    """Return the full byte content of every file below *root*."""

    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_comparison_matrix_contains_forty_six_eligible_gradient_method_tasks() -> None:
    config = load_config("config/local.json")
    methods, tasks, seeds = _selection(config, None, None, None)
    gradient_modes = list(config["comparison"]["belief_gradient_modes"])
    eligible = [
        (gradient_mode, method, task)
        for gradient_mode in gradient_modes
        for method in methods
        for task in tasks
        if _eligible(config, method, task)[0]
    ]
    skipped = [
        (gradient_mode, method, task)
        for gradient_mode in gradient_modes
        for method in methods
        for task in tasks
        if not _eligible(config, method, task)[0]
    ]

    assert len(eligible) == 46
    assert skipped == [
        ("full", "score_gaussian", "masked_cartpole"),
        ("tbptt_1", "score_gaussian", "masked_cartpole"),
    ]
    assert seeds == [0]


def test_production_coverage_requires_all_eighty_shards() -> None:
    config = load_config("config/production.json")
    expected = _expected_coverage(config)

    assert len(expected) == 80
    assert _validate_coverage(expected, config) == 80
    with pytest.raises(RuntimeError, match="missing="):
        _validate_coverage(set(list(expected)[1:]), config)


def test_production_bulk_index_accepts_80_and_rejects_81(capsys) -> None:
    assert (
        comparison_main(
            [
                "--config",
                "config/production.json",
                "--bulk-index",
                "80",
                "--validate-only",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["bulk_index"] == 80
    assert payload["belief_gradient_modes"] == ["tbptt_1"]
    assert len(payload["eligible_method_tasks"]) == 1
    assert payload["methods"] == ["gru"]
    assert payload["tasks"] == ["light_dark"]
    assert payload["seeds"] == [14]

    assert (
        comparison_main(
            [
                "--config",
                "config/production.json",
                "--bulk-index",
                "81",
                "--validate-only",
            ]
        )
        == 1
    )
    assert "[1, 80]" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("index", "gradient_mode", "method"),
    [
        (11, "full", "score_transformer"),
        (51, "tbptt_1", "score_transformer"),
        (31, "full", "gru"),
    ],
)
def test_production_three_shard_pilot_indices_are_stable(
    index: int,
    gradient_mode: str,
    method: str,
    capsys,
) -> None:
    assert (
        comparison_main(
            [
                "--config",
                "config/production.json",
                "--bulk-index",
                str(index),
                "--campaign-id",
                "pilot-index-audit",
                "--validate-only",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["belief_gradient_modes"] == [gradient_mode]
    assert payload["methods"] == [method]
    assert payload["tasks"] == ["masked_mountain_car_continuous"]
    assert payload["seeds"] == [10]


def test_filtered_campaign_shard_index_is_written_to_validation_payload(capsys) -> None:
    assert (
        comparison_main(
            [
                "--config",
                "config/local.json",
                "--gradient-modes",
                "tbptt_1",
                "--methods",
                "score_transformer",
                "--tasks",
                "light_dark",
                "--seeds",
                "0",
                "--campaign-shard-index",
                "8",
                "--validate-only",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["bulk_index"] == 8
    assert payload["belief_gradient_modes"] == ["tbptt_1"]


def test_segmented_cli_requires_and_reports_one_resumable_run_dir(capsys) -> None:
    arguments = [
        "--config",
        "config/production.json",
        "--bulk-index",
        "11",
        "--campaign-id",
        "segment-validation",
        "--segment-updates",
        "10",
        "--run-dir",
        "results/campaigns/segment-validation/shard011",
        "--resume",
        "--validate-only",
    ]
    assert comparison_main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["segment_updates"] == 10
    assert payload["resume"] is True
    assert payload["bulk_index"] == 11

    missing_resume = [argument for argument in arguments if argument != "--resume"]
    assert comparison_main(missing_resume) == 1
    assert "requires --run-dir and --resume" in capsys.readouterr().err


def test_segmented_comparison_reopens_one_fixed_root_until_completed(tmp_path) -> None:
    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / "shard001"
    arguments = {
        "methods": ["score_transformer"],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "segmented-integration",
        "campaign_id": "pytest-segmented",
        "bulk_index": 1,
        "verbose": False,
    }

    first_run, first_summaries, first_skipped = run_comparisons(config, **arguments)
    second_run, second_summaries, second_skipped = run_comparisons(config, **arguments)

    assert first_run.root == second_run.root == run_dir.resolve()
    assert first_skipped == second_skipped == []
    assert first_summaries[0]["status"] == "partial"
    assert first_summaries[0]["updates"] == 1
    assert second_summaries[0]["status"] == "completed"
    assert second_summaries[0]["updates"] == 2
    metadata = json.loads(second_run.metadata_path.read_text(encoding="utf-8"))
    assert [segment["status"] for segment in metadata["segments"]] == [
        "partial",
        "completed",
    ]
    result_dir = second_run.root / second_summaries[0]["result_dir"]
    with (result_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_segmented_comparison_resumes_a_baseline_shard_like_a_score_shard(tmp_path) -> None:
    """Baseline shards go through the identical segmented flow, not a refusal."""

    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / "shard005"
    arguments = {
        "methods": ["gru"],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "segmented-baseline",
        "campaign_id": "pytest-baseline",
        "bulk_index": 5,
        "verbose": False,
    }

    first_run, first_summaries, first_skipped = run_comparisons(config, **arguments)
    second_run, second_summaries, _ = run_comparisons(config, **arguments)

    assert first_run.root == second_run.root == run_dir.resolve()
    assert first_skipped == []
    assert first_summaries[0]["status"] == "partial"
    assert first_summaries[0]["updates"] == 1
    assert second_summaries[0]["status"] == "completed"
    assert second_summaries[0]["updates"] == 2
    assert second_summaries[0]["stochastic_return_mean"] is not None
    metadata = json.loads(second_run.metadata_path.read_text(encoding="utf-8"))
    assert [segment["status"] for segment in metadata["segments"]] == ["partial", "completed"]
    result_dir = second_run.root / second_summaries[0]["result_dir"]
    with (result_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2

    # The completed-shard guard is symmetric: a redundant baseline segment must
    # leave the artifacts byte-identical and still succeed.
    before = _artifact_bytes(run_dir)
    _, third_summaries, _ = run_comparisons(config, **arguments)
    assert third_summaries == second_summaries
    assert _artifact_bytes(run_dir) == before


def test_legacy_segment_artifacts_resume_at_the_legacy_config_defaults(tmp_path) -> None:
    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / "shard004"
    arguments = {
        "methods": ["score_transformer"],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "segmented-legacy",
        "campaign_id": "pytest-legacy",
        "bulk_index": 1,
        "verbose": False,
    }

    _, first_summaries, _ = run_comparisons(config, **arguments)
    assert first_summaries[0]["status"] == "partial"

    # Rewrite the segment's artifacts in the pre-F-1 schema, exactly as the
    # submitted campaign wrote them, and resume against the current schema.
    for path in sorted(run_dir.rglob("resolved_config.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["ppo"]["bootstrap_on_truncation"]
        path.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint_path = run_dir / first_summaries[0]["result_dir"] / "checkpoints" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    del checkpoint["config"]["ppo"]["bootstrap_on_truncation"]
    torch.save(checkpoint, checkpoint_path)

    _, second_summaries, _ = run_comparisons(config, **arguments)

    assert second_summaries[0]["status"] == "completed"
    assert second_summaries[0]["updates"] == 2


def test_extra_segment_on_a_completed_shard_changes_nothing_and_exits_zero(
    tmp_path,
    capsys,
) -> None:
    config = _segmented_shard_config(tmp_path)
    config_path = tmp_path / "segmented_config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    run_dir = tmp_path / "campaign" / "shard001"
    arguments = [
        "--config",
        str(config_path),
        "--gradient-modes",
        "full",
        "--methods",
        "score_transformer",
        "--tasks",
        "masked_cartpole",
        "--seeds",
        "0",
        "--segment-updates",
        "1",
        "--run-dir",
        str(run_dir),
        "--resume",
        "--label",
        "segmented-guard",
        "--campaign-id",
        "pytest-guard",
        "--campaign-shard-index",
        "1",
        "--quiet",
    ]

    assert comparison_main(arguments) == 0
    assert comparison_main(arguments) == 0
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["runs"][0]["updates"] == 2
    before = _artifact_bytes(run_dir)
    capsys.readouterr()

    assert comparison_main(arguments) == 0

    assert "Already completed" in capsys.readouterr().err
    assert _artifact_bytes(run_dir) == before


@pytest.mark.parametrize(
    ("method", "trainer_name", "bulk_index"),
    (
        ("score_transformer", "train_single", 1),
        ("gru", "train_baseline_single", 5),
    ),
)
def test_final_checkpoint_recovers_when_the_process_dies_before_root_commit(
    tmp_path,
    monkeypatch,
    method: str,
    trainer_name: str,
    bulk_index: int,
) -> None:
    """A durable final trainer checkpoint is sufficient to finish the root."""

    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / f"shard{bulk_index:03d}"
    arguments = {
        "methods": [method],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "final-checkpoint-recovery",
        "campaign_id": f"pytest-final-recovery-{method}",
        "bulk_index": bulk_index,
        "verbose": False,
    }

    _, first_summaries, _ = run_comparisons(config, **arguments)
    seed_root = run_dir / first_summaries[0]["result_dir"]
    stale_arrays = (seed_root / "arrays.npz").read_bytes()

    import sb_pomdp.compare as comparison_module

    original_trainer = getattr(comparison_module, trainer_name)

    class SimulatedProcessDeath(BaseException):
        pass

    def die_after_final_checkpoint(*args, **kwargs):
        summary = original_trainer(*args, **kwargs)
        assert summary["status"] == "completed"
        raise SimulatedProcessDeath

    with monkeypatch.context() as patch:
        patch.setattr(comparison_module, trainer_name, die_after_final_checkpoint)
        with pytest.raises(SimulatedProcessDeath):
            run_comparisons(config, **arguments)

    checkpoint = torch.load(
        seed_root / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["update"] == 2
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == (
        "running"
    )

    # At the actual checkpoint-save boundary arrays.npz still describes the
    # preceding segment.  Recovery must rebuild it from trainer_state too.
    (seed_root / "arrays.npz").write_bytes(stale_arrays)
    _, recovered, _ = run_comparisons(config, **arguments)

    assert recovered[0]["status"] == "completed"
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == (
        "completed"
    )
    with np.load(seed_root / "arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["update"], np.asarray([1, 2]))


def test_inconsistent_completed_marker_is_repaired_instead_of_frozen(tmp_path) -> None:
    """Legacy commit order may leave completed manifest with stale root files."""

    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / "shard007"
    arguments = {
        "methods": ["score_transformer"],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "completed-marker-recovery",
        "campaign_id": "pytest-completed-marker-recovery",
        "bulk_index": 7,
        "verbose": False,
    }
    run_comparisons(config, **arguments)
    run_comparisons(config, **arguments)

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["status"] = "running"
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary[0]["status"] = "partial"
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    _, repaired, _ = run_comparisons(config, **arguments)

    assert repaired[0]["status"] == "completed"
    assert json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))["status"] == (
        "completed"
    )
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))[0][
        "status"
    ] == "completed"


def test_source_fingerprint_failure_leaves_retryable_failure_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / "shard008"
    arguments = {
        "methods": ["score_transformer"],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "fingerprint-failure",
        "campaign_id": "pytest-fingerprint-failure",
        "bulk_index": 8,
        "verbose": False,
    }

    def fail_fingerprint() -> str:
        raise OSError("test source read failure")

    with monkeypatch.context() as patch:
        patch.setattr("sb_pomdp.compare.source_sha256", fail_fingerprint)
        with pytest.raises(OSError, match="test source read failure"):
            run_comparisons(config, **arguments)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert manifest["status"] == metadata["status"] == "failed"
    assert manifest["runs"] == []
    assert (run_dir / "failure.txt").is_file()

    _, summaries, _ = run_comparisons(config, **arguments)
    assert summaries[0]["status"] == "partial"


def test_segmented_resume_rejects_a_changed_runtime_identity(tmp_path) -> None:
    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / "shard009"
    arguments = {
        "methods": ["score_transformer"],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "runtime-identity",
        "campaign_id": "pytest-runtime-identity",
        "bulk_index": 9,
        "verbose": False,
    }
    run_comparisons(config, **arguments)

    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_torch = metadata["torch"]
    metadata["torch"] = f"{original_torch}-different"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ConfigError, match="runtime identity differs"):
        run_comparisons(config, **arguments)

    rejected = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert rejected["status"] == "failed"
    assert rejected["torch"] == f"{original_torch}-different"

    # An explicit opt-in (MIG -> full GPU after an out-of-memory segment)
    # continues the run and records the change instead of failing it.
    monkeypatch_value = os.environ.get("SB_POMDP_ALLOW_RUNTIME_CHANGE")
    os.environ["SB_POMDP_ALLOW_RUNTIME_CHANGE"] = "1"
    try:
        run_comparisons(config, **arguments)
    finally:
        if monkeypatch_value is None:
            os.environ.pop("SB_POMDP_ALLOW_RUNTIME_CHANGE", None)
        else:
            os.environ["SB_POMDP_ALLOW_RUNTIME_CHANGE"] = monkeypatch_value
    accepted = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert accepted["status"] != "failed"
    assert accepted["runtime_changes"] and "torch" in accepted["runtime_changes"][0]["differences"]


def test_first_segment_setup_failure_records_source_and_restarts_without_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / "shard006"
    arguments = {
        "methods": ["score_transformer"],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "first-segment-setup-failure",
        "campaign_id": "pytest-first-setup-failure",
        "bulk_index": 6,
        "verbose": False,
    }

    def fail_device(_: str) -> None:
        raise RuntimeError("test first-segment device failure")

    monkeypatch.setattr("sb_pomdp.compare.resolve_device", fail_device)
    with pytest.raises(RuntimeError, match="test first-segment device failure"):
        run_comparisons(config, **arguments)

    failed_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    failed_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = failed_metadata["source_sha256"]
    seed_root = run_dir / "full" / "score_transformer" / "masked_cartpole" / "seed_000"
    assert failed_metadata["status"] == failed_manifest["status"] == "failed"
    assert isinstance(fingerprint, str) and len(fingerprint) == 64
    assert failed_manifest["runs"] == []
    assert not seed_root.exists()
    monkeypatch.undo()

    # The narrow checkpoint-free restart must not weaken the source guard.
    mismatched_fingerprint = ("0" if fingerprint[0] != "0" else "1") + fingerprint[1:]
    failed_metadata["source_sha256"] = mismatched_fingerprint
    (run_dir / "metadata.json").write_text(json.dumps(failed_metadata), encoding="utf-8")
    with pytest.raises(ConfigError, match="source code differs"):
        run_comparisons(config, **arguments)
    rejected_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    rejected_metadata["source_sha256"] = fingerprint
    (run_dir / "metadata.json").write_text(json.dumps(rejected_metadata), encoding="utf-8")

    second_run, second_summaries, second_skipped = run_comparisons(config, **arguments)

    assert second_run.root == run_dir.resolve()
    assert second_skipped == []
    assert second_summaries[0]["status"] == "partial"
    assert second_summaries[0]["updates"] == 1
    checkpoint = seed_root / "checkpoints" / "latest.pt"
    assert checkpoint.is_file()
    second_metadata = json.loads(second_run.metadata_path.read_text(encoding="utf-8"))
    assert second_metadata["source_sha256"] == fingerprint
    assert [segment["status"] for segment in second_metadata["segments"]] == ["partial"]

    _, third_summaries, _ = run_comparisons(config, **arguments)
    assert third_summaries[0]["status"] == "completed"
    assert third_summaries[0]["updates"] == 2


def test_first_update_hard_kill_quarantines_uncommitted_seed_before_restart(
    tmp_path,
    monkeypatch,
) -> None:
    """A seed tree without any committed checkpoint is safe to restart at zero."""

    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / "shard007"
    arguments = {
        "methods": ["score_transformer"],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "first-update-hard-kill",
        "campaign_id": "pytest-first-update-hard-kill",
        "bulk_index": 7,
        "verbose": False,
    }
    seed_root = run_dir / "full" / "score_transformer" / "masked_cartpole" / "seed_000"

    class SimulatedProcessDeath(BaseException):
        pass

    def die_before_first_checkpoint(*args, artifacts, **kwargs) -> None:
        del args, kwargs
        (artifacts.path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (artifacts.path / "uncommitted.marker").write_text("old attempt", encoding="utf-8")
        (artifacts.path / "checkpoints" / "._interrupted.pt.tmp").write_text(
            "not a committed checkpoint",
            encoding="utf-8",
        )
        raise SimulatedProcessDeath

    monkeypatch.setattr("sb_pomdp.compare.train_single", die_before_first_checkpoint)
    with pytest.raises(SimulatedProcessDeath):
        run_comparisons(config, **arguments)

    assert (seed_root / "uncommitted.marker").is_file()
    assert not list(seed_root.rglob("*.pt"))
    original_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    original_torch = original_metadata["torch"]
    monkeypatch.undo()

    # Identity checks happen before quarantine: a retry under a different
    # runtime must leave the interrupted attempt exactly where it was.
    original_metadata["torch"] = "different-test-runtime"
    (run_dir / "metadata.json").write_text(json.dumps(original_metadata), encoding="utf-8")
    with pytest.raises(ConfigError, match="runtime identity differs"):
        run_comparisons(config, **arguments)
    assert (seed_root / "uncommitted.marker").is_file()

    failed_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    failed_metadata["torch"] = original_torch
    (run_dir / "metadata.json").write_text(json.dumps(failed_metadata), encoding="utf-8")
    _, summaries, _ = run_comparisons(config, **arguments)

    assert summaries[0]["status"] == "partial"
    assert summaries[0]["updates"] == 1
    assert (seed_root / "checkpoints" / "latest.pt").is_file()
    assert not (seed_root / "uncommitted.marker").exists()
    quarantined_markers = list(
        (run_dir / ".uncommitted-attempts").rglob("uncommitted.marker")
    )
    assert len(quarantined_markers) == 1
    assert quarantined_markers[0].read_text(encoding="utf-8") == "old attempt"
    assert (
        quarantined_markers[0].parent / "checkpoints" / "._interrupted.pt.tmp"
    ).is_file()


def test_missing_latest_checkpoint_resumes_from_newest_committed_step(tmp_path) -> None:
    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / "shard008"
    arguments = {
        "methods": ["score_transformer"],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "missing-latest-step-fallback",
        "campaign_id": "pytest-missing-latest-step-fallback",
        "bulk_index": 8,
        "verbose": False,
    }

    first_run, first_summaries, _ = run_comparisons(config, **arguments)
    assert first_summaries[0]["status"] == "partial"
    seed_root = (
        first_run.root / "full" / "score_transformer" / "masked_cartpole" / "seed_000"
    )
    checkpoints = seed_root / "checkpoints"
    committed_steps = sorted(checkpoints.glob("step_*.pt"))
    assert len(committed_steps) == 1
    (checkpoints / "latest.pt").unlink()

    _, resumed_summaries, _ = run_comparisons(config, **arguments)

    assert resumed_summaries[0]["status"] == "completed"
    assert resumed_summaries[0]["updates"] == 2
    assert (checkpoints / "latest.pt").is_file()


def test_failed_segment_keeps_source_sha256_so_the_next_resume_is_accepted(
    tmp_path,
    monkeypatch,
) -> None:
    config = _segmented_shard_config(tmp_path)
    run_dir = tmp_path / "campaign" / "shard002"
    arguments = {
        "methods": ["score_transformer"],
        "tasks": ["masked_cartpole"],
        "seeds": [0],
        "gradient_modes": ["full"],
        "max_updates": None,
        "segment_updates": 1,
        "run_dir": run_dir,
        "resume": True,
        "label": "segment-fingerprint",
        "campaign_id": "pytest-fingerprint",
        "bulk_index": 2,
        "verbose": False,
    }

    first_run, first_summaries, _ = run_comparisons(config, **arguments)
    fingerprint = json.loads(first_run.metadata_path.read_text(encoding="utf-8"))["source_sha256"]
    assert first_summaries[0]["status"] == "partial"

    def fail_device(_: str) -> None:
        raise RuntimeError("test device failure")

    monkeypatch.setattr("sb_pomdp.compare.resolve_device", fail_device)
    with pytest.raises(RuntimeError, match="test device failure"):
        run_comparisons(config, **arguments)
    failed_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert failed_metadata["status"] == "failed"
    assert failed_metadata["source_sha256"] == fingerprint
    monkeypatch.undo()

    third_run, third_summaries, _ = run_comparisons(config, **arguments)

    assert third_summaries[0]["status"] == "completed"
    assert third_summaries[0]["updates"] == 2
    metadata = json.loads(third_run.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["source_sha256"] == fingerprint
    assert [segment["status"] for segment in metadata["segments"]] == ["partial", "completed"]


def test_gradient_mode_cli_filter_selects_only_the_requested_mode(capsys) -> None:
    assert (
        comparison_main(
            [
                "--config",
                "config/local.json",
                "--gradient-modes",
                "tbptt_1",
                "--methods",
                "observation_mlp",
                "--tasks",
                "masked_pendulum",
                "--seeds",
                "0",
                "--validate-only",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["belief_gradient_modes"] == ["tbptt_1"]
    assert payload["eligible_method_tasks"] == [
        {
            "belief_gradient_mode": "tbptt_1",
            "method": "observation_mlp",
            "task": "masked_pendulum",
        }
    ]


def test_method_config_is_strict_and_records_exact_shard() -> None:
    config = load_config("config/local.json")
    expected = {
        "score_transformer": ("transformer", "diffusion"),
        "score_deepsets": ("deep_sets", "diffusion"),
        "score_gaussian": ("transformer", "gaussian"),
        "observation_mlp": ("transformer", "gaussian"),
        "gru": ("transformer", "diffusion"),
        "oracle_state": ("transformer", "gaussian"),
    }
    for gradient_mode in ("full", "tbptt_1"):
        for method, (encoder, policy) in expected.items():
            shard = _method_config(
                config,
                method,
                "masked_pendulum",
                7,
                gradient_mode=gradient_mode,
            )
            assert shard["comparison"]["belief_gradient_modes"] == [gradient_mode]
            assert shard["comparison"]["methods"] == [method]
            assert shard["experiment"]["tasks"] == ["masked_pendulum"]
            assert shard["experiment"]["seeds"] == [7]
            assert shard["model"]["belief_gradient_mode"] == gradient_mode
            assert shard["model"]["encoder_kind"] == encoder
            assert shard["model"]["continuous_policy_kind"] == policy


@pytest.mark.parametrize(
    "task",
    [
        "masked_cartpole",
        "masked_pendulum",
        "masked_mountain_car_continuous",
        "light_dark",
    ],
)
def test_production_score_and_gru_are_capacity_matched(task: str) -> None:
    from sb_pomdp.envs import make_env
    from sb_pomdp.policies import DiffusionPolicy, ScoreBeliefActorCritic
    from sb_pomdp.train_baselines import _make_model

    config = load_config("config/production.json")
    environment = make_env(task, config.environment_for_task(task), seed=0)
    action_spec = environment.action_spec
    score_config = _method_config(config, "score_transformer", task, 10, "full")
    score = ScoreBeliefActorCritic(
        observation_dim=environment.observation_dim,
        state_dim=environment.state_dim,
        action_kind=action_spec.kind,
        action_feature_dim=action_spec.feature_dim,
        discrete_actions=action_spec.n,
        action_low=None if action_spec.low is None else action_spec.low.tolist(),
        action_high=None if action_spec.high is None else action_spec.high.tolist(),
        model_config=score_config["model"],
    )
    gru_config = _method_config(config, "gru", task, 10, "full")
    gru = _make_model(
        environment,
        "gru",
        gru_config["model"],
        gru_config["comparison"],
        torch.device("cpu"),
    )

    score_parameters = sum(parameter.numel() for parameter in score.parameters())
    gru_parameters = sum(parameter.numel() for parameter in gru.parameters())
    assert abs(gru_parameters - score_parameters) / score_parameters < 0.0012
    if action_spec.is_continuous:
        assert isinstance(score.diffusion_policy, DiffusionPolicy)
        assert isinstance(gru.heads.diffusion_policy, DiffusionPolicy)
        assert score.diffusion_policy.num_steps == gru.heads.diffusion_policy.num_steps
        torch.testing.assert_close(score.diffusion_policy.betas, gru.heads.diffusion_policy.betas)


def test_setup_failure_writes_fetchable_diagnostics(tmp_path, monkeypatch) -> None:
    config = load_config("config/local.json").with_overrides(
        {"experiment.output_dir": str(tmp_path)}
    )

    def fail_device(_: str) -> None:
        raise RuntimeError("test device failure")

    monkeypatch.setattr("sb_pomdp.compare.resolve_device", fail_device)
    with pytest.raises(RuntimeError, match="test device failure"):
        run_comparisons(
            config,
            methods=["score_transformer"],
            tasks=["masked_cartpole"],
            seeds=[0],
            max_updates=1,
            label="setup-failure",
            verbose=False,
        )

    roots = list(tmp_path.iterdir())
    assert len(roots) == 1
    root = roots[0]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert manifest["status"] == metadata["status"] == "failed"
    assert manifest["runs"] == []
    assert (root / "failure.txt").is_file()
    _verify_staged_result(root)


def _aggregation_summary(
    method: str,
    seed: int,
    value: float,
    *,
    status: str = "completed",
) -> dict[str, object]:
    """Return the minimal completed summary shape :func:`_aggregate` consumes."""

    return {
        "belief_gradient_mode": "full",
        "method": method,
        "task": "task",
        "seed": seed,
        "status": status,
        "deterministic_mode": ("mean_chain" if method == "score_transformer" else "mean_action"),
        "deterministic_return_mean": value,
        "stochastic_return_mean": value + 1.0,
        "global_steps": 100,
        "wall_time_seconds": 2.0,
        "parameter_count": 10,
    }


def test_comparison_aggregate_uses_seed_paired_difference() -> None:
    summaries = [
        _aggregation_summary(method, seed, value)
        for method, seed, value in (
            ("score_transformer", 0, 10.0),
            ("score_transformer", 1, 20.0),
            ("observation_mlp", 0, 12.0),
            ("observation_mlp", 1, 19.0),
        )
    ]
    rows = _aggregate(summaries)
    baseline = next(row for row in rows if row["method"] == "observation_mlp")
    proposed = next(row for row in rows if row["method"] == "score_transformer")

    assert baseline["deterministic_return_mean"] == 15.5
    assert baseline["deterministic_proposed_return_mean"] == 15.0
    assert baseline["deterministic_paired_difference_mean"] == -0.5
    assert baseline["stochastic_return_mean"] == 16.5
    assert baseline["stochastic_proposed_return_mean"] == 16.0
    assert baseline["stochastic_paired_difference_mean"] == -0.5
    # Sign convention: score_transformer minus row method, so the proposal's own
    # row is exactly zero and every column is populated for it too.
    assert proposed["deterministic_paired_difference_mean"] == 0.0
    assert proposed["stochastic_paired_difference_mean"] == 0.0
    assert proposed["deterministic_paired_difference_std"] == 0.0
    assert proposed["deterministic_paired_difference_n_seeds"] == 2


def test_comparison_aggregate_reports_paired_std_and_seed_count() -> None:
    """Hand-computed five-seed example fixes the ddof=1 paired statistics."""

    proposed_values = [10.0, 12.0, 14.0, 16.0, 18.0]
    baseline_values = [11.0, 15.0, 14.0, 20.0, 18.0]
    summaries = [
        *(
            _aggregation_summary("score_transformer", seed, value)
            for seed, value in enumerate(proposed_values)
        ),
        *(
            _aggregation_summary("observation_mlp", seed, value)
            for seed, value in enumerate(baseline_values)
        ),
    ]

    rows = _aggregate(summaries)
    baseline = next(row for row in rows if row["method"] == "observation_mlp")

    # Per-seed paired differences (proposal - baseline): -1, -3, 0, -4, 0.
    # mean = -8/5 = -1.6; sum of squared deviations = 0.36 + 1.96 + 2.56 + 5.76
    # + 2.56 = 13.2; ddof=1 variance = 13.2/4 = 3.3; std = sqrt(3.3).
    assert baseline["deterministic_paired_difference_n_seeds"] == 5
    assert baseline["deterministic_paired_difference_mean"] == pytest.approx(-1.6)
    assert baseline["deterministic_paired_difference_std"] == pytest.approx(3.3**0.5)
    # The stochastic columns shift both arms by the same constant, so the paired
    # differences - and therefore their std - are identical.
    assert baseline["stochastic_paired_difference_n_seeds"] == 5
    assert baseline["stochastic_paired_difference_mean"] == pytest.approx(-1.6)
    assert baseline["stochastic_paired_difference_std"] == pytest.approx(3.3**0.5)
    # The per-seed return std is the plain ddof=1 std of the five seed means:
    # mean = 15.6, sum of squared deviations = 21.16 + 0.36 + 2.56 + 19.36
    # + 5.76 = 49.2, variance = 49.2/4 = 12.3.
    assert baseline["seeds_completed"] == 5
    assert baseline["deterministic_return_mean"] == pytest.approx(15.6)
    assert baseline["deterministic_return_std"] == pytest.approx(12.3**0.5)


def test_comparison_aggregate_leaves_single_seed_std_empty() -> None:
    """One completed seed carries no dispersion, so std is null rather than 0.0."""

    summaries = [
        _aggregation_summary("score_transformer", 0, 10.0),
        _aggregation_summary("observation_mlp", 0, 12.0),
    ]

    rows = _aggregate(summaries)
    baseline = next(row for row in rows if row["method"] == "observation_mlp")

    assert baseline["seeds_completed"] == 1
    assert baseline["deterministic_return_std"] is None
    assert baseline["stochastic_return_std"] is None
    assert baseline["deterministic_paired_difference_n_seeds"] == 1
    assert baseline["deterministic_paired_difference_mean"] == -2.0
    assert baseline["deterministic_paired_difference_std"] is None
    assert baseline["stochastic_paired_difference_std"] is None


def test_comparison_csv_writes_empty_cells_for_single_seed_std(tmp_path) -> None:
    """The CSV schema stays fixed; an unavailable std becomes an empty cell."""

    from sb_pomdp.artifacts import RunArtifacts

    rows = _aggregate(
        [
            _aggregation_summary("score_transformer", 0, 10.0),
            _aggregation_summary("observation_mlp", 0, 12.0),
        ]
    )
    run = RunArtifacts.create(tmp_path, "std-columns", timestamp="20260818T120000")
    run.write_comparison_csv(rows, COMPARISON_FIELDS)
    run.write_comparison(rows)

    with run.comparison_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPARISON_FIELDS)
        written = {row["method"]: row for row in reader}
    assert written["observation_mlp"]["deterministic_return_std"] == ""
    assert written["observation_mlp"]["deterministic_paired_difference_std"] == ""
    assert written["observation_mlp"]["deterministic_paired_difference_n_seeds"] == "1"
    payload = json.loads(run.comparison_json_path.read_text(encoding="utf-8"))
    assert all(row["stochastic_paired_difference_std"] is None for row in payload)


def test_selection_rejects_a_seed_the_config_does_not_declare() -> None:
    config = load_config("config/production.json")

    with pytest.raises(ConfigError, match=r"requested seeds are not enabled"):
        _selection(config, None, None, [7])
    with pytest.raises(ConfigError, match=r"configured seeds: \[10, 11, 12, 13, 14\]"):
        _selection(config, None, None, [10, 7])

    assert _selection(config, None, None, [12])[2] == [12]
    assert _selection(config, None, None, None)[2] == [10, 11, 12, 13, 14]


def test_comparison_cli_rejects_an_unconfigured_seed(capsys) -> None:
    assert (
        comparison_main(
            [
                "--config",
                "config/production.json",
                "--methods",
                "score_transformer",
                "--tasks",
                "masked_cartpole",
                "--seeds",
                "0",
                "--validate-only",
            ]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert "requested seeds are not enabled by the config: [0]" in error
    assert "configured seeds: [10, 11, 12, 13, 14]" in error


def test_all_comparison_methods_complete_one_update_and_write_nested_artifacts(
    tmp_path,
) -> None:
    config = load_config("config/local.json").with_overrides(
        {"experiment.output_dir": str(tmp_path)}
    )
    methods = list(config["comparison"]["methods"])
    tasks = list(config["experiment"]["tasks"])
    run, summaries, skipped = run_comparisons(
        config,
        methods=methods,
        tasks=tasks,
        seeds=[0],
        max_updates=1,
        label="pytest-comparison",
        verbose=False,
        campaign_id="pytest-campaign",
    )

    assert len(summaries) == 46
    assert all(summary["status"] == "completed" for summary in summaries)
    assert skipped == [
        {
            "belief_gradient_mode": gradient_mode,
            "method": "score_gaussian",
            "task": "masked_cartpole",
            "reason": "score_gaussian only differs from the proposal on continuous actions",
        }
        for gradient_mode in ("full", "tbptt_1")
    ]
    _verify_staged_result(run.root)
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["campaign_id"] == "pytest-campaign"
    assert manifest["belief_gradient_modes"] == ["full", "tbptt_1"]
    assert len(manifest["eligible_runs"]) == 46

    with run.comparison_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPARISON_FIELDS)
        comparison_rows = list(reader)
    assert len(comparison_rows) == 46
    assert {row["belief_gradient_mode"] for row in comparison_rows} == {"full", "tbptt_1"}
    assert all(row["deterministic_mode"] for row in comparison_rows)
    assert all(row["stochastic_return_mean"] for row in comparison_rows)
    # One seed per condition: the paired difference exists but its std does not.
    assert all(row["deterministic_paired_difference_n_seeds"] == "1" for row in comparison_rows)
    assert all(row["deterministic_paired_difference_std"] == "" for row in comparison_rows)
    assert all(row["stochastic_paired_difference_std"] == "" for row in comparison_rows)
    assert all(row["deterministic_return_std"] == "" for row in comparison_rows)

    with run.summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 46
    # The alias is populated from the same deterministic-readout value, never
    # replacing the legacy key.
    assert all(
        row["deterministic_readout_return_mean"] == row["evaluation_return_mean"]
        for row in summary_rows
    )
    assert all(row["deterministic_readout_return_mean"] for row in summary_rows)
    assert all(
        summary["deterministic_readout_return_mean"] == summary["evaluation_return_mean"]
        for summary in summaries
    )

    for summary in summaries:
        seed_directory = run.root / summary["result_dir"]
        shard_config = load_config(seed_directory / "resolved_config.json")
        assert summary["result_dir"].split("/")[0] == summary["belief_gradient_mode"]
        assert shard_config["comparison"]["belief_gradient_modes"] == [
            summary["belief_gradient_mode"]
        ]
        assert shard_config["comparison"]["methods"] == [summary["method"]]
        assert shard_config["experiment"]["tasks"] == [summary["task"]]
        assert shard_config["experiment"]["seeds"] == [0]
        assert shard_config["model"]["belief_gradient_mode"] == summary["belief_gradient_mode"]
        with (seed_directory / "evaluation.csv").open("r", encoding="utf-8", newline="") as handle:
            modes = {row["mode"] for row in csv.DictReader(handle)}
        deterministic_mode = (
            "greedy"
            if summary["task"] == "masked_cartpole"
            else "mean_chain"
            if (
                str(summary["method"]).startswith("score_")
                and summary["method"] != "score_gaussian"
            )
            or summary["method"] == "gru"
            else "mean_action"
        )
        assert modes == {deterministic_mode, "stochastic"}


def test_production_light_dark_nd5_reaches_score_and_gru_training(tmp_path) -> None:
    config = load_config("config/production.json").with_overrides(
        {
            "experiment.output_dir": str(tmp_path),
            "experiment.device": "cpu",
            "experiment.deterministic": True,
            "experiment.eval_episodes": 1,
            "experiment.eval_interval_updates": 1,
            "experiment.checkpoint_interval_updates": 1,
            "environment.tasks.light_dark.horizon": 2,
            "model.num_particles": 2,
            "model.langevin_steps": 1,
            "model.energy_hidden": [8],
            "model.d_model": 8,
            "model.num_heads": 1,
            "model.num_transformer_layers": 1,
            "model.transformer_ff_dim": 16,
            "model.policy_hidden": [8],
            "model.diffusion_steps": 1,
            "ppo.total_steps": 4,
            "ppo.num_envs": 2,
            "ppo.rollout_steps": 2,
            "ppo.epochs": 1,
            "ppo.minibatch_size": 2,
            "ppo.sequence_microbatch_size": 2,
            "comparison.gru_encoder_hidden": [8, 8],
            "comparison.gru_hidden_dim": 8,
        }
    )

    _, summaries, skipped = run_comparisons(
        config,
        methods=["score_transformer", "gru"],
        tasks=["light_dark"],
        seeds=[10],
        gradient_modes=["tbptt_1"],
        max_updates=1,
        label="nd5-training-smoke",
        verbose=False,
    )

    assert skipped == []
    assert {summary["method"] for summary in summaries} == {"score_transformer", "gru"}
    assert all(summary["status"] == "completed" for summary in summaries)
    assert all(summary["global_steps"] == 4 for summary in summaries)
