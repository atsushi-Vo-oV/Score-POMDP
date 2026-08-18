from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sb_pomdp.aggregate import _parse_args, aggregate_runs
from sb_pomdp.compare import COMPARISON_FIELDS, _method_config
from sb_pomdp.config import ExperimentConfig, load_config


def _summary(
    method: str,
    task: str,
    seed: int,
    *,
    gradient_mode: str = "full",
) -> dict[str, object]:
    return {
        "belief_gradient_mode": gradient_mode,
        "method": method,
        "task": task,
        "seed": seed,
        "status": "completed",
        "updates": 1,
        "global_steps": 16,
        "episodes_completed": 1,
        "training_return_mean": 1.0,
        "evaluation_return_mean": 2.0,
        "deterministic_readout_return_mean": 2.0,
        "parameter_count": 10,
        "wall_time_seconds": 0.1,
        "result_dir": f"{gradient_mode}/{method}/{task}/seed_{seed:03d}",
        "error": "",
        "deterministic_mode": (
            "mean_chain" if method in {"score_transformer", "score_deepsets"} else "mean_action"
        ),
        "deterministic_return_mean": 2.0,
        "stochastic_return_mean": 1.5,
    }


def _write_source_root(
    root: Path,
    config: ExperimentConfig,
    summaries: list[dict[str, object]],
    *,
    campaign_id: str = "test-campaign",
    source_sha256: str = "a" * 64,
) -> Path:
    root.mkdir()
    (root / "resolved_config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "belief_gradient_modes": config["comparison"]["belief_gradient_modes"],
                "methods": config["comparison"]["methods"],
                "campaign_id": campaign_id,
                "runs": summaries,
            }
        ),
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(
        json.dumps({"source_sha256": source_sha256}),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(json.dumps(summaries), encoding="utf-8")
    (root / "summary.csv").write_text("status\ncompleted\n", encoding="utf-8")
    (root / "comparison.csv").write_text("method\n", encoding="utf-8")
    (root / "comparison.json").write_text("[]", encoding="utf-8")

    for summary in summaries:
        seed_root = root / str(summary["result_dir"])
        (seed_root / "checkpoints").mkdir(parents=True)
        for filename in ("run.log", "metrics.csv", "evaluation.csv"):
            (seed_root / filename).write_text("ok\n", encoding="utf-8")
        (seed_root / "metadata.json").write_text(
            json.dumps({"source_sha256": source_sha256}),
            encoding="utf-8",
        )
        method_config = _method_config(
            config,
            str(summary["method"]),
            str(summary["task"]),
            int(summary["seed"]),
            gradient_mode=str(summary["belief_gradient_mode"]),
        )
        (seed_root / "resolved_config.json").write_text(
            json.dumps(method_config.to_dict()),
            encoding="utf-8",
        )
        (seed_root / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
        np.savez(seed_root / "arrays.npz", reward=np.asarray([1.0], dtype=np.float32))
    return root


def _config(
    *,
    methods: list[str],
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
    gradient_modes: list[str] | None = None,
    label: str = "master",
    output_dir: str = "results",
) -> ExperimentConfig:
    selected_gradient_modes = gradient_modes or ["full"]
    return load_config("config/local.json").with_overrides(
        {
            "comparison.methods": methods,
            "comparison.belief_gradient_modes": selected_gradient_modes,
            "model.belief_gradient_mode": selected_gradient_modes[0],
            "experiment.tasks": tasks or ["masked_pendulum"],
            "experiment.seeds": seeds or [0],
            "experiment.label": label,
            "experiment.output_dir": output_dir,
        }
    )


def test_aggregate_requires_master_config_and_records_exact_coverage(tmp_path: Path) -> None:
    gradient_modes = ["full", "tbptt_1"]
    master = _config(
        methods=["score_transformer", "observation_mlp"],
        gradient_modes=gradient_modes,
    )
    score_config = _config(
        methods=["score_transformer"],
        gradient_modes=gradient_modes,
        label="score-shard",
        output_dir="remote-results",
    )
    baseline_config = _config(
        methods=["observation_mlp"],
        gradient_modes=gradient_modes,
        label="baseline-shard",
        output_dir="downloaded-results",
    )
    task = "masked_pendulum"
    sources = [
        _write_source_root(
            tmp_path / "score-source",
            score_config,
            [
                _summary("score_transformer", task, 0, gradient_mode=gradient_mode)
                for gradient_mode in gradient_modes
            ],
        ),
        _write_source_root(
            tmp_path / "baseline-source",
            baseline_config,
            [
                _summary("observation_mlp", task, 0, gradient_mode=gradient_mode)
                for gradient_mode in gradient_modes
            ],
        ),
    ]

    report = aggregate_runs(
        sources,
        expected_config=master,
        output_dir=tmp_path / "reports",
        label="aggregate-test",
    )

    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    resolved = json.loads(report.resolved_config_path.read_text(encoding="utf-8"))
    for artifact in (manifest, metadata):
        assert artifact["coverage_complete"] is True
        assert artifact["belief_gradient_modes"] == gradient_modes
        assert artifact["expected_gradient_mode_method_task_seed_runs"] == 4
        assert artifact["unique_gradient_mode_method_task_seed_runs"] == 4
        assert artifact["coverage"] == {
            "complete": True,
            "expected_runs": 4,
            "observed_runs": 4,
            "source_count": 2,
            "belief_gradient_modes": gradient_modes,
            "campaign_id": "test-campaign",
            "source_sha256": "a" * 64,
            "sources": [
                {
                    "source": str(sources[0].resolve()),
                    "expected_runs": 2,
                    "observed_runs": 2,
                    "belief_gradient_modes": gradient_modes,
                    "complete": True,
                },
                {
                    "source": str(sources[1].resolve()),
                    "expected_runs": 2,
                    "observed_runs": 2,
                    "belief_gradient_modes": gradient_modes,
                    "complete": True,
                },
            ],
        }
        assert artifact["campaign_id"] == "test-campaign"
        assert artifact["source_sha256"] == "a" * 64
    assert resolved == master.to_dict()
    comparison = json.loads(report.comparison_json_path.read_text(encoding="utf-8"))
    assert len(comparison) == 4
    assert {row["belief_gradient_mode"] for row in comparison} == set(gradient_modes)


def _strip_post_campaign_keys(root: Path) -> None:
    """Rewrite every resolved config exactly as the pre-F-1 trainer wrote it."""

    for path in sorted(root.rglob("resolved_config.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["ppo"]["bootstrap_on_truncation"]
        path.write_text(json.dumps(payload), encoding="utf-8")


def _strip_post_campaign_summary_keys(root: Path) -> None:
    """Rewrite the manifest/summary rows exactly as the pre-A-7 trainer wrote them."""

    for name in ("manifest.json", "summary.json"):
        path = root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["runs"] if name == "manifest.json" else payload
        for row in rows:
            del row["deterministic_readout_return_mean"]
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_accepts_a_legacy_source_config_at_its_defaults(tmp_path: Path) -> None:
    master = _config(methods=["score_transformer"])
    source = _write_source_root(
        tmp_path / "legacy-source",
        master,
        [_summary("score_transformer", "masked_pendulum", 0)],
    )
    _strip_post_campaign_keys(source)

    report = aggregate_runs(
        [source],
        expected_config=master,
        output_dir=tmp_path / "reports",
        label="legacy-aggregate",
    )

    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["coverage_complete"] is True
    assert manifest["unique_gradient_mode_method_task_seed_runs"] == 1


def test_aggregate_accepts_legacy_summaries_without_the_readout_alias(tmp_path: Path) -> None:
    """A shard written before the summary alias existed stays aggregatable."""

    master = _config(methods=["score_transformer"])
    source = _write_source_root(
        tmp_path / "legacy-summary-source",
        master,
        [_summary("score_transformer", "masked_pendulum", 0)],
    )
    _strip_post_campaign_keys(source)
    _strip_post_campaign_summary_keys(source)

    report = aggregate_runs(
        [source],
        expected_config=master,
        output_dir=tmp_path / "reports",
        label="legacy-summary-aggregate",
    )

    summaries = json.loads(report.summary_json_path.read_text(encoding="utf-8"))
    assert len(summaries) == 1
    assert summaries[0]["evaluation_return_mean"] == 2.0
    assert summaries[0]["deterministic_readout_return_mean"] == 2.0
    with report.summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["deterministic_readout_return_mean"] == "2.0"


def test_aggregate_writes_the_paired_std_and_seed_count_columns(tmp_path: Path) -> None:
    """The new comparison columns flow through the shared _aggregate helper."""

    seeds = [0, 1, 2]
    master = _config(methods=["score_transformer", "observation_mlp"], seeds=seeds)
    task = "masked_pendulum"
    sources = [
        _write_source_root(
            tmp_path / "score-source",
            _config(methods=["score_transformer"], seeds=seeds, label="score-shard"),
            [_summary("score_transformer", task, seed) for seed in seeds],
        ),
        _write_source_root(
            tmp_path / "baseline-source",
            _config(methods=["observation_mlp"], seeds=seeds, label="baseline-shard"),
            [_summary("observation_mlp", task, seed) for seed in seeds],
        ),
    ]

    report = aggregate_runs(
        sources,
        expected_config=master,
        output_dir=tmp_path / "reports",
        label="paired-columns",
    )

    with report.comparison_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPARISON_FIELDS)
        rows = {row["method"]: row for row in reader}
    # Every seed shares the same fixture returns, so the paired differences are
    # all zero: the mean is 0.0 and the ddof=1 std over three seeds is 0.0.
    assert rows["observation_mlp"]["deterministic_paired_difference_n_seeds"] == "3"
    assert float(rows["observation_mlp"]["deterministic_paired_difference_mean"]) == 0.0
    assert float(rows["observation_mlp"]["deterministic_paired_difference_std"]) == 0.0
    assert float(rows["observation_mlp"]["stochastic_paired_difference_std"]) == 0.0
    assert float(rows["observation_mlp"]["deterministic_return_std"]) == 0.0


def test_aggregate_rejects_nonselector_config_difference(tmp_path: Path) -> None:
    master = _config(methods=["score_transformer"])
    changed = master.with_overrides({"ppo.learning_rate": 0.0004})
    source = _write_source_root(
        tmp_path / "changed-source",
        changed,
        [_summary("score_transformer", "masked_pendulum", 0)],
    )

    with pytest.raises(RuntimeError, match=r"ppo\.learning_rate"):
        aggregate_runs(
            [source],
            expected_config=master,
            output_dir=tmp_path / "reports",
        )


def test_aggregate_rejects_tampered_completed_run_config(tmp_path: Path) -> None:
    master = _config(methods=["score_transformer"])
    summary = _summary("score_transformer", "masked_pendulum", 0)
    source = _write_source_root(tmp_path / "tampered-run-source", master, [summary])
    run_config_path = source / str(summary["result_dir"]) / "resolved_config.json"
    tampered = load_config(run_config_path).with_overrides({"ppo.learning_rate": 0.0004})
    run_config_path.write_text(json.dumps(tampered.to_dict()), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"run config differs.*ppo\.learning_rate"):
        aggregate_runs(
            [source],
            expected_config=master,
            output_dir=tmp_path / "reports",
        )


def test_aggregate_rejects_manifest_coverage_that_disagrees_with_source_config(
    tmp_path: Path,
) -> None:
    master = _config(methods=["score_transformer"])
    source = _write_source_root(
        tmp_path / "wrong-seed-source",
        master,
        [_summary("score_transformer", "masked_pendulum", 1)],
    )

    with pytest.raises(RuntimeError, match="source resolved config"):
        aggregate_runs(
            [source],
            expected_config=master,
            output_dir=tmp_path / "reports",
        )


def test_aggregate_rejects_incomplete_combined_master_coverage(tmp_path: Path) -> None:
    master = _config(methods=["score_transformer", "observation_mlp"])
    source_config = _config(methods=["score_transformer"])
    source = _write_source_root(
        tmp_path / "only-score-source",
        source_config,
        [_summary("score_transformer", "masked_pendulum", 0)],
    )

    with pytest.raises(RuntimeError, match="missing="):
        aggregate_runs(
            [source],
            expected_config=master,
            output_dir=tmp_path / "reports",
        )


def test_aggregate_cli_requires_config() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--runs", "one"])


def test_aggregate_rejects_mixed_campaigns(tmp_path: Path) -> None:
    master = _config(methods=["score_transformer", "observation_mlp"])
    task = "masked_pendulum"
    first = _write_source_root(
        tmp_path / "first-campaign",
        _config(methods=["score_transformer"]),
        [_summary("score_transformer", task, 0)],
        campaign_id="campaign-a",
    )
    second = _write_source_root(
        tmp_path / "second-campaign",
        _config(methods=["observation_mlp"]),
        [_summary("observation_mlp", task, 0)],
        campaign_id="campaign-b",
    )

    with pytest.raises(RuntimeError, match="common non-empty campaign_id"):
        aggregate_runs(
            [first, second],
            expected_config=master,
            output_dir=tmp_path / "reports",
        )


def test_aggregate_rejects_mixed_source_code(tmp_path: Path) -> None:
    master = _config(methods=["score_transformer", "observation_mlp"])
    task = "masked_pendulum"
    first = _write_source_root(
        tmp_path / "first-source",
        _config(methods=["score_transformer"]),
        [_summary("score_transformer", task, 0)],
        source_sha256="a" * 64,
    )
    second = _write_source_root(
        tmp_path / "second-source",
        _config(methods=["observation_mlp"]),
        [_summary("observation_mlp", task, 0)],
        source_sha256="b" * 64,
    )

    with pytest.raises(RuntimeError, match="different source code"):
        aggregate_runs(
            [first, second],
            expected_config=master,
            output_dir=tmp_path / "reports",
        )
