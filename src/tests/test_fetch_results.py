from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sb_pomdp.fetch_results import (
    _parse_args,
    _verify_staged_result,
    fetch_campaign,
    fetch_result,
)


def _write_seed_run(root: Path, relative: str, *, with_evaluation: bool = True) -> Path:
    """Write one task/seed artifact tree exactly as a trainer segment leaves it."""

    seed_directory = root.joinpath(*relative.split("/"))
    (seed_directory / "checkpoints").mkdir(parents=True)
    names = ["run.log", "metrics.csv"]
    if with_evaluation:
        names.append("evaluation.csv")
    for name in names:
        (seed_directory / name).write_text("ok\n", encoding="utf-8")
    for name in ("metadata.json", "resolved_config.json"):
        (seed_directory / name).write_text("{}", encoding="utf-8")
    (seed_directory / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
    np.savez(seed_directory / "arrays.npz", reward=np.asarray([1.0], dtype=np.float32))
    return seed_directory


def _write_result_root(
    root: Path,
    *,
    status: str,
    relative: str = "full/score_transformer/light_dark/seed_010",
    with_evaluation: bool = True,
    campaign_id: str | None = None,
    bulk_index: int | None = None,
) -> None:
    """Write one comparison result root whose manifest reports ``status``."""

    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": status,
        "methods": ["score_transformer"],
        "runs": [{"status": status, "result_dir": relative}],
    }
    if campaign_id is not None:
        manifest["campaign_id"] = campaign_id
    if bulk_index is not None:
        manifest["bulk_index"] = bulk_index
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("metadata.json", "resolved_config.json"):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "summary.json").write_text("[]", encoding="utf-8")
    (root / "summary.csv").write_text(f"status\n{status}\n", encoding="utf-8")
    (root / "comparison.csv").write_text("method\nscore_transformer\n", encoding="utf-8")
    (root / "comparison.json").write_text("[]", encoding="utf-8")
    _write_seed_run(root, relative, with_evaluation=with_evaluation)


def test_fetch_result_uses_argument_list_and_verifies_archives(tmp_path, monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        captured.append(command)
        destination = Path(command[-1])
        destination.mkdir()
        seed_directory = destination / "method" / "task" / "seed_000"
        (seed_directory / "checkpoints").mkdir(parents=True)
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "methods": ["method"],
                    "runs": [
                        {
                            "status": "completed",
                            "result_dir": "method/task/seed_000",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        for name in ("metadata.json", "resolved_config.json", "summary.json"):
            (destination / name).write_text("{}", encoding="utf-8")
        (destination / "summary.csv").write_text("status\ncompleted\n", encoding="utf-8")
        (destination / "comparison.csv").write_text("method\nmethod\n", encoding="utf-8")
        (destination / "comparison.json").write_text("[]", encoding="utf-8")
        for name in ("run.log", "metrics.csv", "evaluation.csv"):
            (seed_directory / name).write_text("ok\n", encoding="utf-8")
        for name in ("metadata.json", "resolved_config.json"):
            (seed_directory / name).write_text("{}", encoding="utf-8")
        (seed_directory / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
        np.savez(
            seed_directory / "arrays.npz",
            reward=np.asarray([1.0], dtype=np.float32),
        )

    monkeypatch.setattr("sb_pomdp.fetch_results.subprocess.run", fake_run)
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Host genkai\n", encoding="utf-8")
    result = fetch_result(
        host="genkai",
        remote_results_dir="/home/project/results",
        run_name="20260808T120000_debug",
        local_results_dir=tmp_path / "local-results",
        ssh_config=ssh_config,
    )
    assert result.is_dir()
    assert captured[0][0] == "scp"
    assert captured[0][-2] == "genkai:/home/project/results/20260808T120000_debug"


def test_comparison_verification_requires_comparison_artifacts(tmp_path) -> None:
    for name in ("metadata.json", "resolved_config.json", "summary.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "summary.csv").write_text("status\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "completed", "methods": ["score_transformer"], "runs": []}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="aggregate artifacts"):
        _verify_staged_result(tmp_path)


def test_comparison_verification_accepts_gradient_mode_result_hierarchy(tmp_path) -> None:
    relative = "full/score_transformer/light_dark/seed_000"
    seed_directory = tmp_path.joinpath(*relative.split("/"))
    (seed_directory / "checkpoints").mkdir(parents=True)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "methods": ["score_transformer"],
                "runs": [{"status": "completed", "result_dir": relative}],
            }
        ),
        encoding="utf-8",
    )
    for name in ("metadata.json", "resolved_config.json", "summary.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "summary.csv").write_text("status\ncompleted\n", encoding="utf-8")
    (tmp_path / "comparison.csv").write_text("method\nscore_transformer\n", encoding="utf-8")
    (tmp_path / "comparison.json").write_text("[]", encoding="utf-8")
    for name in ("run.log", "metrics.csv", "evaluation.csv"):
        (seed_directory / name).write_text("ok\n", encoding="utf-8")
    for name in ("metadata.json", "resolved_config.json"):
        (seed_directory / name).write_text("{}", encoding="utf-8")
    (seed_directory / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
    np.savez(seed_directory / "arrays.npz", reward=np.asarray([1.0], dtype=np.float32))

    _verify_staged_result(tmp_path)


def test_fetch_campaign_copies_and_checks_all_bulk_indices(tmp_path, monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        captured.append(command)
        destination = Path(command[-1])
        destination.mkdir()
        for bulk_index in (1, 2):
            root = destination / f"root-{bulk_index:03d}"
            root.mkdir()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "methods": ["score_transformer"],
                        "campaign_id": "campaign-1",
                        "bulk_index": bulk_index,
                        "runs": [],
                    }
                ),
                encoding="utf-8",
            )
            for name in ("metadata.json", "resolved_config.json"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "summary.json").write_text("[]", encoding="utf-8")
            (root / "failure.txt").write_text("diagnostic", encoding="utf-8")

    monkeypatch.setattr("sb_pomdp.fetch_results.subprocess.run", fake_run)
    result = fetch_campaign(
        host="genkai",
        remote_campaigns_dir="/home/project/results/campaigns",
        campaign_id="campaign-1",
        local_results_dir=tmp_path / "local-results",
        ssh_config=None,
        expected_shards=2,
    )

    assert result == (tmp_path / "local-results" / "campaigns" / "campaign-1").resolve()
    assert len(list(result.iterdir())) == 2
    assert captured[0][-2] == "genkai:/home/project/results/campaigns/campaign-1"


def test_fetch_campaign_accepts_an_explicit_sparse_index_set(tmp_path, monkeypatch) -> None:
    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        destination = Path(command[-1])
        destination.mkdir()
        for bulk_index in (11, 31, 51):
            root = destination / f"root-{bulk_index:03d}"
            root.mkdir()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "methods": ["score_transformer"],
                        "campaign_id": "pilot-1",
                        "bulk_index": bulk_index,
                        "runs": [],
                    }
                ),
                encoding="utf-8",
            )
            for name in ("metadata.json", "resolved_config.json"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "summary.json").write_text("[]", encoding="utf-8")
            (root / "failure.txt").write_text("diagnostic", encoding="utf-8")

    monkeypatch.setattr("sb_pomdp.fetch_results.subprocess.run", fake_run)
    result = fetch_campaign(
        host="genkai",
        remote_campaigns_dir="/home/project/results/campaigns",
        campaign_id="pilot-1",
        local_results_dir=tmp_path / "local-results",
        ssh_config=None,
        expected_indices=[11, 51, 31],
    )

    assert len(list(result.iterdir())) == 3


def test_fetch_campaign_rejects_conflicting_expected_index_forms(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        fetch_campaign(
            host="genkai",
            remote_campaigns_dir="/home/project/results/campaigns",
            campaign_id="pilot-1",
            local_results_dir=tmp_path,
            ssh_config=None,
            expected_shards=3,
            expected_indices=[11, 51, 31],
        )


@pytest.mark.parametrize(
    ("host", "remote"),
    [
        ("-oProxyCommand=bad", "/home/project/results"),
        ("genkai", "/home/project/../results"),
        ("genkai", "/home/project/results;bad"),
    ],
)
def test_fetch_result_rejects_unsafe_remote_arguments(tmp_path, host: str, remote: str) -> None:
    with pytest.raises(ValueError):
        fetch_result(
            host=host,
            remote_results_dir=remote,
            run_name="safe-run",
            local_results_dir=tmp_path,
            ssh_config=None,
        )


@pytest.mark.parametrize("run_name", ["bad name", "bad;name", "$(bad)", "../bad"])
def test_fetch_result_rejects_unsafe_run_name(tmp_path, run_name: str) -> None:
    with pytest.raises(ValueError):
        fetch_result(
            host="genkai",
            remote_results_dir="/home/project/results",
            run_name=run_name,
            local_results_dir=tmp_path,
            ssh_config=None,
        )


def test_verification_rejects_a_partial_result_by_default(tmp_path) -> None:
    _write_result_root(tmp_path, status="partial", with_evaluation=False)

    with pytest.raises(RuntimeError, match="--allow-partial"):
        _verify_staged_result(tmp_path)


def test_verification_accepts_a_partial_result_only_when_opted_in(tmp_path) -> None:
    _write_result_root(tmp_path, status="partial", with_evaluation=False)

    assert _verify_staged_result(tmp_path, allow_partial=True) == "partial"


def test_verification_returns_the_accepted_status_of_a_completed_result(tmp_path) -> None:
    _write_result_root(tmp_path, status="completed")

    assert _verify_staged_result(tmp_path) == "completed"
    assert _verify_staged_result(tmp_path, allow_partial=True) == "completed"


@pytest.mark.parametrize(
    ("root_status", "run_status"),
    [
        ("completed", "partial"),
        ("partial", "failed"),
        ("partial", "completed"),
    ],
)
def test_trained_root_rejects_inconsistent_task_run_statuses(
    tmp_path: Path,
    root_status: str,
    run_status: str,
) -> None:
    _write_result_root(
        tmp_path,
        status=root_status,
        with_evaluation=root_status == "completed",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"][0]["status"] = run_status
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest contains|contains no partial"):
        _verify_staged_result(tmp_path, allow_partial=True)


def test_trained_root_rejects_a_non_object_task_run(tmp_path: Path) -> None:
    _write_result_root(tmp_path, status="completed")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"] = ["not-a-run-object"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TypeError, match="non-object run entries"):
        _verify_staged_result(tmp_path)


def test_partial_verification_still_requires_manifest_artifacts_and_archives(tmp_path) -> None:
    relative = "full/score_transformer/light_dark/seed_010"
    _write_result_root(tmp_path, status="partial", relative=relative, with_evaluation=False)
    (tmp_path.joinpath(*relative.split("/")) / "checkpoints" / "latest.pt").unlink()

    with pytest.raises(RuntimeError, match="checkpoints/latest.pt"):
        _verify_staged_result(tmp_path, allow_partial=True)


def test_partial_verification_still_requires_the_root_summary(tmp_path) -> None:
    _write_result_root(tmp_path, status="partial", with_evaluation=False)
    (tmp_path / "summary.csv").unlink()

    with pytest.raises(RuntimeError, match="summary.csv"):
        _verify_staged_result(tmp_path, allow_partial=True)


def test_fetch_result_annotates_an_accepted_partial_root(tmp_path, monkeypatch, capsys) -> None:
    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        _write_result_root(Path(command[-1]), status="partial", with_evaluation=False)

    monkeypatch.setattr("sb_pomdp.fetch_results.subprocess.run", fake_run)
    result = fetch_result(
        host="genkai",
        remote_results_dir="/home/project/results",
        run_name="shard011",
        local_results_dir=tmp_path / "local-results",
        ssh_config=None,
        allow_partial=True,
    )

    assert result.is_dir()
    assert "accepted 1 partial result root(s): shard011" in capsys.readouterr().err


def test_fetch_campaign_annotates_only_the_partial_roots(tmp_path, monkeypatch, capsys) -> None:
    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        destination = Path(command[-1])
        destination.mkdir()
        for bulk_index, status in ((11, "completed"), (12, "partial")):
            _write_result_root(
                destination / f"shard{bulk_index:03d}",
                status=status,
                with_evaluation=status == "completed",
                campaign_id="pilot-1",
                bulk_index=bulk_index,
            )

    monkeypatch.setattr("sb_pomdp.fetch_results.subprocess.run", fake_run)
    result = fetch_campaign(
        host="genkai",
        remote_campaigns_dir="/home/project/results/campaigns",
        campaign_id="pilot-1",
        local_results_dir=tmp_path / "local-results",
        ssh_config=None,
        expected_indices=[11, 12],
        allow_partial=True,
    )

    assert len(list(result.iterdir())) == 2
    assert "accepted 1 partial result root(s): shard012" in capsys.readouterr().err


def test_fetch_campaign_rejects_a_partial_root_without_the_flag(tmp_path, monkeypatch) -> None:
    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        destination = Path(command[-1])
        destination.mkdir()
        _write_result_root(
            destination / "shard011",
            status="partial",
            with_evaluation=False,
            campaign_id="pilot-1",
            bulk_index=11,
        )

    monkeypatch.setattr("sb_pomdp.fetch_results.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="--allow-partial"):
        fetch_campaign(
            host="genkai",
            remote_campaigns_dir="/home/project/results/campaigns",
            campaign_id="pilot-1",
            local_results_dir=tmp_path / "local-results",
            ssh_config=None,
            expected_indices=[11],
        )
    assert not (tmp_path / "local-results" / "campaigns" / "pilot-1").exists()


def test_allow_partial_is_opt_in_on_the_command_line() -> None:
    strict = _parse_args(
        ["--host", "genkai", "--remote-results-dir", "/results", "--run-name", "shard011"]
    )
    permissive = _parse_args(
        [
            "--host",
            "genkai",
            "--remote-results-dir",
            "/results",
            "--run-name",
            "shard011",
            "--allow-partial",
        ]
    )

    assert strict.allow_partial is False
    assert permissive.allow_partial is True
