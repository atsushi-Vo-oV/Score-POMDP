"""Safely copy one Genkai result root or a verified campaign to local results."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np

_HOST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REMOTE_PATH_PATTERN = re.compile(r"^/[A-Za-z0-9_./-]+$")
_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?$")
_ROOT_ARTIFACTS = (
    "manifest.json",
    "metadata.json",
    "resolved_config.json",
    "summary.json",
)
_REQUIRED_SEED_FILES = (
    "run.log",
    "metrics.csv",
    "evaluation.csv",
    "arrays.npz",
    "metadata.json",
    "resolved_config.json",
    "checkpoints/latest.pt",
)
# A partial run may not have reached an evaluation update yet, so only its
# evaluation artifact is optional; everything else is committed every segment.
_REQUIRED_PARTIAL_SEED_FILES = tuple(
    name for name in _REQUIRED_SEED_FILES if name != "evaluation.csv"
)


def _component(value: str, name: str) -> str:
    if value in {".", ".."} or not _COMPONENT_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be one safe path component")
    return value


def _verify_staged_result(staging: Path, *, allow_partial: bool = False) -> str:
    """Verify one staged result root and return its accepted manifest status.

    ``allow_partial`` opts in to accepting a shard that committed resumable
    state without reaching its configured update target.  A partial root is
    verified exactly like a completed one, and its partial task/seed runs are
    verified like completed runs except for ``evaluation.csv``, which a shard
    fetched before its first evaluation update has not written yet.
    """

    missing_root = [name for name in _ROOT_ARTIFACTS if not (staging / name).is_file()]
    if missing_root:
        raise RuntimeError(f"copied run is missing root artifacts: {missing_root}")
    manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    status = manifest.get("status")
    accepted = {"completed", "failed", "partial"} if allow_partial else {"completed", "failed"}
    if status not in accepted:
        hint = (
            " (pass --allow-partial to accept a shard that has not reached its "
            "configured update target)"
            if status == "partial"
            else ""
        )
        raise RuntimeError(f"copied run has an unexpected manifest status: {status!r}{hint}")
    trained = status in {"completed", "partial"}
    if trained and not (staging / "summary.csv").is_file():
        raise RuntimeError(f"copied {status} run is missing root artifact: summary.csv")
    if trained and "methods" in manifest:
        missing_comparison = [
            name for name in ("comparison.csv", "comparison.json") if not (staging / name).is_file()
        ]
        if missing_comparison:
            raise RuntimeError(
                f"copied comparison run is missing aggregate artifacts: {missing_comparison}"
            )

    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise TypeError("copied manifest does not contain a run list")
    if trained and not runs:
        raise RuntimeError("copied manifest does not contain any task/seed runs")
    if trained:
        invalid_entries = [index for index, run in enumerate(runs) if not isinstance(run, dict)]
        if invalid_entries:
            raise TypeError(
                "copied trained manifest contains non-object run entries at "
                f"indices {invalid_entries[:5]}"
            )
        run_statuses = [run.get("status") for run in runs]
        allowed_run_statuses = {"completed"} if status == "completed" else {"completed", "partial"}
        unexpected_statuses = [
            (index, run_status)
            for index, run_status in enumerate(run_statuses)
            if run_status not in allowed_run_statuses
        ]
        if unexpected_statuses:
            raise RuntimeError(
                f"copied {status} manifest contains inconsistent task/seed run statuses: "
                f"{unexpected_statuses[:5]}"
            )
        if status == "partial" and "partial" not in run_statuses:
            raise RuntimeError(
                "copied partial manifest contains no partial task/seed run"
            )
    if status == "failed" and not any(staging.rglob("failure.txt")):
        raise RuntimeError("copied failed run does not contain failure.txt diagnostics")
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_status = run.get("status")
        if run_status == "completed":
            required_seed_files = _REQUIRED_SEED_FILES
        elif allow_partial and run_status == "partial":
            required_seed_files = _REQUIRED_PARTIAL_SEED_FILES
        else:
            continue
        relative_text = run.get("result_dir")
        if not isinstance(relative_text, str):
            raise TypeError(f"{run_status} run has no relative result_dir")
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or len(relative.parts) not in {2, 3, 4}
            or any(not _COMPONENT_PATTERN.fullmatch(part) for part in relative.parts)
        ):
            raise RuntimeError(f"{run_status} run has unsafe result_dir: {relative_text!r}")
        seed_directory = staging.joinpath(*relative.parts)
        missing_seed = [
            name for name in required_seed_files if not (seed_directory / name).is_file()
        ]
        if missing_seed:
            raise RuntimeError(f"{relative_text} is missing artifacts: {missing_seed}")

    for archive in staging.rglob("*.npz"):
        with np.load(archive, allow_pickle=False) as arrays:
            for name in arrays.files:
                _ = arrays[name].shape
    return str(status)


def _report_partial_roots(names: Sequence[str]) -> None:
    """Annotate which fetched roots are partial so a report never hides them."""

    if not names:
        return
    print(
        f"accepted {len(names)} partial result root(s): {', '.join(names)}",
        file=sys.stderr,
    )


def fetch_result(
    *,
    host: str,
    remote_results_dir: str,
    run_name: str,
    local_results_dir: Path,
    ssh_config: Path | None,
    allow_partial: bool = False,
) -> Path:
    """Copy without overwriting, then verify the manifest and NumPy archives.

    ``allow_partial`` additionally accepts a run whose manifest status is
    ``partial``; the default stays completed-only.
    """

    if not _HOST_PATTERN.fullmatch(host):
        raise ValueError("host must be a safe OpenSSH alias or hostname")
    safe_run_name = _component(run_name, "run_name")
    if not _REMOTE_PATH_PATTERN.fullmatch(remote_results_dir):
        raise ValueError("remote_results_dir contains unsupported characters")
    remote_base = PurePosixPath(remote_results_dir)
    if not remote_base.is_absolute() or ".." in remote_base.parts:
        raise ValueError("remote_results_dir must be a normalized absolute POSIX path")
    local_results_dir = local_results_dir.expanduser().resolve()
    local_results_dir.mkdir(parents=True, exist_ok=True)
    destination = local_results_dir / safe_run_name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {destination}")
    staging = local_results_dir / f".partial-{uuid4().hex[:12]}"

    command = ["scp"]
    if ssh_config is not None:
        command.extend(("-F", str(ssh_config.expanduser().resolve())))
    remote_source = f"{host}:{remote_base / safe_run_name}"
    command.extend(("-r", remote_source, str(staging)))
    # A sequence is passed directly (never through a shell), after validating
    # the host and run-name components above.
    try:
        subprocess.run(command, check=True)
        status = _verify_staged_result(staging, allow_partial=allow_partial)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    _report_partial_roots([safe_run_name] if status == "partial" else [])
    return destination


def _expected_campaign_indices(
    expected_shards: int | None,
    expected_indices: list[int] | tuple[int, ...] | None,
) -> set[int]:
    if expected_shards is not None and expected_indices is not None:
        raise ValueError("expected_shards and expected_indices are mutually exclusive")
    if expected_indices is not None:
        if not expected_indices:
            raise ValueError("expected_indices must not be empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index <= 0
            for index in expected_indices
        ):
            raise ValueError("expected_indices must contain only positive integers")
        indices = set(expected_indices)
        if len(indices) != len(expected_indices):
            raise ValueError("expected_indices must not contain duplicates")
        return indices
    count = 80 if expected_shards is None else expected_shards
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("expected_shards must be an integer")
    if count <= 0:
        raise ValueError("expected_shards must be positive")
    return set(range(1, count + 1))


def _verify_campaign(
    staging: Path,
    campaign_id: str,
    expected_indices: set[int],
    *,
    allow_partial: bool = False,
) -> list[str]:
    """Verify every campaign root and return the names of the partial ones."""

    roots = sorted(path for path in staging.iterdir() if path.is_dir())
    if len(roots) != len(expected_indices):
        raise RuntimeError(
            f"campaign contains {len(roots)} result roots; expected {len(expected_indices)}"
        )
    indices: set[int] = set()
    partial_roots: list[str] = []
    for root in roots:
        if _verify_staged_result(root, allow_partial=allow_partial) == "partial":
            partial_roots.append(root.name)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("campaign_id") != campaign_id:
            raise RuntimeError(f"result root has a different campaign_id: {root.name}")
        bulk_index = manifest.get("bulk_index")
        if isinstance(bulk_index, bool) or not isinstance(bulk_index, int):
            raise TypeError(f"result root has no integer bulk_index: {root.name}")
        if bulk_index in indices:
            raise RuntimeError(f"campaign contains duplicate bulk index {bulk_index}")
        indices.add(bulk_index)
    if indices != expected_indices:
        missing = sorted(expected_indices - indices)
        unexpected = sorted(indices - expected_indices)
        raise RuntimeError(
            f"campaign bulk indices are incomplete; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    return sorted(partial_roots)


def fetch_campaign(
    *,
    host: str,
    remote_campaigns_dir: str,
    campaign_id: str,
    local_results_dir: Path,
    ssh_config: Path | None,
    expected_shards: int | None = None,
    expected_indices: list[int] | tuple[int, ...] | None = None,
    allow_partial: bool = False,
) -> Path:
    """Copy and validate one complete or explicitly sparse campaign.

    ``allow_partial`` additionally accepts roots whose manifest status is
    ``partial``; the default stays completed-only.
    """

    if not _HOST_PATTERN.fullmatch(host):
        raise ValueError("host must be a safe OpenSSH alias or hostname")
    safe_campaign = _component(campaign_id, "campaign_id")
    if not _REMOTE_PATH_PATTERN.fullmatch(remote_campaigns_dir):
        raise ValueError("remote_campaigns_dir contains unsupported characters")
    remote_base = PurePosixPath(remote_campaigns_dir)
    if not remote_base.is_absolute() or ".." in remote_base.parts:
        raise ValueError("remote_campaigns_dir must be a normalized absolute POSIX path")
    required_indices = _expected_campaign_indices(expected_shards, expected_indices)

    campaigns_dir = local_results_dir.expanduser().resolve() / "campaigns"
    campaigns_dir.mkdir(parents=True, exist_ok=True)
    destination = campaigns_dir / safe_campaign
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing campaign: {destination}")
    staging = campaigns_dir / f".partial-{uuid4().hex[:12]}"
    command = ["scp"]
    if ssh_config is not None:
        command.extend(("-F", str(ssh_config.expanduser().resolve())))
    command.extend(("-r", f"{host}:{remote_base / safe_campaign}", str(staging)))
    try:
        subprocess.run(command, check=True)
        partial_roots = _verify_campaign(
            staging,
            safe_campaign,
            required_indices,
            allow_partial=allow_partial,
        )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    _report_partial_roots(partial_roots)
    return destination


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-results-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-name")
    selection.add_argument("--campaign-id")
    expected = parser.add_mutually_exclusive_group()
    expected.add_argument(
        "--expected-shards",
        type=int,
        help="require the contiguous 1..N shard set (default: 80)",
    )
    expected.add_argument(
        "--expected-shard-indices",
        type=int,
        nargs="+",
        help="require this explicit sparse set of positive shard indices",
    )
    parser.add_argument("--local-results-dir", type=Path, default=Path("results"))
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "also accept results whose manifest status is 'partial' "
            "(default: completed-only)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.campaign_id is None:
            destination = fetch_result(
                host=args.host,
                remote_results_dir=args.remote_results_dir,
                run_name=args.run_name,
                local_results_dir=args.local_results_dir,
                ssh_config=args.ssh_config,
                allow_partial=args.allow_partial,
            )
        else:
            destination = fetch_campaign(
                host=args.host,
                remote_campaigns_dir=args.remote_results_dir,
                campaign_id=args.campaign_id,
                local_results_dir=args.local_results_dir,
                ssh_config=args.ssh_config,
                expected_shards=args.expected_shards,
                expected_indices=args.expected_shard_indices,
                allow_partial=args.allow_partial,
            )
    except Exception as exc:  # noqa: BLE001 - present a concise CLI failure.
        print(f"Result transfer failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
