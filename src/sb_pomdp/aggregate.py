"""Aggregate completed comparison shards without copying their large artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import RunArtifacts
from .compare import (
    COMPARISON_FIELDS,
    COMPARISON_SUMMARY_FIELDS,
    _aggregate,
    _eligible,
    _method_config,
)
from .config import ExperimentConfig, load_config
from .fetch_results import _verify_staged_result
from .train import normalize_legacy_summary

CoverageKey = tuple[str, str, str, int]
_CAMPAIGN_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?$")

_IGNORED_CONFIG_SELECTORS = {
    "experiment": frozenset({"tasks", "seeds", "label", "output_dir"}),
    "model": frozenset({"belief_gradient_mode"}),
    "comparison": frozenset({"methods", "belief_gradient_modes"}),
}


def _expected_coverage(config: ExperimentConfig) -> set[CoverageKey]:
    """Return the complete eligible comparison matrix declared by ``config``."""

    return {
        (str(gradient_mode), str(method), str(task), int(seed))
        for gradient_mode in config["comparison"]["belief_gradient_modes"]
        for method in config["comparison"]["methods"]
        for task in config["experiment"]["tasks"]
        if _eligible(config, str(method), str(task))[0]
        for seed in config["experiment"]["seeds"]
    }


def _nonselector_config(config: ExperimentConfig) -> dict[str, Any]:
    """Return config data with only permitted per-shard selectors removed."""

    data = config.to_dict()
    for section, keys in _IGNORED_CONFIG_SELECTORS.items():
        for key in keys:
            del data[section][key]
    return data


def _value_differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Describe exact structural differences without leaking unrelated data."""

    location = path or "config"
    if type(expected) is not type(actual):
        return [
            (f"{location} type: expected {type(expected).__name__}, found {type(actual).__name__}")
        ]
    if isinstance(expected, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        differences = [
            *(f"{location}.{key}: missing" for key in sorted(expected_keys - actual_keys)),
            *(f"{location}.{key}: unexpected" for key in sorted(actual_keys - expected_keys)),
        ]
        for key in sorted(expected_keys & actual_keys):
            child = f"{path}.{key}" if path else str(key)
            differences.extend(_value_differences(expected[key], actual[key], child))
        return differences
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if len(expected) != len(actual):
            return [f"{location} length: expected {len(expected)}, found {len(actual)}"]
        differences = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(
                _value_differences(expected_item, actual_item, f"{location}[{index}]")
            )
        return differences
    return [] if expected == actual else [f"{location}: expected {expected!r}, found {actual!r}"]


def _validate_source_config(
    source_config: ExperimentConfig,
    expected_config: ExperimentConfig,
    root: Path,
) -> None:
    expected = _nonselector_config(expected_config)
    actual = _nonselector_config(source_config)
    differences = _value_differences(expected, actual)
    if differences:
        preview = "; ".join(differences[:5])
        suffix = "; ..." if len(differences) > 5 else ""
        raise RuntimeError(
            f"source config differs from master outside permitted selectors in {root}: "
            f"{preview}{suffix}"
        )


def _manifest_runs(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    expected_config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], set[CoverageKey]]:
    """Validate completed manifest rows against the source root's config."""

    if manifest.get("status") != "completed":
        raise RuntimeError(f"comparison source is not completed: {root}")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise RuntimeError(f"comparison source manifest has no runs: {root}")

    summaries: list[dict[str, Any]] = []
    keys: list[CoverageKey] = []
    for index, raw_summary in enumerate(raw_runs):
        if not isinstance(raw_summary, dict):
            raise TypeError(f"manifest run {index} is not an object in {root}")
        gradient_mode = raw_summary.get("belief_gradient_mode")
        method = raw_summary.get("method")
        task = raw_summary.get("task")
        seed = raw_summary.get("seed")
        if (
            not isinstance(gradient_mode, str)
            or not isinstance(method, str)
            or not isinstance(task, str)
        ):
            raise TypeError(
                f"manifest run {index} has invalid gradient mode/method/task in {root}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"manifest run {index} has invalid seed in {root}")
        if raw_summary.get("status") != "completed":
            raise RuntimeError(
                f"manifest run {(gradient_mode, method, task, seed)} "
                f"is not completed in {root}"
            )
        keys.append((gradient_mode, method, task, seed))
        # A shard produced before a summary alias existed carries the same
        # values under the original keys, so it stays aggregatable next to a
        # shard written by the current trainer.
        summaries.append(normalize_legacy_summary(raw_summary))

    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise RuntimeError(f"duplicate runs within comparison source {root}: {duplicates[:5]}")
    coverage = set(keys)
    try:
        _validate_coverage(coverage, expected_config)
    except RuntimeError as exc:
        raise RuntimeError(
            f"manifest runs do not match the source resolved config in {root}: {exc}"
        ) from exc
    return summaries, coverage


def _validate_run_config(
    summary: Mapping[str, Any],
    *,
    root: Path,
    source_config: ExperimentConfig,
    source_fingerprint: str,
) -> None:
    gradient_mode = str(summary["belief_gradient_mode"])
    method = str(summary["method"])
    task = str(summary["task"])
    seed = int(summary["seed"])
    relative = PurePosixPath(str(summary["result_dir"]))
    expected_relative = PurePosixPath(
        gradient_mode,
        method,
        task,
        f"seed_{seed:03d}",
    )
    if relative != expected_relative:
        raise RuntimeError(
            "run result_dir does not match its gradient mode/method/task/seed for "
            f"{(gradient_mode, method, task, seed)} in {root}: {relative}"
        )
    run_root = root.joinpath(*relative.parts)
    run_config_path = run_root / "resolved_config.json"
    actual = load_config(run_config_path)
    expected = _method_config(
        source_config,
        method,
        task,
        seed,
        gradient_mode=gradient_mode,
    )
    differences = _value_differences(expected.to_dict(), actual.to_dict())
    if differences:
        preview = "; ".join(differences[:5])
        suffix = "; ..." if len(differences) > 5 else ""
        raise RuntimeError(
            f"run config differs from the expected method config for "
            f"{(gradient_mode, method, task, seed)} in {root}: {preview}{suffix}"
        )
    run_metadata = json.loads((run_root / "metadata.json").read_text(encoding="utf-8"))
    if run_metadata.get("source_sha256") != source_fingerprint:
        raise RuntimeError(
            "run source_sha256 differs from its root for "
            f"{(gradient_mode, method, task, seed)} in {root}"
        )


def _validate_coverage(
    seen: set[CoverageKey],
    expected_config: ExperimentConfig,
) -> int:
    expected = _expected_coverage(expected_config)
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing or unexpected:

        def preview(items: list[CoverageKey]) -> str:
            suffix = " ..." if len(items) > 5 else ""
            return f"{items[:5]}{suffix}"

        raise RuntimeError(
            "comparison coverage does not match the supplied config; "
            f"missing={preview(missing)}, unexpected={preview(unexpected)}"
        )
    return len(expected)


def aggregate_runs(
    run_roots: list[Path],
    *,
    expected_config: ExperimentConfig,
    output_dir: Path,
    label: str = "comparison-aggregate",
    campaign_id: str | None = None,
) -> RunArtifacts:
    """Validate shards and write a timestamped lightweight comparison report."""

    if not run_roots:
        raise ValueError("at least one comparison run root is required")
    if not isinstance(expected_config, ExperimentConfig):
        raise TypeError("expected_config must be an ExperimentConfig")
    if campaign_id is not None and not _CAMPAIGN_PATTERN.fullmatch(campaign_id):
        raise ValueError("campaign_id contains unsafe characters")
    summaries: list[dict[str, Any]] = []
    sources: list[str] = []
    source_coverage: list[dict[str, Any]] = []
    seen: set[CoverageKey] = set()
    source_campaigns: list[str | None] = []
    source_fingerprints: list[str] = []
    for supplied_root in run_roots:
        root = supplied_root.expanduser().resolve()
        _verify_staged_result(root)
        source_config = load_config(root / "resolved_config.json")
        _validate_source_config(source_config, expected_config, root)
        source_metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        source_fingerprint = source_metadata.get("source_sha256")
        if (
            not isinstance(source_fingerprint, str)
            or len(source_fingerprint) != 64
            or not all(character in "0123456789abcdef" for character in source_fingerprint)
        ):
            raise RuntimeError(f"comparison source has no valid source_sha256: {root}")
        source_fingerprints.append(source_fingerprint)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        source_campaign = manifest.get("campaign_id")
        if source_campaign is not None and (
            not isinstance(source_campaign, str) or not _CAMPAIGN_PATTERN.fullmatch(source_campaign)
        ):
            raise RuntimeError(f"comparison source has an invalid campaign_id: {root}")
        source_campaigns.append(source_campaign)
        source_summaries, source_keys = _manifest_runs(
            manifest,
            root=root,
            expected_config=source_config,
        )
        for summary in source_summaries:
            _validate_run_config(
                summary,
                root=root,
                source_config=source_config,
                source_fingerprint=source_fingerprint,
            )
            key = (
                str(summary["belief_gradient_mode"]),
                str(summary["method"]),
                str(summary["task"]),
                int(summary["seed"]),
            )
            if key in seen:
                raise RuntimeError(f"duplicate comparison shard: {key}")
            seen.add(key)
            summaries.append(summary)
        sources.append(str(root))
        source_coverage.append(
            {
                "source": str(root),
                "expected_runs": len(_expected_coverage(source_config)),
                "observed_runs": len(source_keys),
                "belief_gradient_modes": list(
                    source_config["comparison"]["belief_gradient_modes"]
                ),
                "complete": True,
            }
        )

    distinct_campaigns = set(source_campaigns)
    if campaign_id is not None:
        if distinct_campaigns != {campaign_id}:
            raise RuntimeError("comparison sources do not all match the requested campaign_id")
        resolved_campaign = campaign_id
    elif len(run_roots) > 1:
        if None in distinct_campaigns or len(distinct_campaigns) != 1:
            raise RuntimeError(
                "multiple comparison sources require one common non-empty campaign_id"
            )
        resolved_campaign = next(iter(distinct_campaigns))
    else:
        resolved_campaign = source_campaigns[0]
    distinct_fingerprints = set(source_fingerprints)
    if len(distinct_fingerprints) != 1:
        raise RuntimeError("comparison sources were produced by different source code")
    source_fingerprint = source_fingerprints[0]

    expected_runs = _validate_coverage(seen, expected_config)
    coverage = {
        "complete": True,
        "expected_runs": expected_runs,
        "observed_runs": len(seen),
        "source_count": len(sources),
        "belief_gradient_modes": list(
            expected_config["comparison"]["belief_gradient_modes"]
        ),
        "sources": source_coverage,
        "campaign_id": resolved_campaign,
        "source_sha256": source_fingerprint,
    }

    report = RunArtifacts.create(output_dir, label)
    rows = _aggregate(summaries)
    for row in rows:
        report.comparison_appender(COMPARISON_FIELDS).append(row)
    report.write_comparison(rows)
    report.write_summary(summaries)
    for summary in summaries:
        report.append_summary(summary, COMPARISON_SUMMARY_FIELDS)
    report.write_resolved_config(expected_config.to_dict())
    report.write_metadata(
        {
            "kind": "comparison_aggregate",
            "created_at": datetime.now(UTC).isoformat(),
            "sources": sources,
            "belief_gradient_modes": list(
                expected_config["comparison"]["belief_gradient_modes"]
            ),
            "unique_gradient_mode_method_task_seed_runs": len(seen),
            "expected_gradient_mode_method_task_seed_runs": expected_runs,
            "coverage_complete": True,
            "coverage": deepcopy(coverage),
            "campaign_id": resolved_campaign,
            "source_sha256": source_fingerprint,
        }
    )
    report.write_manifest(
        {
            "status": "completed",
            "kind": "comparison_aggregate",
            "created_at": datetime.now(UTC).isoformat(),
            "sources": sources,
            "comparison_rows": len(rows),
            "belief_gradient_modes": list(
                expected_config["comparison"]["belief_gradient_modes"]
            ),
            "unique_gradient_mode_method_task_seed_runs": len(seen),
            "expected_gradient_mode_method_task_seed_runs": expected_runs,
            "coverage_complete": True,
            "coverage": deepcopy(coverage),
            "campaign_id": resolved_campaign,
            "source_sha256": source_fingerprint,
        }
    )
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--runs", nargs="+", type=Path)
    sources.add_argument(
        "--campaign-dir",
        type=Path,
        help="directory whose immediate child directories are bulk result roots",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--label", default="comparison-aggregate")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help=(
            "master config used for exact settings and "
            "gradient-mode/method/task/seed coverage"
        ),
    )
    parser.add_argument(
        "--campaign-id",
        help="require every source manifest to belong to this campaign",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="validated config override used to describe debug or pilot coverage",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_roots = args.runs
        if args.campaign_dir is not None:
            campaign_directory = args.campaign_dir.expanduser().resolve()
            if not campaign_directory.is_dir():
                raise NotADirectoryError(f"campaign directory does not exist: {campaign_directory}")
            run_roots = sorted(path for path in campaign_directory.iterdir() if path.is_dir())
        assert run_roots is not None
        report = aggregate_runs(
            run_roots,
            expected_config=load_config(args.config, args.override),
            output_dir=args.output_dir,
            label=args.label,
            campaign_id=args.campaign_id,
        )
    except Exception as exc:  # noqa: BLE001 - concise CLI boundary.
        print(f"Aggregation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(report.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["aggregate_runs", "main"]
