"""Run the proposed method and matched POMDP baselines into one result tree."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import RunArtifacts, SeedArtifacts
from .config import (
    BELIEF_GRADIENT_MODES,
    COMPARISON_METHODS,
    ConfigError,
    ExperimentConfig,
    load_config,
    normalize_legacy_resolved_config,
)
from .envs import make_env
from .run_experiments import validate_all_environments
from .train import (
    SUMMARY_FIELDS,
    normalize_legacy_summary,
    resolve_device,
    runtime_metadata,
    source_sha256,
    train_single,
)
from .train_baselines import train_baseline_single

SCORE_METHODS = frozenset(
    {"score_transformer", "score_deepsets", "score_gaussian", "score_alpha"}
)
BASELINE_METHODS = frozenset({"observation_mlp", "gru", "rnn", "oracle_state", "particle_filter"})
COMPARISON_SUMMARY_FIELDS = (
    "belief_gradient_mode",
    "method",
    *SUMMARY_FIELDS,
    "deterministic_mode",
    "deterministic_return_mean",
    "stochastic_return_mean",
)
# Paired-difference sign convention, fixed by :func:`_aggregate` and repeated in
# the README: every ``*_paired_difference_*`` column is
# ``score_transformer minus this row's method`` at the same belief gradient
# mode, task, and seed.  A positive value therefore means the proposed method
# scored *above* the row's method.  The ``score_transformer`` row itself is
# paired against itself and is exactly zero.
COMPARISON_FIELDS = (
    "belief_gradient_mode",
    "method",
    "task",
    "seeds_completed",
    "deterministic_mode",
    "deterministic_return_mean",
    "deterministic_return_std",
    "deterministic_proposed_return_mean",
    "deterministic_paired_difference_mean",
    "deterministic_paired_difference_std",
    "deterministic_paired_difference_n_seeds",
    "stochastic_return_mean",
    "stochastic_return_std",
    "stochastic_proposed_return_mean",
    "stochastic_paired_difference_mean",
    "stochastic_paired_difference_std",
    "stochastic_paired_difference_n_seeds",
    "environment_steps_per_seed",
    "wall_time_seconds_mean",
    "parameter_count_mean",
)
_CAMPAIGN_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?$")
_STEP_CHECKPOINT_PATTERN = re.compile(r"^step_([0-9]{9})\.pt$")
# Root-metadata fields that a later segment reads back to decide whether it may
# resume.  Every rewrite of an existing root must carry them over, otherwise a
# failure before the runtime metadata is merged in erases the fingerprint and the
# source/runtime-identity guard rejects every later segment.
_RUNTIME_IDENTITY_FIELDS = ("python", "torch", "numpy", "device", "cuda", "gpu")
_PERSISTED_IDENTITY_FIELDS = ("source_sha256", *_RUNTIME_IDENTITY_FIELDS)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _selection(
    config: ExperimentConfig,
    requested_methods: list[str] | None,
    requested_tasks: list[str] | None,
    requested_seeds: list[int] | None,
) -> tuple[list[str], list[str], list[int]]:
    configured_methods = list(config["comparison"]["methods"])
    configured_tasks = list(config["experiment"]["tasks"])
    configured_seeds = list(config["experiment"]["seeds"])
    methods = configured_methods if requested_methods is None else requested_methods
    tasks = configured_tasks if requested_tasks is None else requested_tasks
    seeds = configured_seeds if requested_seeds is None else requested_seeds

    unknown_methods = sorted(set(methods) - COMPARISON_METHODS)
    disabled_methods = sorted(set(methods) - set(configured_methods))
    unknown_tasks = sorted(set(tasks) - set(configured_tasks))
    # Seeds are checked against the configured list exactly like methods and
    # tasks: a shard executed at a seed the config never declares would silently
    # fall outside the campaign's coverage matrix and be rejected only much
    # later, by aggregate.py.  Use an explicit ``experiment.seeds`` override to
    # run a validation seed that the master config does not contain.
    unknown_seeds = sorted(set(seeds) - set(configured_seeds))
    if unknown_methods:
        raise ConfigError(f"unknown comparison methods: {unknown_methods}")
    if disabled_methods:
        raise ConfigError(f"requested methods are not enabled by the config: {disabled_methods}")
    if unknown_tasks:
        raise ConfigError(f"requested tasks are not enabled by the config: {unknown_tasks}")
    if unknown_seeds:
        raise ConfigError(
            f"requested seeds are not enabled by the config: {unknown_seeds}; "
            f"configured seeds: {configured_seeds}"
        )
    if not methods or not tasks or not seeds:
        raise ConfigError("at least one method, task, and seed must be selected")
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise ConfigError("seeds must be in [0, 2**32 - 1]")
    return (
        list(dict.fromkeys(methods)),
        list(dict.fromkeys(tasks)),
        list(dict.fromkeys(seeds)),
    )


def _gradient_mode_selection(
    config: ExperimentConfig,
    requested_gradient_modes: list[str] | None,
) -> list[str]:
    configured_modes = list(config["comparison"]["belief_gradient_modes"])
    modes = configured_modes if requested_gradient_modes is None else requested_gradient_modes
    unknown_modes = sorted(set(modes) - BELIEF_GRADIENT_MODES)
    disabled_modes = sorted(set(modes) - set(configured_modes))
    if unknown_modes:
        raise ConfigError(f"unknown belief gradient modes: {unknown_modes}")
    if disabled_modes:
        raise ConfigError(
            f"requested belief gradient modes are not enabled by the config: {disabled_modes}"
        )
    if not modes:
        raise ConfigError("at least one belief gradient mode must be selected")
    return list(dict.fromkeys(modes))


def _eligible(config: ExperimentConfig, method: str, task: str) -> tuple[bool, str]:
    environment = make_env(task, config.environment_for_task(task), seed=0)
    if method == "score_gaussian" and environment.action_spec.is_discrete:
        return False, "score_gaussian only differs from the proposal on continuous actions"
    return True, ""


def _execution_config(
    config: ExperimentConfig,
    *,
    methods: list[str],
    tasks: list[str],
    seeds: list[int],
    gradient_modes: list[str],
    max_updates: int | None,
    label: str | None,
) -> ExperimentConfig:
    data = config.to_dict()
    data["comparison"]["methods"] = methods
    data["comparison"]["belief_gradient_modes"] = gradient_modes
    data["model"]["belief_gradient_mode"] = gradient_modes[0]
    data["experiment"]["tasks"] = tasks
    data["experiment"]["seeds"] = seeds
    data["experiment"]["label"] = label or data["experiment"]["label"]
    if max_updates is not None:
        rollout_batch = data["ppo"]["num_envs"] * data["ppo"]["rollout_steps"]
        data["ppo"]["total_steps"] = min(
            data["ppo"]["total_steps"],
            max_updates * rollout_batch,
        )
    return ExperimentConfig(data, source=config.source)


def _method_config(
    config: ExperimentConfig,
    method: str,
    task: str,
    seed: int,
    gradient_mode: str | None = None,
) -> ExperimentConfig:
    selected_gradient_mode = gradient_mode or str(config["model"]["belief_gradient_mode"])
    data = config.to_dict()
    data["comparison"]["methods"] = [method]
    data["comparison"]["belief_gradient_modes"] = [selected_gradient_mode]
    data["model"]["belief_gradient_mode"] = selected_gradient_mode
    data["experiment"]["tasks"] = [task]
    data["experiment"]["seeds"] = [seed]
    if method == "score_transformer":
        data["model"]["encoder_kind"] = "transformer"
        data["model"]["continuous_policy_kind"] = "diffusion"
    elif method == "score_deepsets":
        data["model"]["encoder_kind"] = "deep_sets"
        data["model"]["continuous_policy_kind"] = "diffusion"
    elif method == "score_alpha":
        data["model"]["encoder_kind"] = "alpha_pool"
        data["model"]["policy_head_kind"] = "alpha_lse"
        data["model"]["continuous_policy_kind"] = "diffusion"
        alpha_ff = data["comparison"].get("alpha_feedforward_dim")
        if alpha_ff is not None:
            # Widen the alpha trunk independently of the transformer ff width
            # (parameter matching without touching the other score methods).
            data["model"]["transformer_ff_dim"] = int(alpha_ff)
    elif method in ("gru", "rnn", "particle_filter"):
        data["model"]["encoder_kind"] = "transformer"
        data["model"]["continuous_policy_kind"] = "diffusion"
    elif method == "score_gaussian" or method in BASELINE_METHODS:
        data["model"]["encoder_kind"] = "transformer"
        data["model"]["continuous_policy_kind"] = "gaussian"
    else:  # pragma: no cover - guarded by strict selection and config validation.
        raise ConfigError(f"unsupported comparison method: {method}")
    return ExperimentConfig(data, source=config.source)


def _read_final_evaluations(evaluation_path: Path) -> dict[str, str | float]:
    """Read the final deterministic-readout and stochastic evaluations."""

    with evaluation_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"comparison evaluation is empty: {evaluation_path}")
    final_update = max(int(row["update"]) for row in rows)
    final_rows = [row for row in rows if int(row["update"]) == final_update]
    stochastic = [row for row in final_rows if row["mode"] == "stochastic"]
    deterministic = [row for row in final_rows if row["mode"] != "stochastic"]
    if len(stochastic) != 1 or len(deterministic) != 1:
        raise RuntimeError(
            "the final comparison evaluation must contain exactly one stochastic and "
            "one deterministic-readout row"
        )
    deterministic_return = float(deterministic[0]["mean_return"])
    stochastic_return = float(stochastic[0]["mean_return"])
    if not math.isfinite(deterministic_return) or not math.isfinite(stochastic_return):
        raise RuntimeError("comparison evaluation returns must be finite")
    return {
        "deterministic_mode": deterministic[0]["mode"],
        "deterministic_return_mean": deterministic_return,
        "stochastic_return_mean": stochastic_return,
    }


def _optional_final_evaluations(evaluation_path: Path) -> dict[str, str | float | None]:
    """Return the latest evaluation, or empty fields for an intermediate segment."""

    if evaluation_path.is_file():
        return _read_final_evaluations(evaluation_path)
    return {
        "deterministic_mode": "",
        "deterministic_return_mean": None,
        "stochastic_return_mean": None,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read resumable run metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"resumable run metadata must be a JSON object: {path}")
    return payload


def _read_json_array(path: Path) -> list[Any]:
    """Read one resumable root artifact that must be a JSON array."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read resumable run artifact {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ConfigError(f"resumable run artifact must be a JSON array: {path}")
    return payload


def _configured_updates(config: ExperimentConfig) -> int:
    """Return the update target one shard must reach to count as completed.

    This mirrors the derivation in :func:`sb_pomdp.train.train_single` so the
    comparison driver and the trainer agree on when no work remains.
    """

    ppo_config = config["ppo"]
    rollout_batch_size = int(ppo_config["num_envs"]) * int(ppo_config["rollout_steps"])
    return math.ceil(int(ppo_config["total_steps"]) / rollout_batch_size)


def _committed_checkpoint_update(checkpoint_path: Path) -> int | None:
    """Return the update stored in a committed checkpoint, or ``None`` when unknown."""

    if not checkpoint_path.is_file():
        return None
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - an unreadable checkpoint is simply not evidence.
        return None
    if not isinstance(payload, Mapping):
        return None
    update = payload.get("update")
    if isinstance(update, (bool, np.bool_)) or not isinstance(update, (int, np.integer)):
        return None
    return int(update)


def _resume_checkpoint_candidate(artifacts: SeedArtifacts) -> Path | None:
    """Return ``latest.pt`` or the newest readable committed fallback."""

    latest = artifacts.checkpoint_path(tag="latest", create_parent=False)
    if latest.is_file():
        return latest
    candidates: list[tuple[int, Path]] = []
    if artifacts.checkpoints_dir.is_dir():
        for path in artifacts.checkpoints_dir.iterdir():
            match = _STEP_CHECKPOINT_PATTERN.fullmatch(path.name)
            if (match is not None or path.name == "best.pt") and path.is_file():
                update = _committed_checkpoint_update(path)
                if update is not None:
                    candidates.append((update, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _run_key(entry: Mapping[str, Any]) -> tuple[str, str, str, int]:
    """Return the gradient-mode/method/task/seed identity of one planned run."""

    return (
        str(entry["belief_gradient_mode"]),
        str(entry["method"]),
        str(entry["task"]),
        int(entry["seed"]),
    )


def _completed_shard_runs(
    run: RunArtifacts,
    *,
    planned_runs: list[dict[str, Any]],
    configured_updates: int,
) -> list[dict[str, Any]] | None:
    """Return the persisted summaries when an existing root already holds every run.

    ``None`` means "no proof that the shard is finished" and keeps the caller on
    its normal execution path.  Evidence that is missing, partial, or unreadable
    therefore never skips outstanding work, and the check itself writes nothing.
    """

    if not planned_runs or not run.manifest_path.is_file():
        return None
    try:
        manifest = _read_json_object(run.manifest_path)
    except ConfigError:
        return None
    if manifest.get("status") != "completed":
        return None
    recorded = manifest.get("runs")
    if not isinstance(recorded, list) or len(recorded) != len(planned_runs):
        return None
    completed: list[dict[str, Any]] = []
    for entry in recorded:
        if not isinstance(entry, Mapping) or entry.get("status") != "completed":
            return None
        updates = entry.get("updates")
        if (
            isinstance(updates, bool)
            or not isinstance(updates, int)
            or updates < configured_updates
        ):
            return None
        # A manifest written before a summary alias existed is still proof of a
        # finished shard, so the rows are returned in the current schema.
        completed.append(normalize_legacy_summary(entry))
    try:
        recorded_keys = {_run_key(entry) for entry in completed}
    except (KeyError, TypeError, ValueError):
        return None
    if recorded_keys != {_run_key(entry) for entry in planned_runs}:
        return None
    for planned in planned_runs:
        gradient_mode, method, task, seed = _run_key(planned)
        artifacts = SeedArtifacts(
            path=run.root.joinpath(gradient_mode, method, task, f"seed_{seed:03d}"),
            task=task,
            seed=seed,
            method=method,
            gradient_mode=gradient_mode,
        )
        committed_update = _committed_checkpoint_update(
            artifacts.checkpoint_path(tag="latest", create_parent=False)
        )
        if committed_update is None or committed_update < configured_updates:
            return None

    # ``manifest.status=completed`` is the root's final commit marker.  Older
    # writers set it before summary.json and metadata.json, however, so a kill
    # could leave a superficially completed but internally inconsistent root.
    # Such a root must take the idempotent final-checkpoint recovery path below
    # instead of being frozen forever by the byte-preserving no-op guard.
    required_final_artifacts = (
        run.summary_path,
        run.summary_json_path,
        run.metadata_path,
        run.comparison_path,
        run.comparison_json_path,
    )
    if any(not path.is_file() for path in required_final_artifacts):
        return None
    try:
        metadata = _read_json_object(run.metadata_path)
        raw_summary = _read_json_array(run.summary_json_path)
        persisted_summary = [
            normalize_legacy_summary(entry)
            for entry in raw_summary
            if isinstance(entry, Mapping)
        ]
    except (ConfigError, TypeError):
        return None
    if (
        metadata.get("status") != "completed"
        or len(persisted_summary) != len(raw_summary)
        or persisted_summary != completed
    ):
        return None
    return completed


def _restartable_uncommitted_seed_roots(
    run: RunArtifacts,
    *,
    planned_runs: list[dict[str, Any]],
) -> list[Path] | None:
    """Return uncommitted seed roots that may be quarantined before a restart.

    A process can die either during setup or during its first expensive update,
    before ``latest.pt`` is committed.  There is no resumable state in that
    case, but logs/CSV fragments may already exist.  We restart only when the
    manifest still describes the exact planned runs and *no* committed ``.pt``
    file exists.  Existing seed roots are returned so the caller can move them
    aside after source/runtime identity checks; ``None`` means the evidence is
    ambiguous and the normal strict checkpoint path must be used.
    """

    if not planned_runs or not run.manifest_path.is_file():
        return None
    try:
        manifest = _read_json_object(run.manifest_path)
    except ConfigError:
        return None
    if manifest.get("status") not in {"initializing", "running", "failed"}:
        return None
    if manifest.get("eligible_runs") != planned_runs:
        return None
    recorded = manifest.get("runs")
    if not isinstance(recorded, list):
        return None
    if recorded:
        if any(
            not isinstance(entry, Mapping) or entry.get("status") != "failed"
            for entry in recorded
        ):
            return None
        try:
            if {_run_key(entry) for entry in recorded} != {
                _run_key(entry) for entry in planned_runs
            }:
                return None
        except (KeyError, TypeError, ValueError):
            return None

    seed_roots: list[Path] = []
    for planned in planned_runs:
        gradient_mode, method, task, seed = _run_key(planned)
        seed_root = run.root.joinpath(
            gradient_mode,
            method,
            task,
            f"seed_{seed:03d}",
        )
        if not seed_root.exists():
            continue
        if not seed_root.is_dir():
            return None
        # ``save_checkpoint`` publishes only a fully-fsynced ``.pt`` via atomic
        # replace.  Temporary ``*.pt.tmp`` files are not commits and are kept in
        # the quarantined attempt for diagnosis.
        if any(path.is_file() for path in seed_root.rglob("*.pt")):
            return None
        seed_roots.append(seed_root)
    return seed_roots


def _sample_std(values: Sequence[float]) -> float | None:
    """Return the ``ddof=1`` standard deviation, or ``None`` below two values.

    A single completed seed carries no dispersion information, so emitting
    ``0.0`` would claim a precision the row does not have.  ``None`` becomes an
    empty CSV cell and a JSON ``null``, and is read together with the matching
    ``seeds_completed`` / ``*_n_seeds`` count.
    """

    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _aggregate(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate completed run summaries into one row per mode/method/task.

    Every ``*_paired_difference_*`` column follows one fixed sign convention:
    the difference is ``score_transformer minus this row's method`` evaluated at
    the same belief gradient mode, task, and seed, so a positive value means the
    proposed method scored above the row's method.  Only seeds for which the
    ``score_transformer`` run of the same mode and task also completed enter the
    paired statistics; ``*_paired_difference_n_seeds`` reports how many did, and
    ``*_paired_difference_std`` is their ``ddof=1`` standard deviation.
    """

    completed = [summary for summary in summaries if summary["status"] == "completed"]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in completed:
        grouped[
            (
                str(summary["belief_gradient_mode"]),
                str(summary["method"]),
                str(summary["task"]),
            )
        ].append(summary)

    proposed_deterministic = {
        (
            str(summary["belief_gradient_mode"]),
            str(summary["task"]),
            int(summary["seed"]),
        ): float(summary["deterministic_return_mean"])
        for summary in completed
        if summary["method"] == "score_transformer"
    }
    proposed_stochastic = {
        (
            str(summary["belief_gradient_mode"]),
            str(summary["task"]),
            int(summary["seed"]),
        ): float(summary["stochastic_return_mean"])
        for summary in completed
        if summary["method"] == "score_transformer"
    }
    rows: list[dict[str, Any]] = []
    for (gradient_mode, method, task), group in grouped.items():
        deterministic_modes = {str(item["deterministic_mode"]) for item in group}
        if len(deterministic_modes) != 1:
            raise RuntimeError(
                f"inconsistent deterministic evaluation modes for {method}/{task}: "
                f"{sorted(deterministic_modes)}"
            )
        deterministic_returns = np.asarray(
            [float(item["deterministic_return_mean"]) for item in group],
            dtype=np.float64,
        )
        stochastic_returns = np.asarray(
            [float(item["stochastic_return_mean"]) for item in group],
            dtype=np.float64,
        )
        deterministic_paired = [
            proposed_deterministic[(gradient_mode, task, int(item["seed"]))]
            - float(item["deterministic_return_mean"])
            for item in group
            if (gradient_mode, task, int(item["seed"])) in proposed_deterministic
        ]
        stochastic_paired = [
            proposed_stochastic[(gradient_mode, task, int(item["seed"]))]
            - float(item["stochastic_return_mean"])
            for item in group
            if (gradient_mode, task, int(item["seed"])) in proposed_stochastic
        ]
        deterministic_proposed_values = [
            proposed_deterministic[(gradient_mode, task, int(item["seed"]))]
            for item in group
            if (gradient_mode, task, int(item["seed"])) in proposed_deterministic
        ]
        stochastic_proposed_values = [
            proposed_stochastic[(gradient_mode, task, int(item["seed"]))]
            for item in group
            if (gradient_mode, task, int(item["seed"])) in proposed_stochastic
        ]
        rows.append(
            {
                "belief_gradient_mode": gradient_mode,
                "method": method,
                "task": task,
                "seeds_completed": len(group),
                "deterministic_mode": deterministic_modes.pop(),
                "deterministic_return_mean": float(deterministic_returns.mean()),
                "deterministic_return_std": _sample_std(deterministic_returns.tolist()),
                "deterministic_proposed_return_mean": (
                    float(np.mean(deterministic_proposed_values))
                    if deterministic_proposed_values
                    else None
                ),
                # Sign convention: score_transformer minus row method.
                "deterministic_paired_difference_mean": (
                    float(np.mean(deterministic_paired)) if deterministic_paired else None
                ),
                "deterministic_paired_difference_std": _sample_std(deterministic_paired),
                "deterministic_paired_difference_n_seeds": len(deterministic_paired),
                "stochastic_return_mean": float(stochastic_returns.mean()),
                "stochastic_return_std": _sample_std(stochastic_returns.tolist()),
                "stochastic_proposed_return_mean": (
                    float(np.mean(stochastic_proposed_values))
                    if stochastic_proposed_values
                    else None
                ),
                # Sign convention: score_transformer minus row method.
                "stochastic_paired_difference_mean": (
                    float(np.mean(stochastic_paired)) if stochastic_paired else None
                ),
                "stochastic_paired_difference_std": _sample_std(stochastic_paired),
                "stochastic_paired_difference_n_seeds": len(stochastic_paired),
                "environment_steps_per_seed": int(group[0]["global_steps"]),
                "wall_time_seconds_mean": float(
                    np.mean([float(item["wall_time_seconds"]) for item in group])
                ),
                "parameter_count_mean": float(
                    np.mean([float(item["parameter_count"]) for item in group])
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["belief_gradient_mode"]),
            str(row["task"]),
            str(row["method"]),
        ),
    )


def run_comparisons(
    config: ExperimentConfig,
    *,
    methods: list[str],
    tasks: list[str],
    seeds: list[int],
    max_updates: int | None,
    label: str | None,
    verbose: bool,
    gradient_modes: list[str] | None = None,
    campaign_id: str | None = None,
    bulk_index: int | None = None,
    segment_updates: int | None = None,
    run_dir: Path | None = None,
    resume: bool = False,
) -> tuple[RunArtifacts, list[dict[str, Any]], list[dict[str, str]]]:
    if campaign_id is not None and not _CAMPAIGN_PATTERN.fullmatch(campaign_id):
        raise ConfigError(
            "campaign_id must start with an ASCII letter or digit and contain only "
            "letters, digits, '.', '_', or '-'"
        )
    selected_gradient_modes = _gradient_mode_selection(config, gradient_modes)
    if segment_updates is not None:
        if max_updates is not None:
            raise ConfigError("segment_updates and max_updates are mutually exclusive")
        if isinstance(segment_updates, bool) or segment_updates <= 0:
            raise ConfigError("segment_updates must be a positive integer")
        if run_dir is None or not resume:
            raise ConfigError("segmented execution requires one resumable run_dir")
    if resume and run_dir is None:
        raise ConfigError("resume requires run_dir")
    if run_dir is not None and (
        len(selected_gradient_modes) != 1
        or len(methods) != 1
        or len(tasks) != 1
        or len(seeds) != 1
    ):
        raise ConfigError("an explicit run_dir requires exactly one mode/method/task/seed")
    execution_config = _execution_config(
        config,
        methods=methods,
        tasks=tasks,
        seeds=seeds,
        gradient_modes=selected_gradient_modes,
        max_updates=max_updates,
        label=label,
    )
    experiment = execution_config["experiment"]
    resumed_run = False
    if run_dir is None:
        run = RunArtifacts.create(experiment["output_dir"], experiment["label"])
    else:
        output_root = Path(experiment["output_dir"]).expanduser().resolve()
        explicit_root = run_dir.expanduser().resolve()
        if not explicit_root.is_relative_to(output_root):
            raise ConfigError(f"run_dir must be inside experiment.output_dir: {output_root}")
        if explicit_root.exists():
            if not resume:
                raise ConfigError(f"run_dir already exists; use resume explicitly: {explicit_root}")
            run = RunArtifacts.open(explicit_root, experiment["label"])
            # A resolved config written before a schema key existed must resume
            # as an equal config when that key is left at its legacy default.
            existing_config = normalize_legacy_resolved_config(
                _read_json_object(run.resolved_config_path)
            )
            if existing_config != execution_config.to_dict():
                raise ConfigError("resumable run config differs from the original resolved config")
            resumed_run = True
        else:
            run = RunArtifacts.create_at(explicit_root, experiment["label"])
    segment_started_at = _utc_now()
    previous_metadata = (
        _read_json_object(run.metadata_path)
        if resumed_run and run.metadata_path.is_file()
        else {}
    )
    if resumed_run:
        expected_identity = {
            "belief_gradient_modes": selected_gradient_modes,
            "methods": methods,
            "tasks": tasks,
            "seeds": seeds,
            "campaign_id": campaign_id,
            "bulk_index": bulk_index,
        }
        for key, expected_value in expected_identity.items():
            if previous_metadata.get(key) != expected_value:
                raise ConfigError(
                    f"resumable run identity mismatch for {key}: "
                    f"existing={previous_metadata.get(key)!r}, requested={expected_value!r}"
                )
    previous_segments = previous_metadata.get("segments", [])
    if not isinstance(previous_segments, list):
        raise ConfigError("resumable run metadata segments must be a list")
    started_at = str(previous_metadata.get("started_at", segment_started_at))

    eligible_runs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for gradient_mode in selected_gradient_modes:
        for method in methods:
            for task in tasks:
                eligible, reason = _eligible(execution_config, method, task)
                if eligible:
                    eligible_runs.extend(
                        {
                            "belief_gradient_mode": gradient_mode,
                            "method": method,
                            "task": task,
                            "seed": seed,
                        }
                        for seed in seeds
                    )
                else:
                    skipped.append(
                        {
                            "belief_gradient_mode": gradient_mode,
                            "method": method,
                            "task": task,
                            "reason": reason,
                        }
                    )

    configured_updates = _configured_updates(execution_config)
    completed_runs = (
        _completed_shard_runs(
            run,
            planned_runs=eligible_runs,
            configured_updates=configured_updates,
        )
        if resumed_run
        else None
    )
    if completed_runs is not None:
        # Every destructive write stays below this guard: a re-submitted segment
        # of a finished shard must leave the artifacts byte-identical and still
        # exit successfully so a chained job does not cancel its successors.
        print(
            f"Already completed at {configured_updates} updates; "
            f"artifacts left untouched: {run.root}",
            file=sys.stderr,
        )
        return run, completed_runs, skipped

    restartable_seed_roots = (
        _restartable_uncommitted_seed_roots(run, planned_runs=eligible_runs)
        if resumed_run
        else None
    )
    restart_uncommitted_setup = restartable_seed_roots is not None
    run.write_resolved_config(execution_config.to_dict())
    root_metadata: dict[str, Any] = {
        "status": "initializing",
        "started_at": started_at,
        "requested_device": experiment["device"],
        "belief_gradient_modes": selected_gradient_modes,
        "methods": methods,
        "tasks": tasks,
        "seeds": seeds,
        "max_updates": max_updates,
        "label": experiment["label"],
        "campaign_id": campaign_id,
        "bulk_index": bulk_index,
        "segment_updates": segment_updates,
        "resumed": resumed_run,
        "segment_started_at": segment_started_at,
        "segments": previous_segments,
    }
    root_metadata.update(
        {
            field: previous_metadata[field]
            for field in _PERSISTED_IDENTITY_FIELDS
            if field in previous_metadata
        }
    )
    run.write_metadata(root_metadata)
    if not resumed_run:
        run.write_summary([])

    manifest: dict[str, Any] = {
        "status": "initializing",
        "started_at": started_at,
        "belief_gradient_modes": selected_gradient_modes,
        "methods": methods,
        "tasks": tasks,
        "seeds": seeds,
        "max_updates": max_updates,
        "segment_updates": segment_updates,
        "campaign_id": campaign_id,
        "bulk_index": bulk_index,
        "eligible_runs": eligible_runs,
        "skipped": skipped,
        "runs": [],
    }
    run.write_manifest(manifest)
    summaries: list[dict[str, Any]] = []

    try:
        # Fingerprinting is deliberately separate from device/runtime discovery.
        # Persist it first so a later CUDA setup failure cannot erase the resume
        # identity.  A fingerprinting failure itself is inside this diagnostic
        # boundary and therefore leaves normal failed-root artifacts.
        current_source = source_sha256()
        previous_source = previous_metadata.get("source_sha256")
        if resumed_run and previous_source != current_source:
            missing_identity_on_restartable_setup = (
                previous_source is None and restart_uncommitted_setup
            )
            if not missing_identity_on_restartable_setup:
                raise ConfigError(
                    "resumable run source code differs from the segment that created it"
                )
        root_metadata["source_sha256"] = current_source
        run.write_metadata(root_metadata)
        device = resolve_device(experiment["device"])
        current_runtime = runtime_metadata(device, source_fingerprint=current_source)
        if resumed_run:
            runtime_differences = {
                field: (previous_metadata.get(field), current_runtime.get(field))
                for field in _RUNTIME_IDENTITY_FIELDS
                if field in previous_metadata
                and previous_metadata.get(field) != current_runtime.get(field)
            }
            if runtime_differences:
                details = "; ".join(
                    f"{field}: existing={existing!r}, current={current!r}"
                    for field, (existing, current) in runtime_differences.items()
                )
                if os.environ.get("SB_POMDP_ALLOW_RUNTIME_CHANGE", "") != "1":
                    raise ConfigError(
                        "resumable run runtime identity differs from the segment that "
                        f"created it ({details})"
                    )
                # Explicitly allowed (e.g. a MIG slice ran out of memory and the
                # segment continues on a full GPU): record the change and go on.
                history = list(root_metadata.get("runtime_changes", []))
                history.append({"at": _utc_now(), "differences": details})
                root_metadata["runtime_changes"] = history
                print(f"Continuing despite a runtime identity change: {details}", file=sys.stderr)
        root_metadata.update(current_runtime)
        if restartable_seed_roots:
            attempt_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            quarantine_root = run.root / ".uncommitted-attempts" / attempt_id
            quarantined: list[str] = []
            for seed_root in restartable_seed_roots:
                relative = seed_root.relative_to(run.root)
                destination = quarantine_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                seed_root.replace(destination)
                quarantined.append(destination.relative_to(run.root).as_posix())
            root_metadata["quarantined_uncommitted_attempts"] = quarantined
            print(
                "Restarting from update 0 after quarantining uncommitted seed "
                f"artifacts: {', '.join(quarantined)}",
                file=sys.stderr,
            )
    except Exception as exc:
        finished_at = _utc_now()
        error = f"{type(exc).__name__}: {exc}"
        root_metadata.update({"status": "failed", "finished_at": finished_at, "error": error})
        manifest.update({"status": "failed", "finished_at": finished_at, "error": error})
        run.write_metadata(root_metadata)
        run.write_manifest(manifest)
        (run.root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Failure artifacts: {run.root}", file=sys.stderr)
        raise
    root_metadata["status"] = "running"
    run.write_metadata(root_metadata)
    manifest["status"] = "running"
    run.write_manifest(manifest)

    for planned in eligible_runs:
        gradient_mode = str(planned["belief_gradient_mode"])
        method = str(planned["method"])
        task = str(planned["task"])
        seed = int(planned["seed"])
        relative_result = f"{gradient_mode}/{method}/{task}/seed_{seed:03d}"
        artifacts = run.for_gradient_mode_method_seed(
            gradient_mode,
            method,
            task,
            seed,
            exist_ok=resumed_run,
        )
        method_config = _method_config(
            execution_config,
            method,
            task,
            seed,
            gradient_mode=gradient_mode,
        )
        try:
            # Both trainers are resumed through the same contract: the committed
            # checkpoint of the shard's own artifact tree, guarded by the same
            # config, source, and identity checks above.  A checkpoint-free retry
            # is allowed only after uncommitted artifacts were quarantined.
            resume_checkpoint = (
                _resume_checkpoint_candidate(artifacts)
                if resumed_run and not restart_uncommitted_setup
                else None
            )
            if resumed_run and not restart_uncommitted_setup and resume_checkpoint is None:
                raise ConfigError(
                    "resumable run is missing latest.pt and every readable step/best "
                    "checkpoint: "
                    f"{artifacts.checkpoints_dir}"
                )
            if (
                resume_checkpoint is not None
                and resume_checkpoint.name != "latest.pt"
            ):
                print(
                    "latest.pt is missing; resuming from newest committed fallback "
                    f"checkpoint: {resume_checkpoint}",
                    file=sys.stderr,
                )
            if method in SCORE_METHODS:
                summary = train_single(
                    method_config,
                    task=task,
                    seed=seed,
                    artifacts=artifacts,
                    max_updates=max_updates,
                    segment_updates=segment_updates,
                    resume_checkpoint=resume_checkpoint,
                    verbose=verbose,
                    method=method,
                    result_dir=relative_result,
                )
            else:
                summary = train_baseline_single(
                    method_config,
                    method=method,
                    task=task,
                    seed=seed,
                    artifacts=artifacts,
                    max_updates=max_updates,
                    segment_updates=segment_updates,
                    resume_checkpoint=resume_checkpoint,
                    verbose=verbose,
                    result_dir=relative_result,
                )
            summary = {
                **summary,
                "belief_gradient_mode": gradient_mode,
                "method": method,
                "error": "",
                **_optional_final_evaluations(artifacts.path / "evaluation.csv"),
            }
        except Exception as exc:
            summary = {
                "belief_gradient_mode": gradient_mode,
                "method": method,
                "task": task,
                "seed": seed,
                "status": "failed",
                "updates": 0,
                "global_steps": 0,
                "episodes_completed": 0,
                "training_return_mean": None,
                "evaluation_return_mean": None,
                "deterministic_readout_return_mean": None,
                "parameter_count": 0,
                "wall_time_seconds": 0.0,
                "result_dir": relative_result,
                "error": f"{type(exc).__name__}: {exc}",
                "deterministic_mode": "",
                "deterministic_return_mean": None,
                "stochastic_return_mean": None,
            }
            (artifacts.path / "failure.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            summaries.append(summary)
            run.write_summary_csv(summaries, COMPARISON_SUMMARY_FIELDS)
            manifest["runs"] = summaries
            manifest["status"] = "failed"
            manifest["finished_at"] = _utc_now()
            run.write_manifest(manifest)
            run.write_summary(summaries)
            root_metadata.update(
                {
                    "status": "failed",
                    "finished_at": manifest["finished_at"],
                    "error": summary["error"],
                }
            )
            run.write_metadata(root_metadata)
            raise

        summaries.append(summary)
        run.write_summary_csv(summaries, COMPARISON_SUMMARY_FIELDS)
        manifest["runs"] = summaries
        run.write_manifest(manifest)

    comparison_rows = _aggregate(summaries)
    run.write_comparison_csv(comparison_rows, COMPARISON_FIELDS)
    run.write_comparison(comparison_rows)
    run_status = (
        "completed" if summaries and all(row["status"] == "completed" for row in summaries)
        else "partial"
    )
    manifest["status"] = run_status
    manifest["finished_at"] = _utc_now()
    manifest["comparison_rows"] = len(comparison_rows)
    run.write_summary(summaries)
    root_metadata["segments"] = [
        *previous_segments,
        {
            "started_at": segment_started_at,
            "finished_at": manifest["finished_at"],
            "status": run_status,
            "updates": max((int(row["updates"]) for row in summaries), default=0),
            "segment_updates": segment_updates,
        },
    ]
    root_metadata.update({"status": run_status, "finished_at": manifest["finished_at"]})
    run.write_metadata(root_metadata)
    # This is the final commit marker.  Every artifact needed to interpret the
    # completed/partial state is already durable before the status becomes
    # externally visible, so a kill can only leave a recoverable non-final root.
    run.write_manifest(manifest)
    return run, summaries, skipped


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--gradient-modes",
        nargs="+",
        help="optional subset of configured belief gradient modes",
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument(
        "--segment-updates",
        type=int,
        help="run at most this many additional updates and commit resumable state",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="fixed result root used by one resumable mode/method/task/seed shard",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume run-dir when it exists, or initialize it on the first segment",
    )
    parser.add_argument("--label")
    parser.add_argument(
        "--campaign-id",
        help="shared safe identifier used to keep independently executed shards together",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--bulk-index",
        type=int,
        help=(
            "one-based index into the configured eligible "
            "gradient-mode/method/task/seed matrix"
        ),
    )
    parser.add_argument(
        "--campaign-shard-index",
        type=int,
        help="one-based manifest index for an externally filtered bulk campaign",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_config(args.config, args.override)
        methods, tasks, seeds = _selection(config, args.methods, args.tasks, args.seeds)
        gradient_modes = _gradient_mode_selection(config, args.gradient_modes)
        run_label = args.label
        if args.bulk_index is not None and args.campaign_shard_index is not None:
            raise ConfigError("--bulk-index and --campaign-shard-index are mutually exclusive")
        if args.campaign_shard_index is not None:
            if args.campaign_shard_index <= 0:
                raise ConfigError("--campaign-shard-index must be positive")
            if not args.validate_only and args.campaign_id is None:
                raise ConfigError(
                    "--campaign-id is required with --campaign-shard-index"
                )
        if args.bulk_index is not None:
            if any(
                value is not None
                for value in (args.gradient_modes, args.methods, args.tasks, args.seeds)
            ):
                raise ConfigError(
                    "--bulk-index cannot be combined with gradient-mode/method/task/seed filters"
                )
            matrix = [
                (gradient_mode, method, task, seed)
                for gradient_mode in gradient_modes
                for method in methods
                for task in tasks
                if _eligible(config, method, task)[0]
                for seed in seeds
            ]
            if not 1 <= args.bulk_index <= len(matrix):
                raise ConfigError(f"--bulk-index must be in [1, {len(matrix)}]")
            selected_gradient_mode, selected_method, selected_task, selected_seed = matrix[
                args.bulk_index - 1
            ]
            gradient_modes = [selected_gradient_mode]
            methods, tasks, seeds = [selected_method], [selected_task], [selected_seed]
            run_label = run_label or f"genkai-prod-bulk{args.bulk_index:03d}"
            if not args.validate_only and args.campaign_id is None:
                raise ConfigError("--campaign-id is required when executing a bulk shard")
        if args.campaign_id is not None and not _CAMPAIGN_PATTERN.fullmatch(args.campaign_id):
            raise ConfigError("--campaign-id contains unsafe characters")
        validate_all_environments(config, tasks)
        if args.max_updates is not None and args.max_updates <= 0:
            raise ConfigError("--max-updates must be positive")
        if args.segment_updates is not None:
            if args.segment_updates <= 0:
                raise ConfigError("--segment-updates must be positive")
            if args.max_updates is not None:
                raise ConfigError("--segment-updates and --max-updates are mutually exclusive")
            if args.run_dir is None or not args.resume:
                raise ConfigError("--segment-updates requires --run-dir and --resume")
        if args.resume and args.run_dir is None:
            raise ConfigError("--resume requires --run-dir")
        eligible = [
            {
                "belief_gradient_mode": gradient_mode,
                "method": method,
                "task": task,
            }
            for gradient_mode in gradient_modes
            for method in methods
            for task in tasks
            if _eligible(config, method, task)[0]
        ]
        skipped = [
            {
                "belief_gradient_mode": gradient_mode,
                "method": method,
                "task": task,
                "reason": _eligible(config, method, task)[1],
            }
            for gradient_mode in gradient_modes
            for method in methods
            for task in tasks
            if not _eligible(config, method, task)[0]
        ]
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "belief_gradient_modes": gradient_modes,
                        "methods": methods,
                        "tasks": tasks,
                        "seeds": seeds,
                        "eligible_method_tasks": eligible,
                        "skipped": skipped,
                        "campaign_id": args.campaign_id,
                        "bulk_index": args.bulk_index or args.campaign_shard_index,
                        "segment_updates": args.segment_updates,
                        "run_dir": None if args.run_dir is None else str(args.run_dir),
                        "resume": args.resume,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        run, _, _ = run_comparisons(
            config,
            methods=methods,
            tasks=tasks,
            seeds=seeds,
            gradient_modes=gradient_modes,
            max_updates=args.max_updates,
            label=run_label,
            verbose=not args.quiet,
            campaign_id=args.campaign_id,
            bulk_index=args.bulk_index or args.campaign_shard_index,
            segment_updates=args.segment_updates,
            run_dir=args.run_dir,
            resume=args.resume,
        )
    except Exception as exc:  # noqa: BLE001 - concise CLI boundary.
        print(f"Comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(run.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_METHODS",
    "COMPARISON_FIELDS",
    "COMPARISON_SUMMARY_FIELDS",
    "SCORE_METHODS",
    "main",
    "run_comparisons",
]
