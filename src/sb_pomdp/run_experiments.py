"""Run every configured task/seed into one timestamped result directory."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import RunArtifacts
from .config import ConfigError, ExperimentConfig, load_config
from .envs import make_env
from .train import SUMMARY_FIELDS, resolve_device, runtime_metadata, train_single


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _selection(
    config: ExperimentConfig,
    requested_tasks: list[str] | None,
    requested_seeds: list[int] | None,
) -> tuple[list[str], list[int]]:
    configured_tasks = list(config["experiment"]["tasks"])
    configured_seeds = list(config["experiment"]["seeds"])
    tasks = configured_tasks if requested_tasks is None else requested_tasks
    seeds = configured_seeds if requested_seeds is None else requested_seeds
    unknown = sorted(set(tasks) - set(configured_tasks))
    if unknown:
        raise ConfigError(f"requested tasks are not enabled by the config: {unknown}")
    if not tasks or not seeds:
        raise ConfigError("at least one task and seed must be selected")
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise ConfigError("seeds must be in [0, 2**32 - 1]")
    return list(dict.fromkeys(tasks)), list(dict.fromkeys(seeds))


def validate_all_environments(config: ExperimentConfig, tasks: list[str]) -> None:
    for task in tasks:
        environment = make_env(task, config.environment_for_task(task), seed=0)
        observation, _ = environment.reset(seed=0)
        if observation.shape != (environment.observation_dim,):
            raise RuntimeError(f"{task}: reset observation has an invalid shape")
        environment.action_spec.validate(environment.action_spec.sample(np.random.default_rng(0)))


def run_all(
    config: ExperimentConfig,
    *,
    tasks: list[str],
    seeds: list[int],
    max_updates: int | None,
    label: str | None,
    verbose: bool,
) -> tuple[RunArtifacts, list[dict[str, Any]]]:
    execution_data = config.to_dict()
    run_label = label or execution_data["experiment"]["label"]
    execution_data["experiment"]["label"] = run_label
    execution_data["experiment"]["tasks"] = tasks
    execution_data["experiment"]["seeds"] = seeds
    if max_updates is not None:
        rollout_batch = execution_data["ppo"]["num_envs"] * execution_data["ppo"]["rollout_steps"]
        execution_data["ppo"]["total_steps"] = min(
            execution_data["ppo"]["total_steps"],
            max_updates * rollout_batch,
        )
    execution_config = ExperimentConfig(execution_data, source=config.source)
    experiment = execution_config["experiment"]
    run = RunArtifacts.create(experiment["output_dir"], run_label)
    run.write_resolved_config(execution_config.to_dict())
    device = resolve_device(experiment["device"])
    root_metadata = runtime_metadata(device)
    root_metadata.update(
        {
            "started_at": _utc_now(),
            "tasks": tasks,
            "seeds": seeds,
            "max_updates": max_updates,
            "label": run_label,
        }
    )
    run.write_metadata(root_metadata)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": root_metadata["started_at"],
        "tasks": tasks,
        "seeds": seeds,
        "max_updates": max_updates,
        "runs": [],
    }
    run.write_manifest(manifest)
    summaries: list[dict[str, Any]] = []

    for task in tasks:
        for seed in seeds:
            seed_artifacts = run.for_seed(task, seed)
            try:
                summary = train_single(
                    execution_config,
                    task=task,
                    seed=seed,
                    artifacts=seed_artifacts,
                    max_updates=max_updates,
                    verbose=verbose,
                )
                summary["error"] = ""
            except Exception as exc:
                summary = {
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
                    "result_dir": f"{task}/seed_{seed:03d}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                (seed_artifacts.path / "failure.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                summaries.append(summary)
                run.append_summary(summary, SUMMARY_FIELDS)
                manifest["runs"] = summaries
                manifest["status"] = "failed"
                manifest["finished_at"] = _utc_now()
                run.write_manifest(manifest)
                run.write_summary(summaries)
                raise
            summaries.append(summary)
            run.append_summary(summary, SUMMARY_FIELDS)
            manifest["runs"] = summaries
            run.write_manifest(manifest)
    manifest["status"] = "completed"
    manifest["finished_at"] = _utc_now()
    run.write_manifest(manifest)
    run.write_summary(summaries)
    return run, summaries


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--label")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_config(args.config, args.override)
        tasks, seeds = _selection(config, args.tasks, args.seeds)
        validate_all_environments(config, tasks)
        if args.max_updates is not None and args.max_updates <= 0:
            raise ConfigError("--max-updates must be positive")
        if args.validate_only:
            print(
                json.dumps(
                    {"status": "valid", "tasks": tasks, "seeds": seeds},
                    ensure_ascii=False,
                )
            )
            return 0
        run, _ = run_all(
            config,
            tasks=tasks,
            seeds=seeds,
            max_updates=args.max_updates,
            label=args.label,
            verbose=not args.quiet,
        )
    except Exception as exc:  # noqa: BLE001 - a CLI must report training failures cleanly.
        print(f"Experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(run.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
