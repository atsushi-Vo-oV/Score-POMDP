"""Per-shard artifact transaction, adapted from compare.run_comparisons at 44a1a52.

Uses the same identity checks, manifests, resume guards and completion markers.
Only the score trainer call is cooperative; validation restricts the entry point
to exactly one full/score_transformer/task/seed per shard.
"""
from __future__ import annotations

from .session import training_session
from .validation import source_digest
from ..compare import (
    Any,
    COMPARISON_FIELDS,
    COMPARISON_SUMMARY_FIELDS,
    ConfigError,
    ExperimentConfig,
    Path,
    RunArtifacts,
    UTC,
    _CAMPAIGN_PATTERN,
    _PERSISTED_IDENTITY_FIELDS,
    _RUNTIME_IDENTITY_FIELDS,
    _aggregate,
    _completed_shard_runs,
    _configured_updates,
    _eligible,
    _execution_config,
    _gradient_mode_selection,
    _method_config,
    _optional_final_evaluations,
    _read_json_object,
    _restartable_uncommitted_seed_roots,
    _resume_checkpoint_candidate,
    _utc_now,
    datetime,
    os,
    resolve_device,
    resume_comparable_config,
    runtime_metadata,
    source_sha256,
    sys,
    traceback,
)


def comparison_session(
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
            existing_config = resume_comparable_config(_read_json_object(run.resolved_config_path))
            if existing_config != resume_comparable_config(execution_config.to_dict()):
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
    seedbatch_source = source_digest()
    if resumed_run and previous_metadata.get("seedbatch_source_sha256") != seedbatch_source:
        raise ConfigError("seedbatch source code differs from the segment that created it")
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
        "seedbatch_source_sha256": seedbatch_source,
        "training_path": "seedbatch",
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
            summary = yield from training_session(
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
