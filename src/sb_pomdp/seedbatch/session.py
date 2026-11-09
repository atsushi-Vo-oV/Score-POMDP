"""Isolated cooperative training loop, adapted from train_single at 44a1a52.

The scheduling changes are explicit suspension points for batched tensor work.
All host-side bookkeeping and artifact semantics follow the original trainer.
This is ordinary source code, not runtime rewriting or global monkey-patching.
"""
from __future__ import annotations

from .protocol import Request, sample_request, policy_request
from ..train import (
    Any,
    BeliefContext,
    BeliefReplayHistory,
    EPISODE_FIELDS,
    EVALUATION_FIELDS,
    ExperimentConfig,
    LOGGER,
    METRIC_FIELDS,
    Path,
    PredictionSchedule,
    RolloutBuilder,
    SeedArtifacts,
    SyncEnvironmentBatch,
    ValueTransform,
    _checkpoint_payload,
    _checkpoint_rollback_guidance,
    _environment_actions,
    _finish_timed_phase,
    _load_resume_checkpoint,
    _log_langevin_schedules,
    _next_context,
    _restore_global_rng_states,
    _restore_trainer_state,
    _start_timed_phase,
    _trainer_state_payload,
    _truncate_uncommitted_csv_rows,
    belief_latent_dim,
    build_optimizer,
    configure_logging,
    control_conditions,
    encode_action,
    encode_conditions,
    evaluate,
    make_env,
    math,
    model_for_environment,
    np,
    random_actions,
    resolve_device,
    runtime_metadata,
    save_checkpoint,
    seed_everything,
    time,
    torch,
    truncated_step_indices,
    truncation_bootstrap_values,
    value_from_conditions,
)


def training_session(
    config: ExperimentConfig,
    *,
    task: str,
    seed: int,
    artifacts: SeedArtifacts,
    max_updates: int | None = None,
    verbose: bool = True,
    method: str | None = None,
    result_dir: str | None = None,
    resume_checkpoint: str | Path | None = None,
    evaluate_at_end: bool = True,
    segment_updates: int | None = None,
):
    if not isinstance(evaluate_at_end, bool):
        raise TypeError("evaluate_at_end must be boolean")
    if max_updates is not None and segment_updates is not None:
        raise ValueError("max_updates and segment_updates are mutually exclusive")
    if max_updates is not None and (
        isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates <= 0
    ):
        raise ValueError("max_updates must be positive when supplied")
    if segment_updates is not None and (
        isinstance(segment_updates, bool)
        or not isinstance(segment_updates, int)
        or segment_updates <= 0
    ):
        raise ValueError("segment_updates must be positive when supplied")

    resolved = config.to_dict()
    experiment_config = resolved["experiment"]
    environment_config = {"name": task, **config.environment_for_task(task)}
    model_config = resolved["model"]
    ppo_config = resolved["ppo"]
    device = resolve_device(experiment_config["device"])
    seed_everything(seed, bool(experiment_config["deterministic"]))
    configure_logging(artifacts.path / "run.log", verbose=verbose)

    rollout_batch_size = int(ppo_config["num_envs"]) * int(ppo_config["rollout_steps"])
    prediction_schedule = PredictionSchedule.from_config(
        model_config,
        rollout_batch_size=rollout_batch_size,
        ppo_updates=math.ceil(int(ppo_config["total_steps"]) / rollout_batch_size),
    )
    configured_updates = prediction_schedule.total_updates
    bootstrap_on_truncation = bool(ppo_config["bootstrap_on_truncation"])
    value_transform = ValueTransform.from_config(ppo_config)
    checkpoint: dict[str, Any] | None = None
    start_update = 0
    if resume_checkpoint is not None:
        checkpoint = _load_resume_checkpoint(
            resume_checkpoint,
            config=resolved,
            task=task,
            seed=seed,
            method=method,
            rollout_batch_size=rollout_batch_size,
            configured_updates=configured_updates,
        )
        start_update = int(checkpoint["update"])
    if segment_updates is not None:
        total_updates = min(configured_updates, start_update + segment_updates)
    elif max_updates is not None:
        total_updates = min(configured_updates, max_updates)
    else:
        total_updates = configured_updates
    if total_updates < start_update or (
        total_updates == start_update and start_update != configured_updates
    ):
        raise ValueError(
            "requested training target must be greater than the resume checkpoint update "
            f"({start_update})"
        )

    artifacts.write_resolved_config(resolved)
    metadata = runtime_metadata(device)
    metadata.update(
        {
            "task": task,
            "seed": seed,
            "method": method,
            "max_updates": max_updates,
            "segment_updates": segment_updates,
            "resume_checkpoint": (
                None if resume_checkpoint is None else str(Path(resume_checkpoint).expanduser())
            ),
        }
    )
    artifacts.write_metadata(metadata)
    LOGGER.info("Starting task=%s seed=%d device=%s", task, seed, device)

    environments = [
        make_env(task, environment_config, seed=seed * 10_000 + index)
        for index in range(int(ppo_config["num_envs"]))
    ]
    batch = SyncEnvironmentBatch(environments)
    first_environment = environments[0]
    model = model_for_environment(first_environment, model_config, device)
    optimizer = build_optimizer(model, ppo_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Model parameters: %d", parameter_count)

    reset_seeds = [seed * 10_000 + index for index in range(batch.size)]
    observation = batch.reset(reset_seeds)
    context = BeliefContext(
        observation=torch.as_tensor(observation, dtype=torch.float32, device=device),
        previous_particles=torch.zeros(
            batch.size,
            int(model_config["num_particles"]),
            belief_latent_dim(first_environment, model_config),
            dtype=torch.float32,
            device=device,
        ),
        previous_action=torch.zeros(
            batch.size,
            first_environment.action_spec.feature_dim,
            dtype=torch.float32,
            device=device,
        ),
        initial=torch.ones(batch.size, dtype=torch.bool, device=device),
    )
    belief_histories = [BeliefReplayHistory() for _ in range(batch.size)]

    episode_returns = np.zeros(batch.size, dtype=np.float64)
    episode_lengths = np.zeros(batch.size, dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    episode_appender = artifacts.csv_appender(EPISODE_FIELDS, name="episodes")
    best_evaluation = -math.inf
    latest_evaluation: dict[str, float] | None = None
    accumulated_wall_time = 0.0
    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
        except (TypeError, ValueError, RuntimeError, KeyError) as exc:
            raise ValueError("resume checkpoint model or optimizer state is incompatible") from exc
        try:
            (
                context,
                belief_histories,
                episode_returns,
                episode_lengths,
                completed_returns,
                completed_lengths,
                metric_rows,
                best_evaluation,
                latest_evaluation,
                accumulated_wall_time,
            ) = _restore_trainer_state(
                checkpoint["trainer_state"],
                environments=environments,
                update=start_update,
                rollout_batch_size=rollout_batch_size,
                model_config=model_config,
                device=device,
            )
        except (TypeError, ValueError) as exc:
            # Rejected state (non-finite beliefs above all) stays fail-fast, but
            # the operator needs the concrete rollback target in the message.
            assert resume_checkpoint is not None
            raise type(exc)(
                f"{exc}{_checkpoint_rollback_guidance(resume_checkpoint)}"
            ) from exc
        _restore_global_rng_states(checkpoint, device=device)
        LOGGER.info(
            "Resuming task=%s seed=%d from update=%d step=%d",
            task,
            seed,
            start_update,
            start_update * rollout_batch_size,
        )
        _truncate_uncommitted_csv_rows(
            artifacts,
            update=start_update,
            global_step=start_update * rollout_batch_size,
        )
        if not artifacts.metrics_path.exists() or artifacts.metrics_path.stat().st_size == 0:
            artifacts.csv_appender(METRIC_FIELDS).append_many(metric_rows)
    segment_started = time.perf_counter()

    def current_checkpoint_payload(update: int, global_step: int) -> dict[str, Any]:
        wall_time_seconds = accumulated_wall_time + (time.perf_counter() - segment_started)
        trainer_state = _trainer_state_payload(
            environments=environments,
            context=context,
            belief_histories=belief_histories,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            completed_returns=completed_returns,
            completed_lengths=completed_lengths,
            metric_rows=metric_rows,
            best_evaluation=best_evaluation,
            latest_evaluation=latest_evaluation,
            wall_time_seconds=wall_time_seconds,
        )
        return _checkpoint_payload(
            model,
            optimizer,
            task=task,
            seed=seed,
            update=update,
            global_step=global_step,
            config=resolved,
            trainer_state=trainer_state,
            method=method,
        )

    yield Request("ready", (model, optimizer, start_update, total_updates))

    for update in range(start_update + 1, total_updates + 1):
        update_started = time.perf_counter()
        phase = prediction_schedule.phase(update)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rollout_started = _start_timed_phase(
            "rollout_collection",
            update=update,
            total_updates=total_updates,
            device=device,
        )
        prefixes = tuple(
            history.snapshot(
                context,
                environment_index,
                langevin_steps=int(model_config["langevin_steps"]),
            )
            for environment_index, history in enumerate(belief_histories)
        )
        builder = RolloutBuilder(belief_prefixes=prefixes)
        update_returns: list[float] = []
        update_lengths: list[int] = []
        # Keep rollout diagnostics on the accelerator and transfer their small
        # aggregate once at the phase boundary.  Calling ``.cpu()``/``.item()``
        # inside every environment step serialises the CUDA stream and can make
        # a diagnostic substantially more expensive than the computation it
        # measures.
        diagnostic_float = torch.zeros((), dtype=torch.float64, device=device)
        diagnostic_count = torch.zeros((), dtype=torch.int64, device=device)
        score_norm_sum_device = diagnostic_float.clone()
        score_norm_count = 0
        initial_score_norm_sum_device = diagnostic_float.clone()
        initial_score_norm_count_device = diagnostic_count.clone()
        recursive_score_norm_sum_device = diagnostic_float.clone()
        recursive_score_norm_count_device = diagnostic_count.clone()
        initial_particle_norm_sum_device = diagnostic_float.clone()
        initial_particle_norm_count_device = diagnostic_count.clone()
        recursive_particle_norm_sum_device = diagnostic_float.clone()
        recursive_particle_norm_count_device = diagnostic_count.clone()
        score_nonfinite_count_device = diagnostic_count.clone()
        particle_nonfinite_count_device = diagnostic_count.clone()
        particle_abs_max_device = torch.zeros((), dtype=torch.float32, device=device)
        particle_clip_fraction_sum_device = diagnostic_float.clone()
        action_saturation_fraction_sum_device = diagnostic_float.clone()
        action_saturation_fraction_count = 0
        truncation_bootstrap_sum = 0.0
        truncation_bootstrap_count = 0
        episode_rows: list[dict[str, Any]] = []
        model.eval()

        for rollout_index in range(int(ppo_config["rollout_steps"])):
            # Diagnostics describe the belief sampled for the current input,
            # before ``_next_context`` advances/reset the vector environments.
            belief_initial = context.initial.detach()
            particles, scores, ula_initial_noise, ula_step_noises = (
                (yield from sample_request(model, context, model_config, rollout=True))
            )
            (
                action,
                old_log_prob,
                _,
                diffusion_chain,
                pre_tanh_action,
                old_value,
            ) = yield from policy_request(model, particles, scores)
            if phase == "pretrain":
                # The pretraining block explores with uniformly random actions; the policy
                # outputs above are discarded (the prediction objective does not read them).
                action = random_actions(first_environment.action_spec, batch.size, device=device)
            environment_action = _environment_actions(first_environment.action_spec, action)
            # ``action_features`` is a pure function of the sampled action.  It
            # is materialised here because a truncation bootstrap conditions the
            # discarded final observation on the action that produced it.
            action_features = encode_action(first_environment.action_spec, action)
            next_observation, reward, done, step_infos = batch.step(environment_action)
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=device)
            done_tensor = torch.as_tensor(done, dtype=torch.bool, device=device)
            truncation_values: torch.Tensor | None = None
            if bootstrap_on_truncation:
                truncation_values = truncation_bootstrap_values(
                    model,
                    model_config,
                    infos=step_infos,
                    particles=particles,
                    action_features=action_features,
                    device=device,
                    value_transform=value_transform,
                )
                truncation_bootstrap_sum += float(truncation_values.sum().cpu())
                truncation_bootstrap_count += len(truncated_step_indices(step_infos))

            builder.append(
                observations=context.observation,
                previous_particles=context.previous_particles,
                previous_actions=context.previous_action,
                initial=context.initial,
                particles=particles,
                actions=action,
                old_log_prob=old_log_prob,
                old_values=value_transform.to_raw(old_value),
                rewards=reward_tensor,
                dones=done_tensor,
                ula_initial_noise=ula_initial_noise,
                ula_step_noises=ula_step_noises,
                diffusion_chains=diffusion_chain,
                pre_tanh_actions=pre_tanh_action,
                truncation_values=truncation_values,
            )
            for environment_index, history in enumerate(belief_histories):
                history.append(
                    context,
                    environment_index,
                    ula_initial_noise,
                    ula_step_noises,
                )
                if bool(done[environment_index]):
                    history.clear()
            if first_environment.action_spec.is_continuous:
                action_saturation_fraction_sum_device = (
                    action_saturation_fraction_sum_device
                    + (action_features.abs() >= 0.99).float().mean().double()
                )
                action_saturation_fraction_count += 1
            context = _next_context(
                observation=next_observation,
                particles=particles,
                action_features=action_features,
                done=done,
                device=device,
            )
            episode_returns += reward
            episode_lengths += 1
            score_l2 = scores.norm(dim=-1)
            particle_l2 = particles.norm(dim=-1)
            initial_mask = belief_initial[:, None].expand_as(score_l2)
            recursive_mask = ~initial_mask
            finite_score_l2 = torch.nan_to_num(score_l2, nan=0.0, posinf=0.0, neginf=0.0)
            finite_particle_l2 = torch.nan_to_num(
                particle_l2, nan=0.0, posinf=0.0, neginf=0.0
            )
            score_norm_sum_device = score_norm_sum_device + finite_score_l2.sum().double()
            score_norm_count += score_l2.numel()
            initial_score_norm_sum_device = (
                initial_score_norm_sum_device
                + (finite_score_l2 * initial_mask).sum().double()
            )
            initial_score_norm_count_device = (
                initial_score_norm_count_device + initial_mask.sum()
            )
            initial_particle_norm_sum_device = (
                initial_particle_norm_sum_device
                + (finite_particle_l2 * initial_mask).sum().double()
            )
            initial_particle_norm_count_device = (
                initial_particle_norm_count_device + initial_mask.sum()
            )
            recursive_score_norm_sum_device = (
                recursive_score_norm_sum_device
                + (finite_score_l2 * recursive_mask).sum().double()
            )
            recursive_score_norm_count_device = (
                recursive_score_norm_count_device + recursive_mask.sum()
            )
            recursive_particle_norm_sum_device = (
                recursive_particle_norm_sum_device
                + (finite_particle_l2 * recursive_mask).sum().double()
            )
            recursive_particle_norm_count_device = (
                recursive_particle_norm_count_device + recursive_mask.sum()
            )
            score_nonfinite_count_device = (
                score_nonfinite_count_device + (~torch.isfinite(scores)).sum()
            )
            particle_nonfinite_count_device = (
                particle_nonfinite_count_device + (~torch.isfinite(particles)).sum()
            )
            particle_abs_max_device = torch.maximum(
                particle_abs_max_device, particles.abs().max()
            )
            particle_clip = float(model_config["particle_clip"])
            if particle_clip > 0.0:
                particle_clip_fraction_sum_device = (
                    particle_clip_fraction_sum_device
                    + (particles.abs() >= particle_clip - 1e-6).float().mean().double()
                )
            for environment_index in np.flatnonzero(done):
                episode_return = float(episode_returns[environment_index])
                episode_length = int(episode_lengths[environment_index])
                if phase != "pretrain":  # random-action episodes stay out of the learning curve
                    update_returns.append(episode_return)
                    update_lengths.append(episode_length)
                    completed_returns.append(episode_return)
                    completed_lengths.append(episode_length)
                    episode_rows.append(
                        {
                            "global_step": (
                                (update - 1) * rollout_batch_size
                                + (rollout_index + 1) * batch.size
                            ),
                            "environment": int(environment_index),
                            "episode_return": episode_return,
                            "episode_length": episode_length,
                        }
                    )
                episode_returns[environment_index] = 0.0
                episode_lengths[environment_index] = 0

        episode_appender.append_many(episode_rows)
        if bootstrap_on_truncation:
            # Reported through the log rather than metrics.csv: METRIC_FIELDS is
            # a closed CSV schema that every existing run and every reader
            # shares, and making it depend on a config flag would fork the
            # artifact contract for one diagnostic.
            LOGGER.info(
                "truncation_bootstrap update=%d truncations=%d value_mean=%s",
                update,
                truncation_bootstrap_count,
                (
                    f"{truncation_bootstrap_sum / truncation_bootstrap_count:.6f}"
                    if truncation_bootstrap_count
                    else "n/a"
                ),
            )
        bootstrap_particles, bootstrap_scores = (yield from sample_request(model, context, model_config, rollout=False))
        with torch.no_grad():
            policy_condition, critic_condition = encode_conditions(
                model, bootstrap_particles, bootstrap_scores
            )
            policy_condition, critic_condition = control_conditions(
                model, policy_condition, critic_condition, context.observation
            )
            last_value = value_transform.to_raw(
                value_from_conditions(model, policy_condition, critic_condition)
            )
        rollout = builder.finish(
            last_value=last_value,
            wml_weights=None,
            gamma=float(ppo_config["gamma"]),
            gae_lambda=float(ppo_config["gae_lambda"]),
        )
        _finish_timed_phase(
            "rollout_collection",
            rollout_started,
            update=update,
            total_updates=total_updates,
            device=device,
        )
        score_norm_sum = float(score_norm_sum_device.cpu())
        initial_score_norm_sum = float(initial_score_norm_sum_device.cpu())
        initial_score_norm_count = int(initial_score_norm_count_device.cpu())
        recursive_score_norm_sum = float(recursive_score_norm_sum_device.cpu())
        recursive_score_norm_count = int(recursive_score_norm_count_device.cpu())
        initial_particle_norm_sum = float(initial_particle_norm_sum_device.cpu())
        initial_particle_norm_count = int(initial_particle_norm_count_device.cpu())
        recursive_particle_norm_sum = float(recursive_particle_norm_sum_device.cpu())
        recursive_particle_norm_count = int(recursive_particle_norm_count_device.cpu())
        score_nonfinite_count = int(score_nonfinite_count_device.cpu())
        particle_nonfinite_count = int(particle_nonfinite_count_device.cpu())
        particle_abs_max = float(particle_abs_max_device.cpu())
        particle_clip_fraction = float(particle_clip_fraction_sum_device.cpu()) / int(
            ppo_config["rollout_steps"]
        )
        action_saturation_fraction = (
            float(action_saturation_fraction_sum_device.cpu())
            / action_saturation_fraction_count
            if action_saturation_fraction_count
            else float("nan")
        )
        model.train()
        ppo_started = _start_timed_phase(
            "ppo_update",
            update=update,
            total_updates=total_updates,
            device=device,
        )
        ppo_metrics = yield Request(
            "update", (rollout, "ppo" if phase == "ppo" else "prediction")
        )
        if phase != "ppo":
            LOGGER.info(
                "prediction phase=%s update=%d next_observation_nll=%.4f",
                phase,
                update,
                ppo_metrics.policy_loss,
            )
        _finish_timed_phase(
            "ppo_update",
            ppo_started,
            update=update,
            total_updates=total_updates,
            device=device,
        )
        recondition_started = _start_timed_phase(
            "recondition",
            update=update,
            total_updates=total_updates,
            device=device,
        )
        rebuilt = yield Request("recondition", (context, belief_histories))
        context = BeliefContext(context.observation, rebuilt, context.previous_action, context.initial)
        _finish_timed_phase(
            "recondition",
            recondition_started,
            update=update,
            total_updates=total_updates,
            device=device,
        )

        global_step = update * rollout_batch_size
        duration = max(time.perf_counter() - update_started, 1e-9)
        row = {
            "update": update,
            "global_step": global_step,
            "policy_loss": ppo_metrics.policy_loss,
            "value_loss": ppo_metrics.value_loss,
            "entropy": ppo_metrics.entropy,
            "approximate_kl": ppo_metrics.approximate_kl,
            "clip_fraction": ppo_metrics.clip_fraction,
            "gradient_norm": ppo_metrics.gradient_norm,
            "initial_energy_gradient_norm": ppo_metrics.initial_energy_gradient_norm,
            "observation_energy_gradient_norm": ppo_metrics.observation_energy_gradient_norm,
            "transition_energy_gradient_norm": ppo_metrics.transition_energy_gradient_norm,
            "episode_return_mean": (
                float(np.mean(update_returns)) if update_returns else float("nan")
            ),
            "episode_length_mean": (
                float(np.mean(update_lengths)) if update_lengths else float("nan")
            ),
            "episodes_completed": len(update_returns),
            "score_l2_mean": score_norm_sum / score_norm_count,
            "initial_score_l2_mean": (
                initial_score_norm_sum / initial_score_norm_count
                if initial_score_norm_count
                else float("nan")
            ),
            "recursive_score_l2_mean": (
                recursive_score_norm_sum / recursive_score_norm_count
                if recursive_score_norm_count
                else float("nan")
            ),
            "initial_particle_l2_mean": (
                initial_particle_norm_sum / initial_particle_norm_count
                if initial_particle_norm_count
                else float("nan")
            ),
            "recursive_particle_l2_mean": (
                recursive_particle_norm_sum / recursive_particle_norm_count
                if recursive_particle_norm_count
                else float("nan")
            ),
            "initial_belief_particle_count": initial_particle_norm_count,
            "recursive_belief_particle_count": recursive_particle_norm_count,
            "score_nonfinite_count": score_nonfinite_count,
            "particle_nonfinite_count": particle_nonfinite_count,
            "particle_abs_max": particle_abs_max,
            "particle_clip_fraction": particle_clip_fraction,
            "action_saturation_fraction": action_saturation_fraction,
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
            "steps_per_second": rollout_batch_size / duration,
        }
        metric_rows.append(row)
        artifacts.append_metrics(row, METRIC_FIELDS)
        if update % int(experiment_config["log_interval_updates"]) == 0:
            LOGGER.info(
                "update=%d/%d step=%d policy=%.4f value=%.4f return=%s fps=%.1f",
                update,
                total_updates,
                global_step,
                ppo_metrics.policy_loss,
                ppo_metrics.value_loss,
                "n/a" if not update_returns else f"{np.mean(update_returns):.3f}",
                row["steps_per_second"],
            )
            _log_langevin_schedules(model, update)

        should_evaluate = (
            phase == "ppo" and update % int(experiment_config["eval_interval_updates"]) == 0
        ) or (
            update == configured_updates
            or (
                evaluate_at_end
                and segment_updates is None
                and update == total_updates
            )
        )
        if should_evaluate:
            evaluation_started = _start_timed_phase(
                "evaluation",
                update=update,
                total_updates=total_updates,
                device=device,
            )
            evaluation_seed = seed + 1_000_000
            if first_environment.action_spec.is_discrete:
                deterministic_mode = "greedy"
            elif model.continuous_policy_kind == "diffusion":
                deterministic_mode = "mean_chain"
            else:
                deterministic_mode = "mean_action"
            for mode, deterministic_evaluation in (
                (deterministic_mode, True),
                ("stochastic", False),
            ):
                evaluation, trajectories = evaluate(
                    model,
                    task=task,
                    environment_config=environment_config,
                    model_config=model_config,
                    episodes=int(experiment_config["eval_episodes"]),
                    seed=evaluation_seed,
                    device=device,
                    deterministic=deterministic_evaluation,
                )
                evaluation_row = {
                    "update": update,
                    "global_step": global_step,
                    "mode": mode,
                    "evaluation_seed": evaluation_seed,
                    **evaluation,
                }
                artifacts.csv_appender(EVALUATION_FIELDS, name="evaluation").append(evaluation_row)
                artifacts.save_npz(
                    trajectories,
                    name=f"evaluation_{mode}_trajectories_{update:06d}",
                )
                LOGGER.info(
                    "evaluation mode=%s update=%d return=%.3f +/- %.3f",
                    mode,
                    update,
                    evaluation["mean_return"],
                    evaluation["std_return"],
                )
                if deterministic_evaluation:
                    latest_evaluation = evaluation
            _finish_timed_phase(
                "evaluation",
                evaluation_started,
                update=update,
                total_updates=total_updates,
                device=device,
            )
            assert latest_evaluation is not None
            if latest_evaluation["mean_return"] > best_evaluation:
                best_evaluation = latest_evaluation["mean_return"]
                save_checkpoint(
                    current_checkpoint_payload(update, global_step),
                    artifacts.checkpoint_path(tag="best"),
                )

        should_save_step_checkpoint = (
            update % int(experiment_config["checkpoint_interval_updates"]) == 0
            or update == total_updates
        )
        if should_save_step_checkpoint or segment_updates is not None:
            payload = current_checkpoint_payload(update, global_step)
            if should_save_step_checkpoint:
                save_checkpoint(payload, artifacts.checkpoint_path(step=global_step))
            save_checkpoint(payload, artifacts.checkpoint_path(tag="latest"))

    numeric_fields = [field for field in METRIC_FIELDS if field not in {"update", "global_step"}]
    arrays: dict[str, np.ndarray] = {
        "update": np.asarray([row["update"] for row in metric_rows], dtype=np.int64),
        "global_step": np.asarray([row["global_step"] for row in metric_rows], dtype=np.int64),
        "episode_returns": np.asarray(completed_returns, dtype=np.float32),
        "episode_lengths": np.asarray(completed_lengths, dtype=np.int32),
    }
    for field in numeric_fields:
        arrays[field] = np.asarray([row[field] for row in metric_rows], dtype=np.float64)
    artifacts.save_npz(arrays)

    # ``latest_evaluation`` is the deterministic-readout evaluation of the last
    # evaluated update, so both keys below intentionally hold the same number.
    readout_return_mean = None if latest_evaluation is None else latest_evaluation["mean_return"]
    summary = {
        "task": task,
        "seed": seed,
        "status": "completed" if total_updates == configured_updates else "partial",
        "updates": total_updates,
        "global_steps": total_updates * rollout_batch_size,
        "episodes_completed": len(completed_returns),
        "training_return_mean": (float(np.mean(completed_returns)) if completed_returns else None),
        "evaluation_return_mean": readout_return_mean,
        "deterministic_readout_return_mean": readout_return_mean,
        "parameter_count": parameter_count,
        "wall_time_seconds": accumulated_wall_time + (time.perf_counter() - segment_started),
        "result_dir": result_dir or f"{task}/seed_{seed:03d}",
    }
    LOGGER.info(
        "%s task=%s seed=%d at update=%d/%d",
        "Completed" if total_updates == configured_updates else "Paused",
        task,
        seed,
        total_updates,
        configured_updates,
    )
    return summary
