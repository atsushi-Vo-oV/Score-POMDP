"""Explicit POPX fork of train_baseline_single (base commit 6711845).

Original rollout, recurrent history, prediction phases, evaluation and resume
protocol are retained. Population hooks are local; train_baselines is untouched.
"""
from __future__ import annotations
from ..train_baselines import (
    Any,
    BASELINE_CHECKPOINT_FORMAT_VERSION,
    BASELINE_METHODS,
    BaselineMethod,
    BaselineRollout,
    EPISODE_FIELDS,
    EVALUATION_FIELDS,
    ExperimentConfig,
    LOGGER,
    METRIC_FIELDS,
    Path,
    PredictionSchedule,
    RECURRENT_BASELINE_TYPES,
    RecurrentReplayHistory,
    RecurrentReplayPrefix,
    SIL_METRIC_FIELDS,
    SeedArtifacts,
    SelfImitationBuffer,
    SyncEnvironmentBatch,
    TemporalGradientMode,
    ValueTransform,
    _baseline_trainer_state_payload,
    _checkpoint_payload,
    _current_inputs,
    _environment_actions,
    _load_resume_checkpoint,
    _make_model,
    _recondition_recurrent_hidden,
    _restore_baseline_trainer_state,
    _restore_global_rng_states,
    _truncate_uncommitted_csv_rows,
    _update_feedforward,
    _update_recurrent_prediction,
    configure_logging,
    critic_only,
    encode_action,
    evaluate_baseline,
    generalised_advantage_estimate,
    initialize_model,
    make_env,
    math,
    np,
    random_actions,
    replace,
    resolve_device,
    runtime_metadata,
    save_checkpoint,
    seed_everything,
    sil_config,
    time,
    torch,
    truncated_step_indices,
    update_self_imitation)
from ..artifacts import CsvAppender
from .controller import Controller, DIAGNOSTIC_FIELDS
from .gru_learning import update_recurrent as _update_recurrent, truncation_values as _truncation_bootstrap_values


def train_single(
    config: ExperimentConfig,
    *,
    method: BaselineMethod,
    task: str,
    seed: int,
    artifacts: SeedArtifacts,
    max_updates: int | None = None,
    verbose: bool = True,
    result_dir: str,
    resume_checkpoint: str | Path | None = None,
    evaluate_at_end: bool = True,
    segment_updates: int | None = None,
) -> dict[str, Any]:
    """Train one comparison baseline with PPO and write one seed artifact tree.

    ``segment_updates`` runs at most that many further updates and commits a
    resumable checkpoint, and ``resume_checkpoint`` continues from one.  Both
    mirror :func:`sb_pomdp.train.train_single` so a baseline shard can be chained
    exactly like a score shard.
    """

    if method not in BASELINE_METHODS:
        raise ValueError(f"unsupported baseline method: {method!r}")
    if artifacts.method is not None and artifacts.method != method:
        raise ValueError(
            f"artifact method {artifacts.method!r} does not match requested method {method!r}"
        )
    if not isinstance(result_dir, str) or not result_dir:
        raise ValueError("result_dir must be a non-empty relative artifact description")
    if not isinstance(evaluate_at_end, bool):
        raise TypeError("evaluate_at_end must be boolean")
    if max_updates is not None and segment_updates is not None:
        raise ValueError("max_updates and segment_updates are mutually exclusive")
    if max_updates is not None and (
        isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates <= 0
    ):
        raise ValueError("max_updates must be a positive integer when supplied")
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
    imitation_config = sil_config(ppo_config)
    metric_fields = DIAGNOSTIC_FIELDS + METRIC_FIELDS + (SIL_METRIC_FIELDS if imitation_config["enabled"] else ())
    comparison_config = resolved["comparison"]
    device = resolve_device(experiment_config["device"])
    seed_everything(seed, bool(experiment_config["deterministic"]))
    configure_logging(artifacts.path / "run.log", verbose=verbose)

    rollout_batch_size = int(ppo_config["num_envs"]) * int(ppo_config["rollout_steps"])
    schedule_model_config = dict(model_config)
    if resolved["popexplore"]["freeze_belief"]:
        schedule_model_config.update(prediction_pretrain_steps=0, prediction_interval_updates=0, prediction_phase_updates=0)
    prediction_schedule = PredictionSchedule.from_config(
        schedule_model_config,
        rollout_batch_size=rollout_batch_size,
        ppo_updates=math.ceil(int(ppo_config["total_steps"]) / rollout_batch_size),
        critic_warmup_updates=int(ppo_config.get("critic_warmup_updates", 0)),
    )
    configured_updates = prediction_schedule.total_updates
    bootstrap_on_truncation = bool(ppo_config["bootstrap_on_truncation"])
    if str(ppo_config.get("algorithm", "ppo")) != "ppo":
        raise NotImplementedError(
            "baseline trainers support only ppo.algorithm=ppo for now"
        )
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
            format_version=BASELINE_CHECKPOINT_FORMAT_VERSION,
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
    LOGGER.info("Starting method=%s task=%s seed=%d device=%s", method, task, seed, device)

    environments = [
        make_env(task, environment_config, seed=seed * 10_000 + index)
        for index in range(int(ppo_config["num_envs"]))
    ]
    batch = SyncEnvironmentBatch(environments)
    first_environment = environments[0]
    model = _make_model(
        first_environment,
        method,
        model_config,
        comparison_config,
        device,
    )
    initialize_model(model, model_config, resume=checkpoint is not None)
    controller = Controller(model, resolved, task, seed, artifacts, first_environment, device, resume=checkpoint is not None)
    object.__setattr__(model, "_popx", controller)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=float(ppo_config["learning_rate"]), eps=1e-5
    )
    controller.exclude_actor_parameters(optimizer)
    save_checkpoint = controller.save_checkpoint
    imitation = (
        SelfImitationBuffer(int(imitation_config["buffer_size"]), batch.size, float(ppo_config["gamma"]))
        if imitation_config["enabled"] else None
    )
    if imitation is not None and checkpoint is not None:
        if "self_imitation" not in checkpoint:
            raise ValueError("enabled self-imitation requires replay state in the resume checkpoint")
        imitation.load_state_dict(checkpoint["self_imitation"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Model parameters: %d", parameter_count)

    observation = batch.reset([seed * 10_000 + index for index in range(batch.size)])
    previous_action = torch.zeros(
        batch.size,
        first_environment.action_spec.feature_dim,
        dtype=torch.float32,
        device=device,
    )
    episode_starts = torch.ones(batch.size, dtype=torch.bool, device=device)
    hidden = (
        model.initial_hidden(batch.size, device=device)
        if isinstance(model, RECURRENT_BASELINE_TYPES)
        else None
    )
    recurrent_histories = (
        [RecurrentReplayHistory() for _ in range(batch.size)]
        if isinstance(model, RECURRENT_BASELINE_TYPES)
        else []
    )

    episode_returns = np.zeros(batch.size, dtype=np.float64)
    episode_lengths = np.zeros(batch.size, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    metric_rows: list[dict[str, Any]] = []
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
        (
            observation,
            previous_action,
            episode_starts,
            recurrent_histories,
            episode_returns,
            episode_lengths,
            completed_returns,
            completed_lengths,
            metric_rows,
            best_evaluation,
            latest_evaluation,
            accumulated_wall_time,
        ) = _restore_baseline_trainer_state(
            checkpoint["trainer_state"],
            environments=environments,
            update=start_update,
            rollout_batch_size=rollout_batch_size,
            recurrent_input_dim=(
                model.observation_dim if isinstance(model, RECURRENT_BASELINE_TYPES) else None
            ),
            device=device,
            metric_fields=metric_fields,
        )
        if isinstance(model, RECURRENT_BASELINE_TYPES):
            # The committed segment ended with a reconditioned hidden state, so
            # the resumed one replays the retained prefixes through the very same
            # helper instead of restoring a separately stored tensor.
            hidden = _recondition_recurrent_hidden(
                model,
                recurrent_histories,
                input_template=_current_inputs(
                    method,
                    observation,
                    batch,
                    previous_action,
                    device,
                    model=model,
                ),
                start_template=episode_starts,
            )
        controller.restore(checkpoint["population"])
        controller.truncate(start_update)
        _restore_global_rng_states(checkpoint, device=device)
        LOGGER.info(
            "Resuming method=%s task=%s seed=%d from update=%d step=%d",
            method,
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
            artifacts.csv_appender(metric_fields).append_many(metric_rows)
    segment_started = time.perf_counter()

    def current_checkpoint_payload(update: int, global_step: int) -> dict[str, Any]:
        wall_time_seconds = accumulated_wall_time + (time.perf_counter() - segment_started)
        trainer_state = _baseline_trainer_state_payload(
            environments=environments,
            observation=observation,
            previous_action=previous_action,
            episode_starts=episode_starts,
            recurrent_histories=recurrent_histories,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            completed_returns=completed_returns,
            completed_lengths=completed_lengths,
            metric_rows=metric_rows,
            best_evaluation=best_evaluation,
            latest_evaluation=latest_evaluation,
            wall_time_seconds=wall_time_seconds,
        )
        payload = _checkpoint_payload(
            model,  # type: ignore[arg-type]
            optimizer,
            task=task,
            seed=seed,
            update=update,
            global_step=global_step,
            config=resolved,
            trainer_state=trainer_state,
            method=method,
            format_version=BASELINE_CHECKPOINT_FORMAT_VERSION,
        )
        if imitation is not None:
            payload["self_imitation"] = imitation.state_dict()
        payload["population"] = controller.state_dict()
        return payload

    for update in range(start_update + 1, total_updates + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_started = time.perf_counter()
        phase = prediction_schedule.phase(update)
        controller.begin_update(update, phase)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        input_steps: list[torch.Tensor] = []
        action_steps: list[torch.Tensor] = []
        log_prob_steps: list[torch.Tensor] = []
        value_steps: list[torch.Tensor] = []
        reward_steps: list[torch.Tensor] = []
        done_steps: list[torch.Tensor] = []
        truncation_value_steps: list[torch.Tensor] = []
        truncation_bootstrap_count = 0
        episode_start_steps: list[torch.Tensor] = []
        pre_tanh_steps: list[torch.Tensor] = []
        diffusion_chain_steps: list[torch.Tensor] = []
        update_returns: list[float] = []
        update_lengths: list[int] = []
        action_saturation_fractions: list[float] = []
        episode_rows: list[dict[str, Any]] = []
        model.eval()
        recurrent_prefixes: tuple[RecurrentReplayPrefix, ...] = ()
        if isinstance(model, RECURRENT_BASELINE_TYPES):
            prefix_input_template = _current_inputs(
                method,
                observation,
                batch,
                previous_action,
                device,
                model=model,
            )
            recurrent_prefixes = tuple(
                history.snapshot(
                    input_template=prefix_input_template,
                    start_template=episode_starts,
                )
                for history in recurrent_histories
            )

        for rollout_index in range(int(ppo_config["rollout_steps"])):
            controller.start_step(episode_starts)
            condition_sink = [] if controller.rnd is not None or controller.f_enabled else None
            inputs = _current_inputs(method, observation, batch, previous_action, device, model=model)
            input_steps.append(inputs.detach())
            episode_start_steps.append(episode_starts.detach())
            with torch.no_grad():
                if isinstance(model, RECURRENT_BASELINE_TYPES):
                    assert hidden is not None
                    sample, old_value, hidden = model.act_step(
                        inputs,
                        hidden,
                        episode_starts,
                        deterministic=False,
                        **({"condition_sink": condition_sink} if condition_sink is not None else {}),
                    )
                else:
                    sample, old_value = model.act(
                        inputs, deterministic=False,
                        **({"condition_sink": condition_sink} if condition_sink is not None else {}),
                    )
            if controller.f_enabled:
                with torch.no_grad():
                    old_value = controller.values(condition_sink[0], controller.ids, old_value)
            if phase == "pretrain":
                # Uniformly random exploration; the stored policy outputs are not read by the
                # prediction objective.
                uniform = random_actions(first_environment.action_spec, batch.size, device=device)
                sample = replace(sample, action=torch.where((controller.ids == 0)[:, None], sample.action, uniform)
                                 if controller.f_enabled else uniform)
            if first_environment.action_spec.is_continuous:
                if sample.diffusion_chain is not None:
                    diffusion_chain_steps.append(sample.diffusion_chain.detach())
                elif sample.pre_tanh_action is not None:
                    pre_tanh_steps.append(sample.pre_tanh_action.detach())
                else:
                    raise RuntimeError(
                        "continuous baseline did not expose its stochastic policy path"
                    )
            sample = replace(sample, action=controller.perturb_action(sample.action, sample.diffusion_chain, episode_starts, phase))
            # Controller uses the light path's (policy, critic) condition sink shape.
            pop_sink = [(condition_sink[0], condition_sink[0])] if condition_sink is not None else None
            controller.save_teacher(pop_sink, sample.action)
            environment_action = _environment_actions(first_environment.action_spec, sample.action)
            next_observation, reward, done, step_infos = batch.step(environment_action)
            controller.observe(sample.action, reward, done, phase)
            done_tensor = torch.as_tensor(done, dtype=torch.bool, device=device)
            # The encoded action is needed before it is zeroed for the finished
            # environments: a truncation bootstrap conditions the discarded final
            # observation on the action that produced it.
            action_features = encode_action(
                first_environment.action_spec, sample.action
            ).detach()
            if bootstrap_on_truncation:
                truncation_value_steps.append(
                    _truncation_bootstrap_values(
                        model,
                        method,
                        value_transform=value_transform,
                        infos=step_infos,
                        hidden=hidden,
                        action_features=action_features,
                        device=device,
                    )
                )
                truncation_bootstrap_count += len(truncated_step_indices(step_infos))
            if imitation is not None and condition_sink is not None:
                imitation.observe(
                    condition_sink[0], condition_sink[0], sample.action,
                    torch.as_tensor(reward, dtype=torch.float32, device=device), done_tensor,
                    chains=sample.diffusion_chain, pre_tanh=sample.pre_tanh_action,
                    bootstrap=truncation_value_steps[-1] if bootstrap_on_truncation else None,
                )
            action_steps.append(sample.action.detach())
            log_prob_steps.append(sample.log_prob.detach())
            value_steps.append(value_transform.to_raw(old_value.detach()))
            reward_steps.append(controller.reward(
                torch.as_tensor(reward, dtype=torch.float32, device=device),
                inputs[:, :first_environment.observation_dim], pop_sink, phase))
            done_steps.append(done_tensor)

            if isinstance(model, RECURRENT_BASELINE_TYPES):
                for environment_index, history in enumerate(recurrent_histories):
                    history.append(inputs, episode_starts, environment_index)
                    if bool(done[environment_index]):
                        history.clear()

            previous_action = action_features
            if first_environment.action_spec.is_continuous:
                action_saturation_fractions.append(
                    float((previous_action.abs() >= 0.99).float().mean().cpu())
                )
            previous_action = previous_action.clone()
            previous_action[done_tensor] = 0.0
            observation = next_observation
            episode_starts = done_tensor
            episode_returns += reward
            episode_lengths += 1

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

        # One durable CSV append per update, rather than one write per episode.
        episode_appender.append_many(episode_rows)
        if bootstrap_on_truncation:
            # Logged rather than added to metrics.csv: METRIC_FIELDS is a closed
            # CSV schema shared by every run and reader, and making it depend on
            # a config flag would fork the artifact contract for one diagnostic.
            truncation_sum = float(torch.stack(truncation_value_steps).sum().cpu())
            LOGGER.info(
                "truncation_bootstrap update=%d truncations=%d value_mean=%s",
                update,
                truncation_bootstrap_count,
                (
                    f"{truncation_sum / truncation_bootstrap_count:.6f}"
                    if truncation_bootstrap_count
                    else "n/a"
                ),
            )
        with torch.no_grad():
            bootstrap_inputs = _current_inputs(method, observation, batch, previous_action, device, model=model)
            if isinstance(model, RECURRENT_BASELINE_TYPES):
                assert hidden is not None
                bootstrap_features, _ = model.step(bootstrap_inputs, hidden, episode_starts)
                last_value = value_transform.to_raw(
                    controller.values(model.control_features(bootstrap_features, bootstrap_inputs), controller.ids)
                )
            else:
                last_value = value_transform.to_raw(model.value(bootstrap_inputs))

        rewards = torch.stack(reward_steps)
        dones = torch.stack(done_steps)
        old_values = torch.stack(value_steps)
        advantages, returns = generalised_advantage_estimate(
            rewards,
            dones,
            old_values,
            last_value.detach(),
            gamma=float(ppo_config["gamma"]),
            gae_lambda=float(ppo_config["gae_lambda"]),
            truncation_values=(
                torch.stack(truncation_value_steps) if bootstrap_on_truncation else None
            ),
        )
        rollout = BaselineRollout(
            inputs=torch.stack(input_steps),
            actions=torch.stack(action_steps),
            old_log_prob=torch.stack(log_prob_steps),
            old_values=old_values,
            rewards=rewards,
            dones=dones,
            advantages=advantages,
            returns=returns,
            episode_starts=torch.stack(episode_start_steps),
            pre_tanh_actions=(torch.stack(pre_tanh_steps) if pre_tanh_steps else None),
            diffusion_chains=(
                torch.stack(diffusion_chain_steps) if diffusion_chain_steps else None
            ),
            hidden_states=None,
            recurrent_prefixes=recurrent_prefixes,
        )
        controller.before_learning()
        model.train()
        if isinstance(model, RECURRENT_BASELINE_TYPES):
            temporal_gradient_mode: TemporalGradientMode = model_config["belief_gradient_mode"]
            if phase in {"ppo", "critic"}:
                with critic_only(model, phase == "critic"):
                    ppo_metrics = _update_recurrent(
                        model,
                        optimizer,
                        rollout,
                        model_config,
                        ppo_config,
                        temporal_gradient_mode=temporal_gradient_mode,
                        observation_dim=first_environment.observation_dim,
                        action_feature_dim=first_environment.action_spec.feature_dim,
                    )
            else:
                # Population Adam is separate from the shared optimizer. Clear
                # its previous PPO gradients before prediction-only clipping.
                controller.zero_actor_grad()
                ppo_metrics = _update_recurrent_prediction(
                    model,
                    optimizer,
                    rollout,
                    ppo_config,
                    observation_dim=first_environment.observation_dim,
                    action_feature_dim=first_environment.action_spec.feature_dim,
                    temporal_gradient_mode=temporal_gradient_mode,
                )
                LOGGER.info(
                    "prediction phase=%s update=%d next_observation_nll=%.4f",
                    phase,
                    update,
                    ppo_metrics.policy_loss,
                )
            recondition_inputs = _current_inputs(
                method,
                observation,
                batch,
                previous_action,
                device,
                model=model,
            )
            hidden = _recondition_recurrent_hidden(
                model,
                recurrent_histories,
                input_template=recondition_inputs,
                start_template=episode_starts,
            )
        else:
            with critic_only(model, phase == "critic"):
                ppo_metrics = _update_feedforward(model, optimizer, rollout, ppo_config)

        imitation_metrics = None
        if imitation is not None:
            imitation_metrics = (
                update_self_imitation(model, optimizer, imitation, ppo_config)
                if phase == "ppo" else {name: 0.0 for name in SIL_METRIC_FIELDS}
            )
            imitation_metrics["sil_buffer_size"] = float(imitation.size)

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
            "score_l2_mean": float("nan"),
            "particle_abs_max": float("nan"),
            "particle_clip_fraction": float("nan"),
            "initial_score_l2_mean": float("nan"),
            "recursive_score_l2_mean": float("nan"),
            "initial_particle_l2_mean": float("nan"),
            "recursive_particle_l2_mean": float("nan"),
            "initial_belief_particle_count": float("nan"),
            "recursive_belief_particle_count": float("nan"),
            "score_nonfinite_count": float("nan"),
            "particle_nonfinite_count": float("nan"),
            "action_saturation_fraction": (
                float(np.mean(action_saturation_fractions))
                if action_saturation_fractions
                else float("nan")
            ),
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
            "steps_per_second": rollout_batch_size / duration,
        }
        if imitation_metrics is not None:
            row.update(imitation_metrics)
        row.update(controller.finish_update(rollout, update_returns))
        metric_rows.append(row)
        artifacts.append_metrics(row, metric_fields)
        if update % int(experiment_config["log_interval_updates"]) == 0:
            LOGGER.info(
                "method=%s update=%d/%d step=%d policy=%.4f value=%.4f return=%s fps=%.1f",
                method,
                update,
                total_updates,
                global_step,
                ppo_metrics.policy_loss,
                ppo_metrics.value_loss,
                "n/a" if not update_returns else f"{np.mean(update_returns):.3f}",
                row["steps_per_second"],
            )

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
            controller.evaluate_members(update, global_step)
            inference_model = controller.inference_model()
            evaluation_seed = seed + 1_000_000
            deterministic_mode = (
                "greedy"
                if first_environment.action_spec.is_discrete
                else "mean_chain"
                if isinstance(model, RECURRENT_BASELINE_TYPES)
                and model.continuous_policy_kind == "diffusion"
                else "mean_action"
            )
            for mode, deterministic_evaluation in (
                (deterministic_mode, True),
                ("stochastic", False),
            ):
                evaluation, trajectories = evaluate_baseline(
                    inference_model,
                    method=method,
                    task=task,
                    environment_config=environment_config,
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
                    "evaluation method=%s mode=%s update=%d return=%.3f +/- %.3f",
                    method,
                    mode,
                    update,
                    evaluation["mean_return"],
                    evaluation["std_return"],
                )
                if deterministic_evaluation:
                    latest_evaluation = evaluation

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

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        CsvAppender(artifacts.path / "timing.csv", ("update", "phase", "seconds", "evaluated", "selection_replaced")).append_many([
            dict(update=update, phase=phase, seconds=time.perf_counter()-update_started,
                 evaluated=int(should_evaluate), selection_replaced=row["selection_replaced"])])

    numeric_fields = [field for field in metric_fields if field not in {"update", "global_step"}]
    arrays: dict[str, np.ndarray] = {
        "update": np.asarray([row["update"] for row in metric_rows], dtype=np.int64),
        "global_step": np.asarray([row["global_step"] for row in metric_rows], dtype=np.int64),
        "episode_returns": np.asarray(completed_returns, dtype=np.float32),
        "episode_lengths": np.asarray(completed_lengths, dtype=np.int32),
    }
    for field in numeric_fields:
        arrays[field] = np.asarray([row[field] for row in metric_rows], dtype=np.float64)
    artifacts.save_npz(arrays)

    # Identical to the score trainer: ``latest_evaluation`` is the deterministic
    # readout, so the legacy key and its explicit alias hold the same number.
    readout_return_mean = None if latest_evaluation is None else latest_evaluation["mean_return"]
    summary = {
        "task": task,
        "seed": seed,
        # Identical to the score trainer's rule: reaching ``configured_updates``
        # is what forces the final evaluation above, so an intermediate segment
        # stays 'partial' and never reaches the completed-only aggregation.
        "status": "completed" if total_updates == configured_updates else "partial",
        "updates": total_updates,
        "global_steps": total_updates * rollout_batch_size,
        "episodes_completed": len(completed_returns),
        "training_return_mean": (float(np.mean(completed_returns)) if completed_returns else None),
        "evaluation_return_mean": readout_return_mean,
        "deterministic_readout_return_mean": readout_return_mean,
        "parameter_count": parameter_count,
        "wall_time_seconds": accumulated_wall_time + (time.perf_counter() - segment_started),
        "result_dir": result_dir,
    }
    LOGGER.info(
        "%s method=%s task=%s seed=%d at update=%d/%d",
        "Completed" if total_updates == configured_updates else "Paused",
        method,
        task,
        seed,
        total_updates,
        configured_updates,
    )
    return summary
