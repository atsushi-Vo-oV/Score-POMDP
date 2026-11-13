"""Sequence PPO fork: original replay/objectives; actor routing is local.

Derived from sb_pomdp.ppo.update_ppo. Existing trainer is unmodified.
"""
from __future__ import annotations
from ..ppo import (
    CurlReplaySamples,
    EnergyBelief,
    LOGGER,
    PPOMetrics,
    ValueTransform,
    _module_gradient_norm,
    _time_environment_flatten,
    _wml_policy_objective,
    action_value_advantages,
    belief_mixture_prediction_loss,
    clip_gradients_or_skip,
    condition_prediction_loss,
    control_conditions,
    curl_enabled,
    curl_generator,
    encode_conditions,
    observation_prediction_loss,
    replay_belief_sequence,
    reward_prediction_loss,
    torch,
    value_from_conditions
)
from .objective import policy_objective as _policy_objective

def update_ppo(
    model: ScoreBeliefActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    model_config: dict,
    ppo_config: dict,
    *,
    generator: torch.Generator | None = None,
    objective: str = "ppo",
    score_curl_coef: float | None = None,
    score_curl_seed: int = 0,
    score_curl_update: int = 0,
) -> PPOMetrics:
    """Update all model components through ULA and the recurrent belief.

    ``objective="prediction"`` replays the same belief sequences but optimises only the
    next-observation prediction loss of ``model.prediction_predictor`` (the interleaved
    prediction phases of :mod:`sb_pomdp.prediction_schedule`); its negative log-likelihood is
    reported as ``policy_loss`` and the PPO quantities are zero.

    Environment trajectories are the fixed on-policy samples required by PPO.
    Only the differentiable, model-free belief computation is reparameterized
    and replayed.  Sequence minibatches are required because randomly shuffled
    time steps would sever the recurrent gradient.
    """

    if objective not in {"ppo", "prediction", "critic"}:
        raise ValueError("objective must be ppo, prediction or critic")
    if objective == "prediction" and getattr(model, "prediction_predictor", None) is None:
        raise ValueError("the prediction objective needs model.prediction_predictor")
    use_curl = curl_enabled(model_config)
    if use_curl and (
        not isinstance(model.belief, EnergyBelief)
        or model.belief.score_parameterization == "energy"
    ):
        raise ValueError("score curl requires a direct score belief (not energy or PF)")
    if use_curl and score_curl_coef is None and model_config.get("score_curl_schedule") == "ramp":
        raise ValueError("ramp curl requires score_curl_coef from the PPO progress schedule")
    curl_coef = (
        float(model_config.get("score_curl_penalty", 0.0))
        if score_curl_coef is None else float(score_curl_coef)
    )
    if objective == "critic":
        curl_coef = 0.0
    curl_rng = curl_generator(score_curl_seed, score_curl_update) if use_curl else None
    rollout_steps, num_environments = rollout.rollout_shape
    configured_minibatch_size = int(ppo_config["minibatch_size"])
    sequence_microbatch_size = int(
        ppo_config.get("sequence_microbatch_size", configured_minibatch_size)
    )
    if configured_minibatch_size < rollout_steps:
        raise ValueError(
            "ppo.minibatch_size must be at least ppo.rollout_steps for belief sequence replay"
        )
    if configured_minibatch_size % rollout_steps:
        raise ValueError(
            "ppo.minibatch_size must be divisible by ppo.rollout_steps for belief sequence replay"
        )
    if sequence_microbatch_size < rollout_steps:
        raise ValueError("ppo.sequence_microbatch_size must be at least ppo.rollout_steps")
    if sequence_microbatch_size > configured_minibatch_size:
        raise ValueError("ppo.sequence_microbatch_size must not exceed ppo.minibatch_size")
    if sequence_microbatch_size % rollout_steps:
        raise ValueError("ppo.sequence_microbatch_size must be divisible by ppo.rollout_steps")
    if configured_minibatch_size % sequence_microbatch_size:
        raise ValueError("ppo.minibatch_size must be divisible by ppo.sequence_microbatch_size")
    environments_per_minibatch = min(
        configured_minibatch_size // rollout_steps,
        num_environments,
    )
    environments_per_microbatch = sequence_microbatch_size // rollout_steps

    algorithm = str(ppo_config.get("algorithm", "ppo"))
    if algorithm not in {"ppo", "p3o_wml"}:
        raise ValueError("ppo.algorithm must be ppo or p3o_wml")
    policy_objective = _policy_objective if algorithm == "ppo" else _wml_policy_objective
    if algorithm == "p3o_wml" and rollout.wml_weights is None:
        raise ValueError("ppo.algorithm=p3o_wml requires a tilted rollout (wml_weights)")
    value_transform = ValueTransform.from_config(ppo_config)
    advantages = rollout.advantages
    if ppo_config["normalize_advantage"]:
        if model._popx.f_enabled:
            # Intrinsic advantages from explorers must not change F's baseline/scale.
            advantages = advantages.clone()
            for mask in (model._popx.ids == 0, model._popx.ids != 0):
                subset = advantages[:, mask]
                advantages[:, mask] = (subset - subset.mean()) / (subset.std(unbiased=False) + 1e-8)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approximate_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
        "initial_energy_gradient_norm": 0.0,
        "observation_energy_gradient_norm": 0.0,
        "transition_energy_gradient_norm": 0.0,
    }
    batches = 0
    if use_curl:
        totals.update(score_curl_loss=0.0, score_curl_ratio=0.0, score_curl_coef=0.0)
    skipped_minibatches = 0

    for _ in range(int(ppo_config["epochs"])):
        permutation = torch.randperm(
            num_environments,
            device=advantages.device,
            generator=generator,
        )
        for start in range(0, num_environments, environments_per_minibatch):
            minibatch_indices = permutation[start : start + environments_per_minibatch]
            minibatch_environments = int(minibatch_indices.numel())
            minibatch_metrics = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approximate_kl": 0.0,
                "clip_fraction": 0.0,
            }
            optimizer.zero_grad(set_to_none=True)
            model._popx.zero_actor_grad()
            curl_samples = (
                CurlReplaySamples(rollout, minibatch_indices, model_config, curl_rng)
                if use_curl else None
            )

            # Environment trajectories are independent.  Splitting only this
            # dimension lowers peak memory without truncating time or changing
            # the effective PPO minibatch/optimizer update.
            for micro_start in range(0, minibatch_environments, environments_per_microbatch):
                environment_indices = minibatch_indices[
                    micro_start : micro_start + environments_per_microbatch
                ]
                microbatch_weight = int(environment_indices.numel()) / minibatch_environments
                replay_kwargs = {} if curl_samples is None else {"curl_samples": curl_samples}
                particles, scores = replay_belief_sequence(
                    model,
                    rollout,
                    environment_indices,
                    model_config,
                    **replay_kwargs,
                )
                condition, value_condition = encode_conditions(
                    model,
                    _time_environment_flatten(particles),
                    _time_environment_flatten(scores),
                )
                # The prediction head reads the bare belief conditions; the actor and critic
                # read the control inputs (the same, or with o_t appended).
                policy_input, value_input = control_conditions(
                    model,
                    condition,
                    value_condition,
                    _time_environment_flatten(rollout.observations[:, environment_indices]),
                )
                if objective == "critic":
                    # Warm-up trains ONLY V: no actor, encoder, belief or auxiliary
                    # gradients (nor Adam/AdamW steps on their parameters).
                    if getattr(model, "action_value_head", None) is not None:
                        raise ValueError("critic warmup requires a state-value head")
                    new_values = model._popx.values(value_input.detach(), model._popx.ids[environment_indices].repeat(rollout_steps))
                    returns = _time_environment_flatten(rollout.returns[:, environment_indices])
                    value_loss = 0.5 * (new_values - value_transform.to_target(returns)).square().mean()
                    (float(ppo_config["value_coef"]) * value_loss * microbatch_weight).backward()
                    minibatch_metrics["value_loss"] += float(value_loss.detach()) * microbatch_weight
                    del particles, scores, condition, value_condition, new_values, returns, value_loss
                    continue
                prediction_only = (
                    str(model_config.get("belief_training", "joint")) == "prediction_only"
                )
                prediction_loss = None
                if objective == "prediction" or prediction_only:
                    slice_kwargs = {
                        "observations": rollout.observations[:, environment_indices],
                        "previous_actions": rollout.previous_actions[:, environment_indices],
                        "dones": rollout.dones[:, environment_indices],
                    }
                    if getattr(model, "prediction_head", "condition") == "belief_mixture":
                        prediction_loss = belief_mixture_prediction_loss(
                            model.prediction_predictor, particles, **slice_kwargs
                        )
                    else:
                        prediction_loss = condition_prediction_loss(
                            model.prediction_predictor, condition, **slice_kwargs
                        )
                if objective == "prediction":
                    (prediction_loss * microbatch_weight).backward()
                    minibatch_metrics["policy_loss"] += (
                        float(prediction_loss.detach().cpu()) * microbatch_weight
                    )
                    del particles, scores, condition, value_condition, prediction_loss
                    continue
                if prediction_only:
                    # belief_training=prediction_only: the belief and its readout learn from the
                    # next-observation loss alone (added to the total loss below); PPO reaches the
                    # actor / critic heads through a stop-gradient on their inputs.
                    if value_input is policy_input:
                        policy_input = value_input = policy_input.detach()
                    else:
                        policy_input, value_input = policy_input.detach(), value_input.detach()
                if getattr(model, "action_value_head", None) is not None:
                    taken = _time_environment_flatten(rollout.actions[:, environment_indices])
                    new_values = model.action_value(value_input, taken)
                    minibatch_advantages = action_value_advantages(
                        model,
                        policy_input,
                        new_values,
                        advantages,
                        environment_indices,
                        value_transform,
                        ppo_config,
                        critic_condition=value_input,
                    )
                else:
                    new_values = value_from_conditions(model, policy_input, value_input)
                    if model._popx.f_enabled:
                        new_values = model._popx.values(value_input, model._popx.ids[environment_indices].repeat(rollout_steps), new_values)
                    minibatch_advantages = advantages
                policy_loss, approximate_kl, clip_fraction, entropy = policy_objective(
                    model,
                    policy_input,
                    rollout,
                    environment_indices,
                    minibatch_advantages,
                    model_config,
                    ppo_config,
                )
                returns = _time_environment_flatten(rollout.returns[:, environment_indices])
                value_loss = 0.5 * (new_values - value_transform.to_target(returns)).square().mean()
                total_loss = (
                    policy_loss
                    + float(ppo_config["value_coef"]) * value_loss
                    - float(ppo_config["entropy_coef"]) * entropy
                )
                if prediction_loss is not None:
                    total_loss = total_loss + prediction_loss
                if getattr(model, "observation_predictor", None) is not None:
                    prediction_loss = observation_prediction_loss(
                        model,
                        particles,
                        rollout,
                        environment_indices,
                    )
                    total_loss = (
                        total_loss
                        + float(model.observation_prediction_coef) * prediction_loss
                    )
                if getattr(model, "reward_predictor", None) is not None:
                    total_loss = total_loss + float(
                        model.reward_prediction_coef
                    ) * reward_prediction_loss(model, particles, rollout, environment_indices)
                (total_loss * microbatch_weight).backward()

                microbatch_values = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                }
                for name, value in microbatch_values.items():
                    minibatch_metrics[name] += float(value.detach().cpu()) * microbatch_weight

                # Do not retain the last microbatch graph while constructing
                # the next one; backward has already accumulated its gradients.
                del (
                    particles,
                    scores,
                    condition,
                    new_values,
                    policy_loss,
                    approximate_kl,
                    clip_fraction,
                    entropy,
                    returns,
                    value_loss,
                    total_loss,
                    microbatch_values,
                )

            if curl_samples is not None:
                curl_loss, curl_ratio = curl_samples.evaluate(model.belief, create_graph=curl_coef > 0)
                if curl_coef > 0 and curl_loss.requires_grad:
                    (curl_coef * curl_loss).backward()
                minibatch_metrics.update(
                    score_curl_loss=float(curl_loss.detach()),
                    score_curl_ratio=float(curl_ratio), score_curl_coef=curl_coef,
                )
                del curl_loss, curl_ratio, curl_samples

            energy_gradient_norms = {
                "initial_energy_gradient_norm": _module_gradient_norm(
                    getattr(model.belief, "initial_energy", None),
                    reference=model,
                ),
                "observation_energy_gradient_norm": _module_gradient_norm(
                    getattr(model.belief, "observation_energy", None),
                    reference=model,
                ),
                "transition_energy_gradient_norm": _module_gradient_norm(
                    getattr(model.belief, "transition_energy", None),
                    reference=model,
                ),
            }
            gradient_norm, finite = clip_gradients_or_skip(
                model.parameters(), float(ppo_config["max_grad_norm"])
            )
            if not finite:
                skipped_minibatches += 1
                LOGGER.warning(
                    "non-finite gradient norm (%s); skipping the optimizer step for this "
                    "minibatch (%d skipped in this update)",
                    float(gradient_norm),
                    skipped_minibatches,
                )
                continue
            model._popx.suppress_disabled_singleton()
            optimizer.step()
            model._popx.step_actors()

            for name, value in minibatch_metrics.items():
                totals[name] += value
            totals["gradient_norm"] += float(gradient_norm.detach().cpu())
            for name, value in energy_gradient_norms.items():
                totals[name] += float(value.detach().cpu())
            batches += 1

    if not batches:
        raise RuntimeError("PPO update produced no minibatches")
    return PPOMetrics(**{name: value / batches for name, value in totals.items()})
