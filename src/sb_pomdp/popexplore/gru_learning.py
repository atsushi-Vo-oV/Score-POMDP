"""GRU sequence PPO with shared representation and own-slot actor objectives.

The all-off route calls the original baseline updater, including its exact Adam
epsilon, reduction order and recurrent replay. Population replay adds only the
actor masks, external F critic and sequence microbatch gradient accumulation.
"""
import torch

from ..ppo import clip_gradients_or_skip, condition_prediction_loss
from ..train_baselines import (_update_recurrent, _normalise_advantages,
                               replay_recurrent_sequence, _empty_metric_totals,
                               _accumulate_metrics, _finish_metrics)
from ..train import final_step_features, truncated_step_indices
from ..value_transform import ValueTransform
from .objective import policy_objective


def update_recurrent(model, optimizer, rollout, model_config, ppo_config, *,
                     temporal_gradient_mode, observation_dim, action_feature_dim):
    c = model._popx
    if c.bank is None and c.cfg["actor_ppo_fraction"] == 1 and c.cfg["actor_ppo_scale"] == 1:
        return _update_recurrent(model, optimizer, rollout, model_config, ppo_config,
                                 temporal_gradient_mode=temporal_gradient_mode,
                                 observation_dim=observation_dim, action_feature_dim=action_feature_dim)
    steps, num_envs = rollout.rollout_shape
    batch_size = int(ppo_config["minibatch_size"])
    micro_size = int(ppo_config.get("sequence_microbatch_size", batch_size))
    if batch_size % steps or micro_size % steps or not steps <= micro_size <= batch_size or batch_size % micro_size:
        raise ValueError("GRU minibatch/microbatch must contain integral sequences and divide one another")
    per_batch, per_micro = min(batch_size // steps, num_envs), micro_size // steps
    advantages = rollout.advantages
    if ppo_config["normalize_advantage"]:
        if c.f_enabled:
            advantages = advantages.clone()
            for mask in (c.ids == 0, c.ids != 0):
                advantages[:, mask] = _normalise_advantages(advantages[:, mask], True)
        else:
            advantages = _normalise_advantages(advantages, True)
    prediction_only = model_config.get("belief_training", "joint") == "prediction_only"
    transform = ValueTransform.from_config(ppo_config)
    totals, batches = _empty_metric_totals(), 0
    for _ in range(int(ppo_config["epochs"])):
        permutation = torch.randperm(num_envs, device=rollout.inputs.device)
        for start in range(0, num_envs, per_batch):
            selected = permutation[start:start + per_batch]
            optimizer.zero_grad(set_to_none=True)
            c.zero_actor_grad()
            metrics = torch.zeros(5, device=rollout.inputs.device)
            for micro_start in range(0, len(selected), per_micro):
                indices = selected[micro_start:micro_start + per_micro]
                weight = len(indices) / len(selected)
                features = replay_recurrent_sequence(model, rollout, indices,
                                                      temporal_gradient_mode=temporal_gradient_mode)
                flat = features.flatten(0, 1)
                prediction_loss = None
                if prediction_only:
                    prediction_loss = condition_prediction_loss(
                        model.prediction_predictor, flat,
                        observations=rollout.inputs[:, indices, :observation_dim],
                        previous_actions=rollout.inputs[:, indices, observation_dim:observation_dim + action_feature_dim],
                        dones=rollout.dones[:, indices])
                    flat = flat.detach()
                control = model.control_features(flat, rollout.inputs[:, indices].flatten(0, 1))
                policy, kl, clip, entropy = policy_objective(
                    model, control, rollout, indices, advantages, model_config, ppo_config)
                values = c.values(control, c.ids[indices].repeat(steps))
                targets = transform.to_target(rollout.returns[:, indices].flatten())
                value_loss = 0.5 * (values - targets).square().mean()
                loss = policy + float(ppo_config["value_coef"]) * value_loss - float(ppo_config["entropy_coef"]) * entropy
                if prediction_loss is not None:
                    loss = loss + prediction_loss
                (weight * loss).backward()
                metrics += weight * torch.stack([policy, value_loss, entropy, kl, clip]).detach()
            c.suppress_disabled_singleton()
            gradient_norm, finite = clip_gradients_or_skip(model.parameters(), float(ppo_config["max_grad_norm"]))
            if not finite:
                continue
            optimizer.step()
            c.step_actors()
            _accumulate_metrics(totals, policy_loss=metrics[0], value_loss=metrics[1], entropy=metrics[2],
                                approximate_kl=metrics[3], clip_fraction=metrics[4], gradient_norm=gradient_norm)
            batches += 1
    return _finish_metrics(totals, batches)


def truncation_values(model, method, *, infos, hidden, action_features, device, value_transform):
    """Bootstrap F with its external critic, including discarded time-limit states."""
    from ..train_baselines import _truncation_bootstrap_values
    c = model._popx
    if not c.f_enabled:
        return _truncation_bootstrap_values(model, method, infos=infos, hidden=hidden,
            action_features=action_features, device=device, value_transform=value_transform)
    values = torch.zeros(len(infos), dtype=torch.float32, device=device)
    indices = truncated_step_indices(infos)
    if indices:
        selector = torch.as_tensor(indices, dtype=torch.long, device=device)
        inputs = torch.cat((final_step_features(infos, indices, name="final_observation", device=device),
                            action_features.detach().index_select(0, selector)), -1)
        with torch.no_grad():
            features, _ = model.step(inputs, hidden.detach().index_select(1, selector).clone(),
                                     torch.zeros(len(indices), dtype=torch.bool, device=device))
            bootstrap = value_transform.to_raw(c.values(model.control_features(features, inputs), c.ids[selector]))
        values.index_copy_(0, selector, bootstrap.detach().to(values))
    return values
