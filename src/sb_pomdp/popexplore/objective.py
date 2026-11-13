"""Own-actor PPO ratios; no transitions are borrowed by other actors."""
import torch
from .representation import actor
from ..ppo import diffusion_advantage_weights


def policy_objective(model, condition, rollout, environment_indices, advantages, model_config, ppo_config):
    c = model._popx
    steps = rollout.rewards.shape[0]
    ids = c.ids[environment_indices].repeat(steps)
    chain = rollout.diffusion_chains[:, environment_indices].flatten(0, 1)
    old = rollout.old_log_prob[:, environment_indices].flatten(0, 1)
    if c.bank is not None:
        new, entropies = c.bank.log_prob_for(condition, chain, ids)
    else:
        new, entropies = actor(model).log_prob_chain(condition, chain)
    weights = diffusion_advantage_weights(new.shape[1], model_config["diffusion_advantage_discount"],
                                         device=new.device, dtype=new.dtype)
    advantage = advantages[:, environment_indices].flatten(0, 1)[:, None] * weights
    logratio = new - old
    ratio = logratio.exp()
    loss = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1-ppo_config["clip_coef"], 1+ppo_config["clip_coef"]))
    mask = c.actor_enabled[ids, None].to(loss)
    c.active_actors[ids.unique()] |= c.actor_enabled[ids.unique()]
    # Average over ALL rollout rows, as baseline PPO. Each actor has Adam state
    # and receives only its own data; no uncorrected cross-actor reuse.
    if c.f_enabled:
        scale = torch.full_like(mask, c.cfg["actor_ppo_scale"])
        scale[ids == 0] = 1.0
        policy_loss = (loss * mask * scale).mean()
        entropy = (entropies * mask * scale).mean()
    else:
        policy_loss = (loss * mask).mean() * c.cfg["actor_ppo_scale"]
        entropy = (entropies * mask).mean() * c.cfg["actor_ppo_scale"]
    denominator = (mask.sum() * new.shape[1]).clamp_min(1)
    kl = (((ratio - 1) - logratio).detach() * mask).sum() / denominator
    clip = (((ratio - 1).abs() > ppo_config["clip_coef"]).float() * mask).sum() / denominator
    return policy_loss, kl, clip, entropy
