"""Complete-episode, batched population evaluation on external rewards only."""
import numpy as np
import torch

from ..envs import make_env
from ..ppo import encode_conditions
from ..train import (BeliefContext, SyncEnvironmentBatch, _sample_belief, _environment_actions,
                     _next_context, encode_action, belief_latent_dim)
from .actors import assignments
from .representation import actor, is_gru


def evaluate_population(model, *, task, environment_config, model_config, episodes, seed, device, deterministic):
    if is_gru(model):
        from .gru_evaluation import evaluate_population as evaluate_gru
        return evaluate_gru(model, task=task, environment_config=environment_config, episodes=episodes,
                            seed=seed, device=device, deterministic=deterministic)
    policy = actor(model)
    n = getattr(policy, "size", 1)
    count = n * episodes
    ids = assignments(count, n, device)
    seeds = [seed + k for _ in range(n) for k in range(episodes)]
    envs = [make_env(task, environment_config, seed=s) for s in seeds]
    batch = SyncEnvironmentBatch(envs)
    context = BeliefContext(
        torch.as_tensor(batch.reset(seeds), dtype=torch.float32, device=device),
        torch.zeros(count, model_config["num_particles"], belief_latent_dim(envs[0], model_config), device=device),
        torch.zeros(count, envs[0].action_spec.feature_dim, device=device),
        torch.ones(count, dtype=torch.bool, device=device),
    )
    active = np.ones(count, dtype=bool)
    returns, lengths = np.zeros(count), np.zeros(count, dtype=int)
    sums = np.zeros((count, policy.action_dim))
    squares = np.zeros_like(sums)
    saturation = np.zeros(count)
    training = model.training
    model.eval()
    # Neither fitness nor diagnostics may advance the learner's RNG stream.
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            for _ in range(int(environment_config["horizon"])):
                particles, scores = _sample_belief(model, context, model_config, generator=generator)
                condition, _ = encode_conditions(model, particles, scores)
                if hasattr(policy, "sample_current"):
                    action, _, _ = policy.sample_current(condition, ids, deterministic=deterministic, generator=generator)
                else:
                    action, _, _ = policy.sample(condition, deterministic=deterministic, generator=generator)
                features = encode_action(envs[0].action_spec, action)
                actions = _environment_actions(envs[0].action_spec, action)
                obs, reward, done, _ = batch.step(actions)
                normalized = ((action - policy.action_bias) / policy.action_scale).cpu().numpy()
                returns += active * reward
                lengths += active
                sums += active[:, None] * normalized
                squares += active[:, None] * normalized**2
                saturation += active * (np.abs(normalized) >= .99).mean(-1)
                active &= ~done
                context = _next_context(observation=obs, particles=particles, action_features=features,
                                        done=done, device=device)
                if not active.any():
                    break
    finally:
        model.train(training)
    if active.any():
        raise RuntimeError("population evaluation exceeded the configured episode horizon")
    means = sums / lengths[:, None]
    variances = np.maximum(0, squares / lengths[:, None] - means**2)
    descriptors = np.concatenate((means, variances, lengths[:, None] / environment_config["horizon"]), 1)
    return {"returns": returns.reshape(n, episodes), "lengths": lengths.reshape(n, episodes),
            "descriptors": descriptors.reshape(n, episodes, -1).mean(1),
            "saturation": (saturation / lengths).reshape(n, episodes).mean(1)}


def evaluate_batched(model, **arguments):
    """Single-actor evaluation with the (summary, trajectories) shape of ``train.evaluate``.

    All episodes advance together, so the cost is one horizon of belief updates
    instead of ``episodes`` horizons.  Random numbers are assigned differently
    from the sequential ``train.evaluate``; compare its values only with other
    values from this function.  No trajectories are recorded.
    """
    result = evaluate_population(model, **arguments)
    if result["returns"].shape[0] != 1:
        raise ValueError("evaluate_batched expects an ordinary single-actor model")
    returns, lengths = result["returns"][0], result["lengths"][0]
    summary = {"mean_return": float(returns.mean()), "std_return": float(returns.std()),
               "mean_length": float(lengths.mean()),
               "action_saturation_fraction": float(result["saturation"][0])}
    return summary, {}
