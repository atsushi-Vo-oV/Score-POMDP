"""Batch all GRU actors and complete evaluation episodes; preserve learner RNG."""
import numpy as np
import torch

from ..envs import make_env
from ..train import SyncEnvironmentBatch, _environment_actions, encode_action
from .actors import assignments


def evaluate_population(model, *, task, environment_config, episodes, seed, device, deterministic):
    policy = model.heads.diffusion_policy
    n = getattr(policy, "size", 1)
    count = n * episodes
    ids = assignments(count, n, device)
    seeds = [seed + k for _ in range(n) for k in range(episodes)]
    envs = [make_env(task, environment_config, seed=s) for s in seeds]
    batch = SyncEnvironmentBatch(envs)
    observation = batch.reset(seeds)
    previous_action = torch.zeros(count, envs[0].action_spec.feature_dim, device=device)
    hidden = model.initial_hidden(count, device=device)
    initial = torch.ones(count, dtype=torch.bool, device=device)
    active = np.ones(count, dtype=bool)
    returns, lengths = np.zeros(count), np.zeros(count, dtype=int)
    sums = np.zeros((count, policy.action_dim))
    squares, saturation = np.zeros_like(sums), np.zeros(count)
    training = model.training
    model.eval()
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            for _ in range(int(environment_config["horizon"])):
                inputs = torch.cat((torch.as_tensor(observation, dtype=torch.float32, device=device), previous_action), -1)
                condition, hidden = model.step(inputs, hidden, initial)
                condition = model.control_features(condition, inputs)
                if hasattr(policy, "sample_current"):
                    action, _, _ = policy.sample_current(condition, ids, deterministic=deterministic, generator=generator)
                else:
                    action, _, _ = policy.sample(condition, deterministic=deterministic, generator=generator)
                observation, reward, done, _ = batch.step(_environment_actions(envs[0].action_spec, action))
                normalized = ((action - policy.action_bias) / policy.action_scale).cpu().numpy()
                returns += active * reward
                lengths += active
                sums += active[:, None] * normalized
                squares += active[:, None] * normalized**2
                saturation += active * (np.abs(normalized) >= .99).mean(-1)
                active &= ~done
                initial = torch.as_tensor(done, dtype=torch.bool, device=device)
                previous_action = encode_action(envs[0].action_spec, action).clone()
                previous_action[initial] = 0
                if not active.any():
                    break
    finally:
        model.train(training)
    if active.any():
        raise RuntimeError("GRU population evaluation exceeded the configured episode horizon")
    means = sums / lengths[:, None]
    variances = np.maximum(0, squares / lengths[:, None] - means**2)
    descriptors = np.concatenate((means, variances, lengths[:, None] / environment_config["horizon"]), 1)
    return {"returns": returns.reshape(n, episodes), "lengths": lengths.reshape(n, episodes),
            "descriptors": descriptors.reshape(n, episodes, -1).mean(1),
            "saturation": (saturation / lengths).reshape(n, episodes).mean(1)}
