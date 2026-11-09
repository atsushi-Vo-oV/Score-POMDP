"""Functional seed batching of belief transport and PPO/prediction replay.

Time and transport steps remain sequential. The leading dimension of every
parameter and input is an independent seed; vmap never reduces that dimension.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import fields

import torch
from torch import nn
from torch.distributions import Distribution
from torch.func import functional_call, stack_module_state, vmap

from ..buffer import RolloutBatch
from ..ppo import (
    PPOMetrics, _module_gradient_norm, _policy_objective, clip_gradients_or_skip,
    condition_prediction_loss, encode_conditions, value_from_conditions,
)
from ..value_transform import ValueTransform
from .attention import use_batched_attention


@contextmanager
def functional_kernels():
    # Distribution validation and the fused inference-only attention fast path
    # contain tensor-to-bool operations unsupported by torch 2.3 vmap. Restore
    # both settings before executing any ordinary model/evaluation code.
    validation = Distribution._validate_args
    attention = torch.backends.mha.get_fastpath_enabled()
    Distribution.set_default_validate_args(False)
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        Distribution.set_default_validate_args(validation)
        torch.backends.mha.set_fastpath_enabled(attention)


def branch_for(initial, active=None):
    values = initial if active is None else initial[active]
    if bool(values.all()):
        return "initial"
    if not bool(values.any()):
        return "recursive"
    return "mixed"


def branch_rows(initial, active=None):
    """Classify an entire [seed,time,environment] grid in one host transfer."""
    if active is None:
        active = torch.ones_like(initial)
    counts = torch.stack(((initial & active).sum(dim=(0, 2)), active.sum(dim=(0, 2))), dim=1)
    return tuple("initial" if yes == total else "recursive" if yes == 0 else "mixed"
                 for yes, total in counts.cpu().tolist())


class FunctionalModel(nn.Module):
    def __init__(self, model, model_config, ppo_config):
        super().__init__()
        self.model = model
        self.model_config = model_config
        self.ppo_config = ppo_config

    def step(self, inputs, branch, *, track_grad, need_scores=True):
        observation, previous, action, initial, noise, increments = inputs
        return self.model.belief.particles_from_noise(
            observation, previous, action, initial,
            initial_noise=noise, langevin_noise=increments,
            step_size=float(self.model_config["langevin_step_size"]),
            temporal_gradient_mode=self.model_config["belief_gradient_mode"],
            track_grad=track_grad, potential_branch=branch, need_scores=need_scores,
        )

    def forward(self, operation, inputs, branches, objective="ppo"):
        if operation == "belief":
            return self.step(inputs, branches, track_grad=False)
        if operation == "recondition":
            return self.step(inputs, branches, track_grad=False, need_scores=False)[0]
        if operation == "policy":
            particles, scores, noises = inputs
            condition, value_condition = encode_conditions(self.model, particles, scores)
            policy = self.model.diffusion_policy
            action = noises[0]
            chain, log_probs = [action], []
            for i, step in enumerate(reversed(range(policy.num_steps))):
                distribution = policy.transition_distribution(condition, action, step)
                action = distribution.base_dist.loc + distribution.base_dist.scale * noises[i + 1]
                log_probs.append(distribution.log_prob(action))
                chain.append(action)
            return (torch.tanh(action) * policy.action_scale + policy.action_bias,
                    torch.stack(log_probs, dim=1), policy.chain_entropy(condition),
                    torch.stack(chain, dim=1),
                    value_from_conditions(self.model, condition, value_condition))
        data, prefix, prefix_active, starting_particles = inputs
        previous = starting_particles
        full = self.model_config["belief_gradient_mode"] == "full"
        # Prefixes have variable lengths across environments and seeds. Pad only
        # the replay grid; inactive rows retain their previous state exactly.
        for t, branch in enumerate(branches[0]):
            updated, _ = self.step((
                prefix["observations"][t], previous, prefix["previous_actions"][t],
                prefix["initial"][t], prefix["ula_initial_noise"][t],
                prefix["ula_step_noises"][t],
            ), branch, track_grad=full, need_scores=False)
            previous = torch.where(prefix_active[t, :, None, None], updated, previous)
        particles, scores = [], []
        for t, branch in enumerate(branches[1]):
            if t:
                previous = torch.where(data["initial"][t, :, None, None],
                                       torch.zeros_like(previous), previous)
            particle, score = self.step((
                data["observations"][t], previous, data["previous_actions"][t],
                data["initial"][t], data["ula_initial_noise"][t],
                data["ula_step_noises"][t],
            ), branch, track_grad=True)
            particles.append(particle)
            scores.append(score)
            previous = particle
        particles, scores = torch.stack(particles), torch.stack(scores)
        flatten = lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        condition, value_condition = encode_conditions(
            self.model, flatten(particles), flatten(scores)
        )
        zero = condition.new_zeros(())
        if objective == "prediction":
            loss = condition_prediction_loss(
                self.model.prediction_predictor, condition,
                observations=data["observations"], previous_actions=data["previous_actions"],
                dones=data["dones"],
            )
            return loss, torch.stack((loss, zero, zero, zero, zero))
        rollout = RolloutBatch(**data, pre_tanh_actions=None, belief_prefixes=(), wml_weights=None)
        indices = torch.arange(data["rewards"].shape[1], device=condition.device)
        policy_loss, kl, clip, entropy = _policy_objective(
            self.model, condition, rollout, indices, data["advantages"],
            self.model_config, self.ppo_config,
        )
        new_values = value_from_conditions(self.model, condition, value_condition)
        target = ValueTransform.from_config(self.ppo_config).to_target(flatten(data["returns"]))
        value_loss = 0.5 * (new_values - target).square().mean()
        loss = (policy_loss + float(self.ppo_config["value_coef"]) * value_loss
                - float(self.ppo_config["entropy_coef"]) * entropy)
        return loss, torch.stack((policy_loss, value_loss, entropy, kl, clip))


def replay_inputs(rollouts, selections, normalized_advantages):
    """Select sequence microbatches before vmap (including ragged host metadata)."""
    names = [field.name for field in fields(RolloutBatch)
             if field.name not in {"belief_prefixes", "wml_weights", "pre_tanh_actions"}]
    selected = []
    prefixes = []
    for rollout, indices, advantages in zip(rollouts, selections, normalized_advantages):
        values = {}
        for name in names:
            value = advantages if name == "advantages" else getattr(rollout, name)
            values[name] = None if value is None else value[:, indices]
        selected.append(values)
        prefixes.append([rollout.belief_prefixes[i] for i in indices.cpu().tolist()])
    data = {name: None if selected[0][name] is None else
            torch.stack([row[name] for row in selected]) for name in names}
    length = max(p.length for group in prefixes for p in group)
    count, _, group = data["initial"].shape
    active = torch.zeros(count, length, group, device=data["initial"].device, dtype=torch.bool)
    prefix = {}
    for name in ("observations", "previous_actions", "initial", "ula_initial_noise", "ula_step_noises"):
        sample = data[name]
        prefix[name] = sample.new_zeros((count, length, group, *sample.shape[3:]))
    starting = data["previous_particles"][:, 0].detach().clone()
    for s, row in enumerate(prefixes):
        for e, history in enumerate(row):
            if not history.length:
                continue
            starting[s, e] = 0
            active[s, :history.length, e] = True
            for name in prefix:
                prefix[name][s, :history.length, e] = getattr(history, name)
    prefix_branches = branch_rows(prefix["initial"], active)
    branches = branch_rows(data["initial"])
    return (data, prefix, active, starting), (prefix_branches, branches)


class SeedBatchBackend:
    def __init__(self, models, optimizers, model_config, ppo_config):
        self.models, self.optimizers = models, optimizers
        self.model_config, self.ppo_config = model_config, ppo_config
        wrappers = [FunctionalModel(m, model_config, ppo_config) for m in models]
        self.params, self.buffers = stack_module_state(wrappers)
        self.template = copy.deepcopy(wrappers[0]).to("meta")
        use_batched_attention(self.template)
        self.model_parameters = [dict(m.named_parameters()) for m in models]
        self.batched_calls = 0

    def call(self, operation, inputs, branches, objective="ppo"):
        self.template.train(operation == "loss")
        def single(parameters, buffers, values):
            return functional_call(self.template, (parameters, buffers),
                                   (operation, values, branches, objective), strict=True)
        with functional_kernels():
            result = vmap(single, randomness="error")(self.params, self.buffers, inputs)
        self.batched_calls += 1
        return result

    def belief(self, payloads):
        inputs = tuple(torch.stack(items) for items in zip(*payloads))
        with torch.no_grad():
            return self.call("belief", inputs, branch_for(inputs[3]))

    def policy(self, payloads):
        with torch.no_grad():
            return self.call("policy", tuple(torch.stack(x) for x in zip(*payloads)), None)

    def recondition(self, payloads):
        contexts, histories = zip(*payloads)
        previous = torch.zeros_like(torch.stack([c.previous_particles for c in contexts]))
        count, envs = previous.shape[:2]
        length = max(len(history.observations) for row in histories for history in row)
        device = previous.device
        active = torch.zeros(count, length, envs, dtype=torch.bool, device=device)
        samples = (contexts[0].observation, contexts[0].previous_action, contexts[0].initial,
                   contexts[0].previous_particles,
                   previous.new_zeros((envs, int(self.model_config["langevin_steps"]), *previous.shape[2:])))
        names = ("observations", "previous_actions", "initial", "ula_initial_noise", "ula_step_noises")
        tensors = {name: sample.new_zeros((count, length, *sample.shape))
                   for name, sample in zip(names, samples)}
        for s, row in enumerate(histories):
            for e, history in enumerate(row):
                size = len(history.observations)
                if not size:
                    if not bool(contexts[s].initial[e]):
                        raise RuntimeError("a non-initial belief context is missing its episode history")
                    continue
                active[s, :size, e] = True
                for name in names:
                    tensors[name][s, :size, e] = torch.stack(getattr(history, name))
        branches = branch_rows(tensors["initial"], active)
        with torch.no_grad():
            for t in range(length):
                inputs = (tensors["observations"][:, t], previous, tensors["previous_actions"][:, t],
                          tensors["initial"][:, t], tensors["ula_initial_noise"][:, t],
                          tensors["ula_step_noises"][:, t])
                updated = self.call("recondition", inputs, branches[t])
                previous = torch.where(active[:, t, :, None, None], updated, previous)
        return previous

    def update(self, rollouts, streams, objective):
        cfg = self.ppo_config
        steps, envs = rollouts[0].rollout_shape
        group = min(int(cfg["minibatch_size"]) // steps, envs)
        micro = int(cfg["sequence_microbatch_size"]) // steps
        advantages = [r.advantages for r in rollouts]
        if cfg["normalize_advantage"]:
            advantages = [(a - a.mean()) / (a.std(unbiased=False) + 1e-8) for a in advantages]
        totals = torch.zeros(len(rollouts), 9, dtype=torch.float64)
        batches = [0] * len(rollouts)
        for _ in range(int(cfg["epochs"])):
            permutations = [stream.run(lambda: torch.randperm(envs, device=advantages[0].device))
                            for stream in streams]
            for start in range(0, envs, group):
                size = min(group, envs - start)
                for value in self.params.values():
                    value.grad = None
                metrics = torch.zeros(len(rollouts), 5, dtype=torch.float64)
                used_initial = torch.zeros(len(rollouts), dtype=torch.bool)
                used_recursive = torch.zeros_like(used_initial)
                for offset in range(0, size, micro):
                    width = min(micro, size - offset)
                    selection = [p[start + offset:start + offset + width] for p in permutations]
                    inputs, branches = replay_inputs(rollouts, selection, advantages)
                    data, prefix, active, _ = inputs
                    initial = data["initial"].reshape(len(rollouts), -1)
                    used_initial |= initial.any(dim=1).cpu()
                    used_recursive |= (~initial).any(dim=1).cpu()
                    if self.model_config["belief_gradient_mode"] == "full":
                        used_initial |= (prefix["initial"] & active).flatten(1).any(dim=1).cpu()
                        used_recursive |= (~prefix["initial"] & active).flatten(1).any(dim=1).cpu()
                    losses, values = self.call("loss", inputs, branches, objective)
                    weight = width / size
                    # Sum, never average, across seeds: each seed sees precisely
                    # the gradient scale of its standalone microbatch objective.
                    (losses.sum() * weight).backward()
                    metrics += values.detach().double().cpu() * weight
                    del losses, values, inputs
                for s, (model, optimizer, parameters) in enumerate(zip(
                        self.models, self.optimizers, self.model_parameters)):
                    optimizer.zero_grad(set_to_none=True)
                    for name, parameter in parameters.items():
                        grad = self.params["model." + name].grad
                        used = True
                        if name.startswith("belief.initial_energy."):
                            used = bool(used_initial[s])
                        elif name.startswith(("belief.observation_energy.", "belief.transition_energy.")):
                            used = bool(used_recursive[s])
                        # A masked-out branch must retain grad=None, including
                        # Adam's step counter and AdamW decay semantics.
                        if grad is not None and used:
                            parameter.grad = grad[s].clone()
                    norms = [_module_gradient_norm(getattr(model.belief, name, None), reference=model)
                             for name in ("initial_energy", "observation_energy", "transition_energy")]
                    norm, finite = clip_gradients_or_skip(model.parameters(), float(cfg["max_grad_norm"]))
                    if not finite:
                        continue
                    optimizer.step()
                    totals[s, :5] += metrics[s]
                    totals[s, 5:] += torch.tensor([float(norm), *(float(n) for n in norms)], dtype=torch.float64)
                    batches[s] += 1
                # Optimizers and exported model tensors retain the original
                # compact per-seed layout and checkpoint format. The expensive
                # forward and backward graphs above are shared across seeds.
                with torch.no_grad():
                    for name, value in self.params.items():
                        value.copy_(torch.stack([p[name.removeprefix("model.")] for p in self.model_parameters]))
        if any(count == 0 for count in batches):
            raise RuntimeError("PPO update produced no finite minibatches for at least one seed")
        return [PPOMetrics(*(row / count).tolist()) for row, count in zip(totals, batches)]
