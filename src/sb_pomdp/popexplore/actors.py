"""vmap actor kernels. Belief/readout/value parameters are never replicated."""
import copy
import math

import torch
from torch import nn
from torch.func import functional_call, vmap

from ..seedbatch.backend import functional_kernels


def assignments(num_envs, population_size, device="cpu"):
    if population_size < 1 or num_envs % population_size:
        raise ValueError("num_envs must be a multiple of population_size")
    return torch.arange(population_size, device=device).repeat_interleave(num_envs // population_size)


class ActorKernel(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, operation, condition, values, deterministic=False):
        p = self.policy
        if operation == "log_prob":
            return p.log_prob_chain(condition, values)
        action = values[:, 0]
        chain, logs = [action], []
        for index, step in enumerate(reversed(range(p.num_steps))):
            distribution = p.transition_distribution(condition, action, step)
            action = distribution.base_dist.loc
            if not deterministic:
                action = action + distribution.base_dist.scale * values[:, index + 1]
            logs.append(distribution.log_prob(action))
            chain.append(action)
        return (torch.tanh(action) * p.action_scale + p.action_bias,
                torch.stack(chain, 1), torch.stack(logs, 1))


class PopulationPolicy(nn.Module):
    def __init__(self, policy, size, num_envs, init_std):
        super().__init__()
        self.size, self.num_steps, self.action_dim = size, policy.num_steps, policy.action_dim
        object.__setattr__(self, "prototype", copy.deepcopy(policy))
        object.__setattr__(self, "template", ActorKernel(copy.deepcopy(policy)).to("meta"))
        self.names = tuple(name for name, _ in policy.named_parameters())
        self.weights = nn.ParameterList([
            nn.Parameter(value.detach().unsqueeze(0).repeat(size, *([1] * value.ndim)))
            for value in policy.parameters()
        ])
        self.constant = {"policy." + name: value.detach().clone() for name, value in policy.named_buffers()}
        self.register_buffer("action_scale", policy.action_scale.detach().clone())
        self.register_buffer("action_bias", policy.action_bias.detach().clone())
        with torch.no_grad():
            if init_std:
                for value in self.weights:
                    value.add_(torch.randn_like(value) * init_std)
        self.ids = assignments(num_envs, size, self.action_scale.device)
        self.episode_weights = [p.detach()[self.ids].clone() for p in self.weights]
        self.last_condition = None

    def vectors(self):
        return torch.cat([p.reshape(self.size, -1) for p in self.weights], 1)

    @torch.no_grad()
    def set_vectors(self, vectors):
        offset = 0
        for p in self.weights:
            count = p[0].numel()
            p.copy_(vectors[:, offset:offset + count].reshape_as(p))
            offset += count
        if offset != vectors.shape[1]:
            raise ValueError("actor vector size mismatch")

    @torch.no_grad()
    def start_episodes(self, initial):
        # A slot's new generation takes effect only at the next episode reset.
        for snapshot, current in zip(self.episode_weights, self.weights):
            snapshot[initial] = current[self.ids[initial]]

    def call(self, operation, weights, condition, values, deterministic=False):
        params = {"policy." + name: p for name, p in zip(self.names, weights)}
        def one(parameters, c, x):
            return functional_call(self.template, (parameters, self.constant),
                                   (operation, c, x, deterministic), strict=True)
        with functional_kernels():
            return vmap(one, randomness="error")(params, condition, values)

    def sample(self, condition, *, deterministic, generator=None):
        self.last_condition = condition.detach()
        count = condition.shape[0]
        shape = (count, self.action_dim)
        # Preserve the original order of random draws: initial, then each step.
        draws = [condition.new_zeros(shape) if deterministic else
                 torch.randn(shape, device=condition.device, generator=generator)
                 for _ in range(self.num_steps + 1)]
        noise = torch.stack(draws, 1)
        result = self.call("sample", self.episode_weights, condition[:, None], noise[:, None], deterministic)
        return tuple(value[:, 0] for value in result)

    def sample_current(self, condition, ids, *, deterministic, generator=None):
        # Used by batched fitness evaluation. Actors remain fixed for the entire test.
        previous = self.episode_weights
        try:
            self.episode_weights = [p.detach()[ids] for p in self.weights]
            return self.sample(condition, deterministic=deterministic, generator=generator)
        finally:
            self.episode_weights = previous

    def log_prob_for(self, condition, chain, ids):
        # Group arbitrary sequence minibatches, padding only missing actor rows.
        groups = [torch.nonzero(ids == i).flatten() for i in range(self.size)]
        width = max(len(g) for g in groups)
        index = torch.zeros(self.size, width, device=ids.device, dtype=torch.long)
        row, col = torch.empty_like(ids), torch.empty_like(ids)
        for i, group in enumerate(groups):
            index[i, :len(group)] = group
            row[group], col[group] = i, torch.arange(len(group), device=ids.device)
        logs, entropy = self.call("log_prob", self.weights, condition[index], chain[index])
        return logs[row, col], entropy[row, col]

    def chain_entropy(self, condition):
        return self.prototype.chain_entropy(condition)

    def export(self, vector):
        policy = copy.deepcopy(self.prototype)
        offset = 0
        with torch.no_grad():
            for p in policy.parameters():
                count = p.numel()
                p.copy_(vector[offset:offset + count].reshape_as(p))
                offset += count
        return policy


class OUBias:
    """Stationary OU discretization, independent of all learnable parameters."""
    def __init__(self, envs, actions, std, tau, device):
        self.std, self.rho = std, math.exp(-1.0 / tau)
        self.value = torch.zeros(envs, actions, device=device)
        self.xy = self.xx = self.yy = 0.0
        self.pairs = 0

    def step(self, initial, enabled=None):
        old = self.value.clone()
        eps = torch.randn_like(old)
        self.value = torch.where(initial[:, None], self.std * eps,
                                 self.rho * old + self.std * math.sqrt(1 - self.rho**2) * eps)
        active = ~initial
        if enabled is not None:
            self.value.masked_fill_(~enabled[:, None], 0)
            active = active & enabled
        x, y = old[active], self.value[active]
        self.xy += float((x * y).sum())
        self.xx += float(x.square().sum())
        self.yy += float(y.square().sum())
        self.pairs += x.numel()
        return self.value

    def correlation(self):
        rho = self.xy / math.sqrt(max(self.xx * self.yy, 1e-30))
        # Undefined/negative correlations are NaN, not a fictitious zero time.
        tau = -1.0 / math.log(rho) if 0 < rho < 1 else float("nan")
        return rho, tau
