"""Rank-based CEM, ES, PBT, and a bounded MAP-Elites/CEM hybrid."""
import math
import numpy as np
import torch


def ranks(values):
    """Average ranks, so tied fitness does not invent a selection gradient."""
    values = np.asarray(values, dtype=float)
    result = np.empty(len(values), dtype=float)
    for value in np.unique(values):
        mask = values == value
        result[mask] = (np.count_nonzero(values < value) + (mask.sum() - 1) / 2)
    return result


def rank_correlation(a, b):
    x, y = ranks(a), ranks(b)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


class PopulationSearch:
    def __init__(self, vectors, config):
        self.config = config
        self.mean = vectors.detach().mean(0).clone()
        self.std = torch.full_like(self.mean, config["sigma"])
        self.archive = {}
        self.generation = 0

    def update(self, vectors, fitness, descriptors):
        if not bool(torch.isfinite(vectors).all()) or not np.isfinite(fitness).all():
            raise ValueError("nonfinite population/fitness")
        c, n = self.config, len(vectors)
        order = np.argsort(-np.asarray(fitness), kind="stable")
        elite = order[:max(1, math.ceil(n * c["elite_fraction"]))]
        selected = np.zeros(n, dtype=bool)
        selected[elite] = True
        parents = np.arange(n)
        changed = np.zeros(n, dtype=bool)
        rule = c["update_rule"]
        result = vectors.detach().clone()
        if rule == "none":
            self.mean = vectors.detach().mean(0).clone()
            return result, selected & False, parents, changed
        if rule == "qd":
            for i in range(n):
                key = tuple(np.floor(np.asarray(descriptors[i]) / c["qd_cell_size"]).astype(int))
                if key not in self.archive or fitness[i] > self.archive[key]["fitness"]:
                    self.archive[key] = {"fitness": float(fitness[i]), "vector": vectors[i].detach().cpu().clone(),
                                         "descriptor": np.asarray(descriptors[i]).copy(), "generation": self.generation}
            while len(self.archive) > c["qd_capacity"]:
                del self.archive[min(self.archive, key=lambda k: self.archive[k]["fitness"])]
        if rule in {"cem", "qd"}:
            chosen = vectors[torch.as_tensor(elite, device=vectors.device)]
            rate = c["cem_rate"]
            self.mean = ((1 - rate) * self.mean + rate * chosen.mean(0)).detach()
            self.std = ((1 - rate) * self.std.square() + rate * chosen.var(0, unbiased=False)).sqrt().clamp_min(c["sigma_min"]).detach()
            result = self.mean + self.std * torch.randn_like(vectors)
            parents[:] = -1  # distribution, not a single genetic parent
            changed[:] = True
            if rule == "qd":
                archive = list(self.archive.values())
                count = min(n, math.floor(n * c["qd_parent_fraction"]))
                for slot, pick in zip(range(n - count, n), torch.randint(len(archive), (count,)).tolist()):
                    result[slot] = archive[pick]["vector"].to(vectors) + c["mutation_std"] * torch.randn_like(vectors[slot])
                    parents[slot] = -2 - pick  # archive insertion-order index
        elif rule == "es":
            utility = torch.as_tensor(ranks(fitness), device=vectors.device, dtype=vectors.dtype)
            utility -= utility.mean()
            utility /= utility.abs().sum().clamp_min(1)
            gradient = (utility[:, None] * (vectors - self.mean)).sum(0) / c["sigma"]**2
            self.mean = (self.mean + c["es_learning_rate"] * gradient).detach()
            self.std.fill_(max(c["sigma"], c["sigma_min"]))
            result = self.mean + self.std * torch.randn_like(vectors)
            parents[:] = -1
            changed[:] = True
        elif rule == "pbt":
            count = min(n - 1, max(1, math.ceil(n * c["pbt_fraction"])))
            for k, slot in enumerate(order[-count:]):
                parent = int(elite[k % len(elite)])
                result[slot] = vectors[parent] + c["mutation_std"] * torch.randn_like(vectors[slot])
                parents[slot], changed[slot] = parent, True
            self.mean = result.mean(0).detach()
        self.generation += 1
        return result, selected, parents, changed


class ActorAdam:
    """Vectorized Adam(W), with an independent step counter for every actor."""
    def __init__(self, parameters, config, *, eps=1e-8):
        self.parameters = list(parameters)
        self.eps = eps
        self.lr, self.decay = config["learning_rate"], config.get("weight_decay", 0.0)
        n = self.parameters[0].shape[0]
        self.steps = torch.zeros(n, device=self.parameters[0].device, dtype=torch.long)
        self.first = [torch.zeros_like(p) for p in self.parameters]
        self.second = [torch.zeros_like(p) for p in self.parameters]

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    @torch.no_grad()
    def step(self, active):
        self.steps += active.long()
        for p, m, v in zip(self.parameters, self.first, self.second):
            if p.grad is None:
                continue
            mask = active.reshape(-1, *([1] * (p.ndim - 1)))
            m.copy_(torch.where(mask, 0.9 * m + 0.1 * p.grad, m))
            v.copy_(torch.where(mask, 0.999 * v + 0.001 * p.grad.square(), v))
            count = self.steps.clamp_min(1).reshape_as(mask)
            direction = (m / (1 - 0.9**count)) / ((v / (1 - 0.999**count)).sqrt() + self.eps)
            p.copy_(torch.where(mask, p * (1 - self.lr * self.decay) - self.lr * direction, p))

    @torch.no_grad()
    def reset(self, changed):
        self.steps[changed] = 0
        for m, v in zip(self.first, self.second):
            m[changed] = 0
            v[changed] = 0

    def state_dict(self):
        return {"steps": self.steps, "first": self.first, "second": self.second}

    def load_state_dict(self, state):
        self.steps.copy_(state["steps"])
        for name in ("first", "second"):
            for target, source in zip(getattr(self, name), state[name]):
                target.copy_(source)
