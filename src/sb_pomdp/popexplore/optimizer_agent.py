"""Paired, lineage-aware adoption and diffusion noise-prediction imitation."""
import math
import numpy as np
import torch


def student_quantile(probability, degrees):
    """Student t quantile without a SciPy dependency (angle-domain quadrature)."""
    nodes, weights = np.polynomial.legendre.leggauss(96)
    normalizer = math.exp(math.lgamma((degrees + 1) / 2) - math.lgamma(degrees / 2)) / math.sqrt(math.pi)
    low, high = 0.0, math.pi / 2
    for _ in range(60):
        angle = (low + high) / 2
        x = (nodes + 1) * angle / 2
        cdf = .5 + normalizer * angle / 2 * np.dot(weights, np.cos(x) ** (degrees - 1))
        if cdf < probability:
            low = angle
        else:
            high = angle
    return math.sqrt(degrees) * math.tan((low + high) / 2)


class AdoptionGate:
    """Keep same-update differences for each live exploratory lineage.

    A replacement or adoption invalidates the comparison history. PPO changes
    within the same lineage are allowed; this is a local performance criterion,
    not a formal stationary-policy hypothesis test.
    """
    def __init__(self, config, population_size):
        self.config = config
        self.history = [[] for _ in range(population_size)]
        self.count = 0
        self.last_update = 0
        self.critical = student_quantile(
            1 - (1 - config["confidence"]) / (population_size - 1), config["window"] - 1)

    def observe(self, update, scores):
        candidates = []
        for i in range(1, len(scores)):
            h = self.history[i]
            if not np.isfinite([scores[0], scores[i]]).all():
                h.clear()
                continue
            h.append((update, float(scores[i] - scores[0])))
            del h[:-self.config["window"]]
            values = np.asarray([v for _, v in h])
            mean = float(values.mean())
            lower = (mean - self.critical * float(values.std(ddof=1)) / math.sqrt(len(h))
                     if len(h) == self.config["window"] else math.nan)
            candidates.append(dict(candidate=i, pairs=len(h), difference=mean, lower=lower,
                                   eligible=mean > self.config["margin"] and lower > 0))
        eligible = [c for c in candidates if c["eligible"]]
        return max(eligible or candidates, key=lambda c: c["difference"]) if candidates else dict(
            candidate=-1, pairs=0, difference=math.nan, lower=math.nan, eligible=False)

    def reset(self, replaced=None):
        for i, h in enumerate(self.history):
            if replaced is None or replaced[i]:
                h.clear()


def noise_prediction_loss(policy, condition, actions, steps, noise):
    normalized = ((actions - policy.action_bias) / policy.action_scale).clamp(-1 + 1e-6, 1 - 1e-6)
    clean = torch.atanh(normalized)
    alpha = policy.alpha_bars[steps, None].to(clean)
    noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    time = policy.time_embedding(steps, policy.num_steps).to(condition)
    prediction = policy.denoiser(torch.cat((condition, noisy, time), -1))
    return (prediction - noise).square().mean()


def imitate(policy, condition, actions, config):
    """Only actor parameters train; saved rollout conditions/actions are detached."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["bc_learning_rate"])
    condition, actions = condition.detach(), actions.detach()
    losses = []
    for _ in range(config["bc_steps"]):
        steps = torch.randint(policy.num_steps, (len(actions),), device=actions.device)
        noise = torch.randn_like(actions)
        loss = noise_prediction_loss(policy, condition, actions, steps, noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))
