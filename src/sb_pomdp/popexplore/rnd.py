"""RND with fixed random target, running error variance and detached inputs."""
import torch
from torch import nn


class RunningMoments:
    def __init__(self):
        self.count, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, values):
        values = values.detach().double()
        count = values.numel()
        mean = float(values.mean())
        m2 = float(((values - mean)**2).sum())
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta**2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    @property
    def variance(self):
        return self.m2 / max(self.count, 1)


class RND(nn.Module):
    def __init__(self, inputs, config, device):
        super().__init__()
        def network():
            return nn.Sequential(nn.Linear(inputs, config["rnd_hidden"]), nn.ReLU(),
                                 nn.Linear(config["rnd_hidden"], config["rnd_features"]))
        self.target, self.predictor = network().to(device), network().to(device)
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=config["rnd_learning_rate"])
        self.moments, self.clip = RunningMoments(), config["rnd_clip"]

    def error(self, inputs):
        # Bounded, parameter-free input scaling; normalization cannot train belief.
        x = torch.tanh(inputs.detach())
        return (self.predictor(x) - self.target(x)).square().mean(-1)

    @torch.no_grad()
    def reward(self, inputs):
        error = self.error(inputs)
        self.moments.update(error)
        return (error / max(self.moments.variance, 1e-8)**0.5).clamp(max=self.clip)

    def update_predictor(self, inputs):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.error(inputs).mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def training_state(self):
        return {"model": self.state_dict(), "optimizer": self.optimizer.state_dict(),
                "moments": vars(self.moments).copy()}

    def restore(self, state):
        self.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.moments.__dict__.update(state["moments"])
