"""Local integrability regularisation on detached belief-replay coordinates.

The score callable must be pointwise in its query coordinates (conditioning on a
fixed previous belief is fine). No normalisation by latent dimension is applied.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial

import torch


def curl_enabled(config: Mapping) -> bool:
    return float(config.get("score_curl_penalty", 0.0)) > 0 or bool(
        config.get("score_curl_monitor", False)
    )


def curl_metric_fields(config: Mapping) -> tuple[str, ...]:
    if float(config.get("score_curl_penalty", 0.0)) > 0:
        return ("score_curl_loss", "score_curl_ratio", "score_curl_coef")
    return ("score_curl_ratio",) if config.get("score_curl_monitor", False) else ()


def curl_coefficient(config: Mapping, *, ppo_updates: int, total_ppo_updates: int) -> float:
    coefficient = float(config.get("score_curl_penalty", 0.0))
    if config.get("score_curl_schedule", "constant") == "ramp":
        coefficient *= min(1.0, max(0.0, ppo_updates / max(1, total_ppo_updates)))
    return coefficient


def curl_generator(seed: int, update: int) -> torch.Generator:
    """A private CPU stream, reproducible across devices and segmented resumes."""
    return torch.Generator(device="cpu").manual_seed(
        (int(seed) + 1_000_003 * int(update) + 0x4355524C) % (2**63 - 1)
    )


def score_curl_statistics(
    score_function: Callable[[torch.Tensor], torch.Tensor],
    positions: torch.Tensor,
    *,
    estimator: str = "auto",
    probes: int = 4,
    generator: torch.Generator | None = None,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean ||J-J.T||_F^2 and mean ||J-J.T||_F / ||J||_F.

    Hutchinson uses independent Rademacher u,v: E[(u.T J v-v.T J u)^2]
    = ||J-J.T||_F^2, with no 1/2, 1/4 or D factor. The reported ratio is
    always exact: a ratio of stochastic norm estimates would be biased and can
    be unstable near zero. Its D diagnostic VJPs do not retain higher derivatives.
    A zero Jacobian has ratio zero. Only the loss carries a parameter gradient.
    """
    dimension = positions.shape[-1]
    if estimator == "auto":
        estimator = "exact" if dimension <= 4 else "hutchinson"
    if estimator not in {"exact", "hutchinson"}:
        raise ValueError("score_curl_estimator must be auto, exact or hutchinson")
    if probes < 1:
        raise ValueError("score_curl_probes must be positive")
    if generator is None:
        generator = curl_generator(0, 0)
    with torch.enable_grad():
        query = positions.detach().requires_grad_(True)
        score = score_function(query)
        # Also permit a constant field, whose derivative is exactly zero.
        score = score + query * 0.0
        rows = [
            torch.autograd.grad(
                score[..., i].sum(), query, retain_graph=True,
                create_graph=create_graph and estimator == "exact",
            )[0]
            for i in range(dimension)
        ]
        jacobian = torch.stack(rows, dim=-2)
        difference = jacobian - jacobian.transpose(-1, -2)
        squared = difference.square().sum(dim=(-1, -2))
        norm = jacobian.detach().square().sum(dim=(-1, -2)).sqrt()
        ratio = (squared.detach().sqrt() / norm.clamp_min(torch.finfo(norm.dtype).tiny)).mean()
        if estimator == "exact":
            loss = squared.mean()
        else:
            estimates = []
            for _ in range(probes):
                u, v = (
                    (torch.randint(2, query.shape, generator=generator, device="cpu") * 2 - 1)
                    .to(device=query.device, dtype=query.dtype)
                    for _ in range(2)
                )
                jtu = torch.autograd.grad(
                    score, query, u, retain_graph=True, create_graph=create_graph,
                )[0]
                jtv = torch.autograd.grad(
                    score, query, v, retain_graph=True, create_graph=create_graph,
                )[0]
                estimates.append(((jtu * v).sum(-1) - (jtv * u).sum(-1)).square().mean())
            loss = torch.stack(estimates).mean()
    return loss, ratio


class CurlReplaySamples:
    """Select N (environment,time,chain-stage) tuples per optimiser minibatch.

    Time includes the active-episode prefixes. Sampling is uniform over all
    replayed environment/time pairs, with replacement, then over transport
    stages 0..L-1. Each tuple evaluates all K particles. Captures only detached
    incoming beliefs; no reference to the PPO/BPTT graph survives in this object.
    """

    def __init__(self, rollout, environment_indices, config, generator):
        self.config = config
        self.generator = generator
        times = [
            (environment, time)
            for environment in environment_indices.detach().cpu().tolist()
            for time in range(-rollout.belief_prefixes[environment].length, rollout.rollout_shape[0])
        ]
        count = int(config.get("score_curl_samples", 8))
        selected = torch.randint(len(times), (count,), generator=generator).tolist()
        stages = torch.randint(
            rollout.ula_step_noises.shape[2], (count,), generator=generator,
        ).tolist()
        self.plan: dict[tuple[int, int], list[int]] = {}
        for index, stage in zip(selected, stages):
            self.plan.setdefault(times[index], []).append(stage)
        self.contexts: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}

    def capture(
        self, environments: Sequence[int], times: Sequence[int], *values: torch.Tensor,
    ) -> None:
        for row, key in enumerate(zip(environments, times)):
            if key in self.plan:
                self.contexts[key] = tuple(value[row:row + 1].detach().clone() for value in values)

    def evaluate(self, belief, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        losses, ratios = [], []
        if self.plan.keys() != self.contexts.keys():
            raise RuntimeError("curl samples were not captured by belief replay")
        # Iterate in sampled order, independent of sequence microbatch partitioning.
        for key, stages in self.plan.items():
            observation, previous, action, initial, noise, increments = self.contexts[key]
            for stage in stages:
                with torch.no_grad():
                    query, _ = belief.particles_from_noise(
                        observation, previous, action, initial,
                        initial_noise=noise, langevin_noise=increments,
                        step_size=float(self.config["langevin_step_size"]), track_grad=False,
                        need_scores=False, stop_before_step=stage,
                    )
                    # Resolve a learned initial prior without differentiating its context.
                    resolved_previous, resolved_action, resolved_initial = belief._resolve_context(
                        previous, action, initial,
                    )
                    branch = belief.classify_potential_branch(resolved_initial)

                field = partial(
                    belief._score_at_for_branch, observation=observation,
                    previous_particles=resolved_previous, previous_action=resolved_action,
                    initial=resolved_initial, potential_branch=branch,
                    create_graph=True, detach_query=False,
                )

                loss, ratio = score_curl_statistics(
                    field, query, estimator=str(self.config.get("score_curl_estimator", "auto")),
                    probes=int(self.config.get("score_curl_probes", 4)),
                    generator=self.generator, create_graph=create_graph,
                )
                losses.append(loss)
                ratios.append(ratio)
        return torch.stack(losses).mean(), torch.stack(ratios).mean()
