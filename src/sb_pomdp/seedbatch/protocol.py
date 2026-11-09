"""Cooperative suspension points and independent global RNG streams."""
from __future__ import annotations

import random
import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


@dataclass
class Request:
    kind: str
    payload: Any


class SeedStream:
    """Swap the *complete* RNG state around one cooperative host-side operation.

    No threads run concurrently. Drawing the same tensor shapes in the same
    per-seed order preserves compare's stream, including CUDA Philox offsets.
    Tensor computation inside vmap is deterministic and consumes no randomness.
    """

    def __init__(self):
        self.state = self.capture()
        self.handlers = []

    @staticmethod
    def capture():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

    @staticmethod
    def restore(state):
        python, numpy, cpu, cuda = state
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)

    def run(self, operation: Callable):
        outer = self.capture()
        logger = logging.getLogger("sb_pomdp")
        outer_handlers = logger.handlers[:]
        logger.handlers = self.handlers[:]
        self.restore(self.state)
        try:
            return operation()
        finally:
            self.state = self.capture()
            self.restore(outer)
            self.handlers = logger.handlers[:]
            logger.handlers = outer_handlers

    def close(self):
        for handler in self.handlers:
            handler.close()
        self.handlers.clear()


def sample_request(model, context, model_config, *, rollout):
    initial_noise, step_noises = model.belief.draw_particle_noise(
        context.observation,
        num_particles=int(model_config["num_particles"]),
        num_steps=int(model_config["langevin_steps"]),
    )
    particles, scores = yield Request("belief", (
        context.observation, context.previous_particles, context.previous_action,
        context.initial, initial_noise, step_noises,
    ))
    if rollout:
        return particles, scores, initial_noise, step_noises
    return particles, scores


def policy_request(model, particles, scores):
    policy = model.diffusion_policy
    shape = (particles.shape[0], policy.action_dim)
    # Match DiffusionPolicy.sample's separate RNG calls, including the initial
    # x^N draw. Pre-drawing is safe: all intervening forward operations are pure.
    noises = torch.stack([torch.randn(shape, device=particles.device, dtype=particles.dtype)
                          for _ in range(policy.num_steps + 1)])
    action, log_prob, entropy, chain, value = yield Request("policy", (particles, scores, noises))
    return action, log_prob, entropy, chain, None, value
