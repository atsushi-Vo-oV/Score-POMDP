"""Belief-mixture next-observation predictive (model.prediction_head=belief_mixture).

o_{t+1} is conditionally independent of the history given the state, so the predictive density of
a particle belief is the mixture  sum_k w_k p(o_{t+1} | x_k, a_t); no true state is ever used.
"""

from __future__ import annotations

import math

import pytest
import torch

from sb_pomdp.networks import ObservationPredictor
from sb_pomdp.ppo import belief_mixture_prediction_loss


def test_weighted_mixture_matches_the_formula_and_uniform_weights_match_the_default() -> None:
    torch.manual_seed(0)
    predictor = ObservationPredictor(2, 1, 3, [8])
    particles, actions, target = torch.randn(4, 5, 2), torch.randn(4, 1), torch.randn(4, 3)
    log_weights = torch.randn(4, 5)
    weighted = predictor.log_likelihood(particles, actions, target, log_weights)
    outputs = predictor.network(torch.cat((particles, actions[:, None].expand(-1, 5, -1)), -1))
    mean, log_std = outputs.chunk(2, -1)
    log_std = log_std.clamp(-4.0, 3.0)
    per = (-0.5 * ((target[:, None] - mean) / log_std.exp()).square() - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
    assert torch.allclose(weighted, torch.logsumexp(per + torch.log_softmax(log_weights, -1), 1), atol=1e-5)
    uniform = predictor.log_likelihood(particles, actions, target, torch.zeros(4, 5))
    assert torch.allclose(uniform, predictor.log_likelihood(particles, actions, target), atol=1e-6)


def test_loss_targets_the_next_observation_and_masks_episode_ends() -> None:
    torch.manual_seed(1)
    predictor = ObservationPredictor(2, 1, 3, [8])
    steps, group = 4, 2
    particles = torch.randn(steps, group, 5, 2, requires_grad=True)
    observations, actions = torch.randn(steps, group, 3), torch.randn(steps, group, 1)
    dones = torch.zeros(steps, group, dtype=torch.bool)
    dones[1, 0] = True
    loss = belief_mixture_prediction_loss(predictor, particles, observations=observations,
                                          previous_actions=actions, dones=dones)
    # Each sequence is averaged over its own valid steps, then the sequences are averaged.
    expected = torch.stack([
        -torch.stack([predictor.log_likelihood(particles[t, g][None], actions[t + 1, g][None],
                                               observations[t + 1, g][None])[0]
                      for t in range(steps - 1) if (t, g) != (1, 0)]).mean()
        for g in range(group)
    ]).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert particles.grad is not None and particles.grad[1, 0].abs().sum() == 0  # masked step
    assert particles.grad[0, 0].abs().sum() > 0


@pytest.mark.parametrize("method", ["score_transformer", "particle_filter", "gru"])
def test_prediction_phase_trains_with_the_belief_mixture_head(method: str, tmp_path) -> None:
    import subprocess
    import sys

    config = "config/trysvgd_tps_tf-ula.json" if method == "score_transformer" else "config/trysvgd_tps_base.json"
    overrides = ['experiment.device="cpu"', f'experiment.output_dir="{tmp_path.as_posix()}"', "experiment.seeds=[0]",
                 "experiment.eval_episodes=1", "ppo.num_envs=2", "ppo.rollout_steps=8", "ppo.minibatch_size=16",
                 "ppo.sequence_microbatch_size=8", "ppo.epochs=1", "ppo.total_steps=32",
                 "model.prediction_pretrain_steps=16", "model.prediction_interval_updates=1",
                 "model.prediction_phase_updates=1", 'model.prediction_head="belief_mixture"',
                 "model.belief_activation_checkpoint=false"]
    command = [sys.executable, "-m", "sb_pomdp.compare", "--config", config, "--methods", method, "--seeds", "0",
               "--label", f"bm-{method}"] + [item for o in overrides for item in ("--override", o)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900,
                            env={**__import__("os").environ, "PYTHONPATH": "src", "OMP_NUM_THREADS": "1"})
    log = result.stdout + result.stderr
    assert result.returncode == 0 and "failed" not in log, log[-2000:]
    assert log.count("prediction phase=") >= 2
