"""P3O-style tilted collection and weighted-maximum-likelihood policy update."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sb_pomdp.buffer import RolloutBuilder
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.policies import ScoreBeliefActorCritic
from sb_pomdp.ppo import update_ppo
from sb_pomdp.train import (
    BeliefContext,
    BeliefReplayHistory,
    _clone_belief_history,
    _systematic_resample,
    _tilted_weights,
    build_optimizer,
)

STATE_DIM = 2
OBS_DIM = 3
PARTICLES = 4
STEPS = 2


def test_tilted_weights_normalise_and_order() -> None:
    weights = _tilted_weights(np.array([0.0, 1.0, -1.0]), eta=2.0)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[1] > weights[0] > weights[2]
    flat = _tilted_weights(np.array([3.0, 3.0, 3.0]), eta=5.0)
    assert np.allclose(flat, 1.0 / 3.0)


def test_systematic_resample_identity_and_concentration() -> None:
    rng = np.random.default_rng(0)
    uniform = _systematic_resample(np.full(6, 1.0 / 6.0), rng)
    assert sorted(uniform) == list(range(6))
    point = _systematic_resample(np.array([0.0, 0.0, 1.0, 0.0]), rng)
    assert point == [2, 2, 2, 2]


def test_clone_belief_history_is_independent() -> None:
    history = BeliefReplayHistory()
    history.observations.append(torch.zeros(3))
    clone = _clone_belief_history(history)
    clone.clear()
    assert len(history.observations) == 1


def _builder_with_steps(batch: int, steps: int) -> RolloutBuilder:
    builder = RolloutBuilder(belief_prefixes=())
    for step in range(steps):
        base = torch.arange(batch, dtype=torch.float32) + 100 * step
        builder.append(
            observations=base[:, None].repeat(1, OBS_DIM),
            previous_particles=base[:, None, None].repeat(1, PARTICLES, STATE_DIM),
            previous_actions=base[:, None].repeat(1, 2),
            initial=torch.zeros(batch, dtype=torch.bool),
            particles=base[:, None, None].repeat(1, PARTICLES, STATE_DIM),
            ula_initial_noise=base[:, None, None].repeat(1, PARTICLES, STATE_DIM),
            ula_step_noises=base[:, None, None, None].repeat(1, STEPS, PARTICLES, STATE_DIM),
            actions=base.long(),
            old_log_prob=base.clone(),
            old_values=base.clone(),
            rewards=base.clone(),
            dones=torch.zeros(batch, dtype=torch.bool),
        )
    return builder


def test_builder_resample_composes_ancestral_lineages() -> None:
    builder = _builder_with_steps(batch=3, steps=2)
    builder.resample([1, 1, 0])
    # A later step recorded after the resampling keeps its own identity.
    base = torch.arange(3, dtype=torch.float32) + 200
    builder.append(
        observations=base[:, None].repeat(1, OBS_DIM),
        previous_particles=base[:, None, None].repeat(1, PARTICLES, STATE_DIM),
        previous_actions=base[:, None].repeat(1, 2),
        initial=torch.zeros(3, dtype=torch.bool),
        particles=base[:, None, None].repeat(1, PARTICLES, STATE_DIM),
        ula_initial_noise=base[:, None, None].repeat(1, PARTICLES, STATE_DIM),
        ula_step_noises=base[:, None, None, None].repeat(1, STEPS, PARTICLES, STATE_DIM),
        actions=base.long(),
        old_log_prob=base.clone(),
        old_values=base.clone(),
        rewards=base.clone(),
        dones=torch.ones(3, dtype=torch.bool),
    )
    rollout = builder.finish(
        last_value=torch.zeros(3),
        gamma=0.99,
        gae_lambda=0.95,
        wml_weights=torch.tensor([0.2, 0.3, 0.5]),
    )
    # Steps 0 and 1: slot0/1 carry old slot 1, slot2 carries old slot 0.
    assert rollout.rewards[0].tolist() == [1.0, 1.0, 0.0]
    assert rollout.rewards[1].tolist() == [101.0, 101.0, 100.0]
    # Step 2 (post-resampling) is untouched.
    assert rollout.rewards[2].tolist() == [200.0, 201.0, 202.0]
    assert rollout.observations[0, 0].tolist() == [1.0, 1.0, 1.0]
    assert rollout.wml_weights is not None
    assert rollout.wml_weights.tolist() == pytest.approx([0.2, 0.3, 0.5])


def _tiny_model() -> tuple[ScoreBeliefActorCritic, dict, dict]:
    config = load_config("config/local.json").with_overrides(
        {
            "model.num_particles": PARTICLES,
            "model.langevin_steps": STEPS,
            "ppo.algorithm": "p3o_wml",
        }
    )
    model_config = dict(config["model"])
    ppo_config = dict(config["ppo"])
    ppo_config.update(
        {
            "epochs": 1,
            "minibatch_size": 8,
            "sequence_microbatch_size": 4,
            "learning_rate": 1e-3,
        }
    )
    torch.manual_seed(0)
    model = ScoreBeliefActorCritic(
        observation_dim=OBS_DIM,
        state_dim=STATE_DIM,
        action_kind="discrete",
        action_feature_dim=2,
        discrete_actions=2,
        action_low=None,
        action_high=None,
        model_config=model_config,
    )
    return model, model_config, ppo_config


def _rollout_for(model: ScoreBeliefActorCritic, batch: int, steps: int, weights):
    torch.manual_seed(1)
    context = BeliefContext(
        observation=torch.randn(batch, OBS_DIM),
        previous_particles=torch.zeros(batch, PARTICLES, STATE_DIM),
        previous_action=torch.zeros(batch, 2),
        initial=torch.ones(batch, dtype=torch.bool),
    )
    prefixes = tuple(
        BeliefReplayHistory().snapshot(context, index, langevin_steps=STEPS)
        for index in range(batch)
    )
    builder = RolloutBuilder(belief_prefixes=prefixes)
    for step in range(steps):
        initial = torch.full((batch,), step == 0, dtype=torch.bool)
        particles = torch.randn(batch, PARTICLES, STATE_DIM)
        builder.append(
            observations=torch.randn(batch, OBS_DIM),
            previous_particles=torch.randn(batch, PARTICLES, STATE_DIM),
            previous_actions=torch.randn(batch, 2),
            initial=initial,
            particles=particles,
            ula_initial_noise=torch.randn(batch, PARTICLES, STATE_DIM),
            ula_step_noises=torch.randn(batch, STEPS, PARTICLES, STATE_DIM),
            actions=torch.randint(0, 2, (batch,)),
            old_log_prob=-torch.rand(batch),
            old_values=torch.randn(batch),
            rewards=torch.randn(batch),
            dones=torch.full((batch,), step == steps - 1, dtype=torch.bool),
        )
    return builder.finish(
        last_value=torch.zeros(batch),
        gamma=0.99,
        gae_lambda=0.95,
        wml_weights=weights,
    )


def test_update_ppo_runs_the_weighted_ml_branch() -> None:
    model, model_config, ppo_config = _tiny_model()
    rollout = _rollout_for(model, batch=4, steps=4, weights=torch.tensor([0.7, 0.1, 0.1, 0.1]))
    optimizer = build_optimizer(model, ppo_config)
    metrics = update_ppo(model, optimizer, rollout, model_config, ppo_config)
    assert np.isfinite(metrics.policy_loss)
    assert np.isfinite(metrics.value_loss)
    assert metrics.clip_fraction == pytest.approx(0.0)


def test_update_ppo_wml_requires_weights() -> None:
    model, model_config, ppo_config = _tiny_model()
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    optimizer = build_optimizer(model, ppo_config)
    with pytest.raises(ValueError, match="wml_weights"):
        update_ppo(model, optimizer, rollout, model_config, ppo_config)


def test_config_validates_the_p3o_keys() -> None:
    config = load_config("config/local.json")
    resolved = config.to_dict()["ppo"]
    assert resolved["algorithm"] == "ppo"
    assert resolved["p3o_eta"] == 1.0
    assert resolved["p3o_resample_interval"] == 5
    with pytest.raises(ConfigError):
        config.with_overrides({"ppo.algorithm": "sac"})
    with pytest.raises(ConfigError):
        config.with_overrides({"ppo.p3o_eta": 0.0})
    with pytest.raises(ConfigError):
        config.with_overrides({"ppo.p3o_resample_interval": 0})
    accepted = config.with_overrides(
        {
            "ppo.algorithm": "p3o_wml",
            "ppo.p3o_eta": 2.5,
            "ppo.p3o_resample_interval": 6,
        }
    )
    assert accepted["ppo"]["algorithm"] == "p3o_wml"
    assert accepted["ppo"]["p3o_eta"] == pytest.approx(2.5)
    assert accepted["ppo"]["p3o_resample_interval"] == 6
