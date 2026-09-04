"""Score v2: learned chain proposals, encoder score switch, and reward supervision."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_p3o_wml import OBS_DIM, PARTICLES, STATE_DIM, STEPS, _rollout_for

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.networks import BeliefSetEncoder, DeepSetsBeliefEncoder, RewardPredictor
from sb_pomdp.policies import ScoreBeliefActorCritic
from sb_pomdp.ppo import reward_prediction_loss, update_ppo
from sb_pomdp.train import build_optimizer

ACTION_DIM = 2


def _belief(**overrides: object) -> EnergyBelief:
    kwargs: dict[str, object] = {
        "score_clip": 0.0,
        "particle_clip": 0.0,
    }
    kwargs.update(overrides)
    torch.manual_seed(0)
    return EnergyBelief(OBS_DIM, STATE_DIM, ACTION_DIM, [16, 16], **kwargs)  # type: ignore[arg-type]


def _inputs(batch: int = 4, *, mixed: bool = True):
    torch.manual_seed(3)
    observation = torch.randn(batch, OBS_DIM)
    previous = torch.randn(batch, PARTICLES, STATE_DIM)
    action = torch.randn(batch, ACTION_DIM)
    initial = torch.zeros(batch, dtype=torch.bool)
    if mixed:
        initial[: batch // 2] = True
    initial_noise = torch.randn(batch, PARTICLES, STATE_DIM)
    step_noise = torch.randn(batch, STEPS, PARTICLES, STATE_DIM)
    return observation, previous, action, initial, initial_noise, step_noise


def _run(belief: EnergyBelief, inputs, *, step_size: float = 0.02) -> torch.Tensor:
    observation, previous, action, initial, initial_noise, step_noise = inputs
    particles, _ = belief.particles_from_noise(
        observation,
        previous,
        action,
        initial,
        initial_noise=initial_noise,
        langevin_noise=step_noise,
        step_size=step_size,
        track_grad=False,
    )
    return particles


def _copy_energies(source: EnergyBelief, target: EnergyBelief) -> None:
    for name in ("initial_energy", "observation_energy", "transition_energy"):
        getattr(target, name).load_state_dict(getattr(source, name).state_dict())


def test_zero_initialised_proposals_are_bit_identical_to_the_plain_samplers() -> None:
    inputs = _inputs()
    cold = _belief()
    cold_anchor = _belief(observation_anchor=True)
    _copy_energies(cold, cold_anchor)
    assert torch.equal(_run(cold, inputs), _run(cold_anchor, inputs))

    warm = _belief(langevin_warm_start=True)
    warm_full = _belief(langevin_warm_start=True, transition_proposal=True, observation_anchor=True)
    _copy_energies(warm, warm_full)
    assert torch.equal(_run(warm, inputs), _run(warm_full, inputs))


def test_transition_proposal_moves_only_recursive_rows() -> None:
    inputs = _inputs()
    initial = inputs[3]
    belief = _belief(langevin_warm_start=True, transition_proposal=True)
    reference = _run(belief, inputs, step_size=1e-9)
    with torch.no_grad():
        last = [m for m in belief.transition_proposal.modules() if isinstance(m, torch.nn.Linear)][
            -1
        ]
        last.bias.fill_(0.7)
    shifted = _run(belief, inputs, step_size=1e-9)
    delta = shifted - reference
    assert torch.allclose(delta[initial], torch.zeros_like(delta[initial]), atol=1e-6)
    assert torch.allclose(delta[~initial], torch.full_like(delta[~initial], 0.7), atol=1e-4)


def test_observation_anchor_moves_only_initial_rows() -> None:
    inputs = _inputs()
    _, _, _, initial, _, _ = inputs
    belief = _belief(observation_anchor=True)
    reference = _run(belief, inputs, step_size=1e-9)
    with torch.no_grad():
        last = [m for m in belief.observation_anchor.modules() if isinstance(m, torch.nn.Linear)][
            -1
        ]
        last.bias[:STATE_DIM].fill_(-1.5)  # mean shift; log-scale stays at zero
    shifted = _run(belief, inputs, step_size=1e-9)
    delta = shifted - reference
    assert torch.allclose(delta[initial], torch.full_like(delta[initial], -1.5), atol=1e-4)
    assert torch.allclose(delta[~initial], torch.zeros_like(delta[~initial]), atol=1e-6)


def test_transition_proposal_requires_warm_start() -> None:
    with pytest.raises(ValueError):
        _belief(transition_proposal=True)
    config = load_config("config/local.json")
    with pytest.raises(ConfigError):
        config.with_overrides({"model.langevin_transition_proposal": True})
    accepted = config.with_overrides(
        {
            "model.langevin_transition_proposal": True,
            "model.langevin_warm_start": True,
            "model.langevin_observation_anchor": True,
            "model.encoder_use_scores": False,
            "model.reward_prediction_coef": 0.5,
        }
    )
    resolved = accepted.to_dict()["model"]
    assert resolved["langevin_transition_proposal"] is True
    assert resolved["encoder_use_scores"] is False
    assert config.to_dict()["model"]["encoder_use_scores"] is True
    assert config.to_dict()["model"]["reward_prediction_coef"] == 0.0


def test_encoders_accept_positions_only() -> None:
    torch.manual_seed(0)
    particles = torch.randn(3, PARTICLES, STATE_DIM)
    scores = torch.randn(3, PARTICLES, STATE_DIM)
    transformer = BeliefSetEncoder(STATE_DIM, 8, 2, 1, 16, 0.0, use_scores=False)
    deep_sets = DeepSetsBeliefEncoder(STATE_DIM, 8, 16, 0.0, use_scores=False)
    for encoder in (transformer, deep_sets):
        a = encoder(particles, scores)
        b = encoder(particles, torch.zeros_like(scores))
        assert a.shape == (3, 8) and torch.allclose(a, b)
    with_scores = BeliefSetEncoder(STATE_DIM, 8, 2, 1, 16, 0.0)
    assert not torch.allclose(
        with_scores(particles, scores), with_scores(particles, torch.zeros_like(scores))
    )


def test_reward_predictor_likelihood_and_preference() -> None:
    torch.manual_seed(1)
    predictor = RewardPredictor(1, 0, [32])
    assert predictor.log_likelihood(
        torch.randn(5, 6, 1), torch.zeros(5, 0), torch.randn(5)
    ).shape == (5,)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-2)
    for _ in range(300):
        state = torch.randn(64, 1)
        particles = state[:, None, :] + 0.01 * torch.randn(64, 4, 1)
        reward = state.squeeze(-1).square() + 0.05 * torch.randn(64)
        loss = -predictor.log_likelihood(particles, torch.zeros(64, 0), reward).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    state = torch.randn(256, 1)
    reward = state.squeeze(-1).square()
    informative = predictor.log_likelihood(
        state[:, None, :].repeat(1, 4, 1), torch.zeros(256, 0), reward
    ).mean()
    blob = predictor.log_likelihood(torch.randn(256, 4, 1), torch.zeros(256, 0), reward).mean()
    assert float(informative) > float(blob) + 0.5


def _v2_model(**extra: object) -> tuple[ScoreBeliefActorCritic, dict, dict]:
    overrides: dict[str, object] = {
        "model.num_particles": PARTICLES,
        "model.langevin_steps": STEPS,
        "model.langevin_warm_start": True,
        "model.langevin_transition_proposal": True,
        "model.langevin_observation_anchor": True,
        "model.reward_prediction_coef": 1.0,
        "model.proposal_hidden": [8],
        "model.reward_predictor_hidden": [8],
    }
    overrides.update(extra)
    config = load_config("config/local.json").with_overrides(overrides)
    model_config = dict(config["model"])
    ppo_config = dict(config["ppo"])
    ppo_config.update({"epochs": 1, "minibatch_size": 8, "sequence_microbatch_size": 4})
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


def test_update_ppo_trains_proposals_anchor_and_reward_head() -> None:
    model, model_config, ppo_config = _v2_model()
    assert model.reward_predictor is not None
    assert model.belief.transition_proposal is not None
    assert model.belief.observation_anchor is not None
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    optimizer = build_optimizer(model, ppo_config)
    watched = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(
            ("belief.transition_proposal", "belief.observation_anchor", "reward_predictor")
        )
    }
    assert watched
    metrics = update_ppo(model, optimizer, rollout, model_config, ppo_config)
    assert np.isfinite(metrics.policy_loss)
    moved = {
        name: float((parameter - watched[name]).abs().max())
        for name, parameter in model.named_parameters()
        if name in watched
    }
    for prefix in ("belief.transition_proposal", "belief.observation_anchor", "reward_predictor"):
        assert max(v for n, v in moved.items() if n.startswith(prefix)) > 0.0, prefix


def test_reward_loss_masks_episode_boundaries_and_short_sequences() -> None:
    model, _, _ = _v2_model()
    rollout = _rollout_for(model, batch=4, steps=4, weights=None)
    indices = torch.arange(4)
    particles = rollout.particles[:, indices].clone().requires_grad_(True)
    loss = reward_prediction_loss(model, particles, rollout, indices)
    assert torch.isfinite(loss)
    loss.backward()
    grad = particles.grad
    assert grad is not None
    # The final step has no successor action, so it carries no gradient.
    assert float(grad[-1].abs().sum()) == 0.0
    assert float(grad[:-1].abs().sum()) > 0.0
    assert float(reward_prediction_loss(model, particles[:1], rollout, indices)) == 0.0


def test_configured_alpha_and_transformer_models_run_with_positions_only() -> None:
    for encoder in ("transformer", "deep_sets"):
        model, _, _ = _v2_model(
            **{"model.encoder_kind": encoder, "model.encoder_use_scores": False}
        )
        rollout = _rollout_for(model, batch=4, steps=3, weights=None)
        condition = model.encode(
            rollout.particles.reshape(-1, PARTICLES, STATE_DIM),
            torch.zeros(12, PARTICLES, STATE_DIM),
        )
        assert condition.shape[0] == 12
