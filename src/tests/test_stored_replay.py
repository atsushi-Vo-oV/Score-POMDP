"""model.belief_replay=stored: one-step replay of all (time, environment) rows at once (experiment 28)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.buffer import BeliefReplayPrefix, RolloutBatch
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.ppo import belief_replay_mode, replay_belief_sequence

STEPS, ENVIRONMENTS, PARTICLES = 5, 3, 4
CONFIG = {"belief_gradient_mode": "tbptt_1", "langevin_step_size": 0.2, "belief_activation_checkpoint": False}


def _belief() -> EnergyBelief:
    torch.manual_seed(3)
    return EnergyBelief(
        observation_dim=3, state_dim=2, action_feature_dim=2, hidden_dims=[8], score_clip=0.0, particle_clip=0.0,
        langevin_step_size_learnable=True, langevin_step_size=0.2, langevin_steps=3, observation_anchor=True,
        particle_sampler="svgd", score_parameterization="direct",
    )


def _rollout(belief: EnergyBelief) -> RolloutBatch:
    """Collect a rollout with the belief itself; environment 1 starts a new episode at t = 2."""
    g = torch.Generator().manual_seed(17)
    observations = torch.randn(STEPS, ENVIRONMENTS, 3, generator=g)
    actions = torch.randn(STEPS, ENVIRONMENTS, 2, generator=g)
    initial = torch.zeros(STEPS, ENVIRONMENTS, dtype=torch.bool)
    initial[0] = True
    initial[2, 1] = True
    initial_noise = torch.randn(STEPS, ENVIRONMENTS, PARTICLES, 2, generator=g)
    step_noises = torch.randn(STEPS, ENVIRONMENTS, 3, PARTICLES, 2, generator=g)
    incoming, outgoing = [], []
    particles = torch.zeros(ENVIRONMENTS, PARTICLES, 2)
    for t in range(STEPS):
        incoming.append(particles.clone())  # stored as collected: not zeroed at the episode start
        start = torch.where(initial[t, :, None, None], torch.zeros_like(particles), particles)
        particles, _ = belief.particles_from_noise(
            observations[t], start, actions[t], initial[t], initial_noise=initial_noise[t],
            langevin_noise=step_noises[t], step_size=0.2, track_grad=False,
        )
        outgoing.append(particles.clone())
    empty = BeliefReplayPrefix(
        observations=torch.zeros(0, 3), previous_actions=torch.zeros(0, 2), initial=torch.zeros(0, dtype=torch.bool),
        ula_initial_noise=torch.zeros(0, PARTICLES, 2), ula_step_noises=torch.zeros(0, 3, PARTICLES, 2),
    )
    zeros = torch.zeros(STEPS, ENVIRONMENTS)
    return RolloutBatch(
        observations=observations, previous_particles=torch.stack(incoming), previous_actions=actions, initial=initial,
        particles=torch.stack(outgoing), ula_initial_noise=initial_noise, ula_step_noises=step_noises,
        actions=actions, old_log_prob=zeros, old_values=zeros, rewards=zeros, dones=torch.zeros(STEPS, ENVIRONMENTS, dtype=torch.bool),
        advantages=zeros, returns=zeros, diffusion_chains=None, pre_tanh_actions=None, belief_prefixes=(empty,) * ENVIRONMENTS,
    )


def _gradients(belief: EnergyBelief, particles: torch.Tensor, scores: torch.Tensor) -> list[torch.Tensor]:
    weight = torch.linspace(0.5, 1.5, particles.numel()).reshape(particles.shape)
    loss = (particles * weight).sum() + scores.square().sum()
    parameters = [p for p in belief.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, grads)]


@pytest.mark.parametrize("subset", [(0, 1, 2), (2, 0)])
def test_stored_replay_equals_the_one_step_sequence_replay_before_any_update(subset):
    belief = _belief()
    model = SimpleNamespace(belief=belief)
    rollout = _rollout(belief)
    indices = torch.tensor(subset)
    sequence = replay_belief_sequence(model, rollout, indices, CONFIG)
    stored = replay_belief_sequence(model, rollout, indices, {**CONFIG, "belief_replay": "stored"})
    assert stored[0].shape == sequence[0].shape == (STEPS, len(subset), PARTICLES, 2)
    torch.testing.assert_close(stored[0], sequence[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(stored[1], sequence[1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(stored[0], rollout.particles[:, indices], atol=1e-6, rtol=1e-5)  # and the collected ones
    stored_gradients = _gradients(belief, *stored)
    for a, b in zip(stored_gradients, _gradients(belief, *sequence)):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    full = replay_belief_sequence(model, rollout, indices, {**CONFIG, "belief_gradient_mode": "full"})
    different = [not torch.allclose(a, b, atol=1e-6) for a, b in zip(stored_gradients, _gradients(belief, *full))]
    assert any(different)  # the full replay does send gradients across time


def test_after_an_update_the_stored_replay_keeps_the_collected_incoming_particles():
    belief = _belief()
    model = SimpleNamespace(belief=belief)
    rollout = _rollout(belief)
    indices = torch.arange(ENVIRONMENTS)
    with torch.no_grad():
        for parameter in belief.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    sequence, _ = replay_belief_sequence(model, rollout, indices, CONFIG)
    stored, _ = replay_belief_sequence(model, rollout, indices, {**CONFIG, "belief_replay": "stored"})
    torch.testing.assert_close(stored[0], sequence[0], atol=1e-6, rtol=1e-5)  # t = 0: nothing comes in
    torch.testing.assert_close(stored[2, 1], sequence[2, 1], atol=1e-6, rtol=1e-5)  # an episode start inside the rollout
    assert not torch.allclose(stored[1], sequence[1], atol=1e-4)  # recursive rows start from the old particles
    expected, _ = belief.particles_from_noise(
        rollout.observations[3], rollout.previous_particles[3], rollout.previous_actions[3], rollout.initial[3],
        initial_noise=rollout.ula_initial_noise[3], langevin_noise=rollout.ula_step_noises[3], step_size=0.2, track_grad=False,
    )
    torch.testing.assert_close(stored[3], expected, atol=1e-6, rtol=1e-5)


def test_replay_mode_per_objective():
    base = {"belief_gradient_mode": "full"}
    assert belief_replay_mode(base) == belief_replay_mode(base, "prediction") == "full"
    one = {"belief_gradient_mode": "tbptt_1", "belief_replay": "stored"}
    assert belief_replay_mode(one) == belief_replay_mode(one, "critic") == belief_replay_mode(one, "prediction") == "stored"
    mixed = {**one, "prediction_replay": "full"}
    assert belief_replay_mode(mixed) == "stored" and belief_replay_mode(mixed, "prediction") == "full"
    assert belief_replay_mode({"belief_gradient_mode": "full", "prediction_replay": "tbptt_1"}, "prediction") == "tbptt_1"
    with pytest.raises(ValueError):
        replay_belief_sequence(SimpleNamespace(belief=_belief()), _rollout(_belief()), torch.tensor([0]), CONFIG, replay="half")


def test_config_validation():
    base = load_config("config/trysvgdlight2_mirror2_tf-svgd.json")
    model = base.to_dict()["model"]
    assert (model["belief_replay"], model["prediction_replay"]) == ("sequence", "same")
    with pytest.raises(ConfigError):
        base.with_overrides({"model.belief_replay": "stored"})  # needs one-step gradients
    with pytest.raises(ConfigError):
        base.with_overrides({"model.belief_replay": "parallel"})
    with pytest.raises(ConfigError):
        base.with_overrides({"model.prediction_replay": "stale"})
    one_step = {"model.belief_gradient_mode": "tbptt_1", "comparison.belief_gradient_modes": ["tbptt_1"]}
    stored = base.with_overrides({**one_step, "model.belief_replay": "stored", "model.prediction_replay": "full"}).to_dict()["model"]
    assert (stored["belief_replay"], stored["prediction_replay"]) == ("stored", "full")
    with pytest.raises(ConfigError):
        base.with_overrides({**one_step, "model.belief_replay": "stored", "model.score_curl_monitor": True})


def _generator():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("bptt1_generate", Path("scripts/bptt1_generate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_arms_differ_from_their_default_only_in_the_replay_keys():
    from pathlib import Path

    assets = list(_generator().assets())
    assert len(assets) == 24
    for name, source, label, campaign, stem, keys, updates, _ in assets:
        arm, base = load_config(f"config/{name}.json").to_dict(), load_config(f"config/{source}.json").to_dict()
        assert arm["model"]["belief_gradient_mode"] == "tbptt_1" and arm["comparison"]["belief_gradient_modes"] == ["tbptt_1"]
        assert base["model"]["belief_gradient_mode"] == "full" and base["model"]["belief_replay"] == "sequence"
        expected = ("stored" if "stored" in name else "sequence", "full" if name.endswith("predfull") else "same")
        assert (arm["model"]["belief_replay"], arm["model"]["prediction_replay"]) == expected
        for key in ("belief_gradient_mode", "belief_replay", "prediction_replay"):
            arm["model"][key] = base["model"][key]
        arm["comparison"]["belief_gradient_modes"] = base["comparison"]["belief_gradient_modes"]
        arm["experiment"]["label"] = base["experiment"]["label"]
        assert arm == base
        for seed in range(10, 15):
            text = Path(f"jobs/bptt1/{stem}_s{seed}.sh").read_text(encoding="utf-8")
            assert f'SB_POMDP_CONFIG="config/{name}.json"' in text and f'SB_POMDP_CAMPAIGN_ID="{campaign}"' in text
            assert f"SB_POMDP_SHARD_INDEX={seed - 9}" in text and f"SB_POMDP_SEGMENT_UPDATES={updates}" in text
            assert text.rstrip().endswith("exec bash jobs/genkai_production.sh")
