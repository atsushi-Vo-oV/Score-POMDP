from types import SimpleNamespace

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.train import BeliefContext, BeliefReplayHistory, _recondition_belief_context


def _sequential(model, context, histories):
    rebuilt = context.previous_particles.detach().clone()
    for index, history in enumerate(histories):
        if not history.observations:
            if not bool(context.initial[index]):
                raise RuntimeError("a non-initial belief context is missing its episode history")
            rebuilt[index].zero_()
            continue
        particles = torch.zeros_like(rebuilt[index : index + 1])
        for step in range(len(history.observations)):
            particles, _ = model.belief.particles_from_noise(
                history.observations[step][None],
                particles,
                history.previous_actions[step][None],
                history.initial[step][None],
                initial_noise=history.ula_initial_noise[step][None],
                langevin_noise=history.ula_step_noises[step][None],
                step_size=0.1,
                track_grad=False,
            )
        rebuilt[index] = particles[0]
    return rebuilt


@pytest.mark.parametrize("parameterization", ["energy", "direct", "direct_joint"])
@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
def test_recondition_matches_sequential(parameterization, sampler):
    torch.manual_seed(12)
    model = SimpleNamespace(
        belief=EnergyBelief(
            2,
            2,
            2,
            [8],
            score_parameterization=parameterization,
            particle_sampler=sampler,
            langevin_warm_start=True,
        )
    )
    context = BeliefContext(
        torch.randn(4, 2),
        torch.randn(4, 3, 2),
        torch.randn(4, 2),
        torch.tensor([False, False, True, False]),
    )
    histories = []
    for length in (3, 1, 0, 2):
        history = BeliefReplayHistory()
        for step in range(length):
            history.observations.append(torch.randn(2))
            history.previous_actions.append(torch.randn(2))
            history.initial.append(torch.tensor(step == 0))
            history.ula_initial_noise.append(torch.randn(3, 2))
            history.ula_step_noises.append(torch.randn(2, 3, 2))
        histories.append(history)
    state = torch.get_rng_state().clone()
    expected = _sequential(model, context, histories)
    assert torch.equal(torch.get_rng_state(), state)
    actual = _recondition_belief_context(model, context, histories, {"langevin_step_size": 0.1})
    assert torch.equal(torch.get_rng_state(), state)
    torch.testing.assert_close(actual.previous_particles, expected, rtol=1e-6, atol=1e-6)
    assert not actual.previous_particles.requires_grad
    context.initial[2] = False
    with pytest.raises(RuntimeError, match="missing its episode history"):
        _recondition_belief_context(model, context, histories, {"langevin_step_size": 0.1})
