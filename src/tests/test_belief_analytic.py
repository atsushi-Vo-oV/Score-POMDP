from __future__ import annotations

import copy
from dataclasses import fields

import pytest
import torch
from test_speedup_exactness import _fixture, _grads

from sb_pomdp import ppo
from sb_pomdp.belief import EnergyBelief
from sb_pomdp.policies import ObservationPredictor


def _close(a, b, tol):
    assert (a is None) == (b is None)
    if a is not None:
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("parameterization", ["energy", "direct", "direct_joint"])
@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
@pytest.mark.parametrize("branch", ["initial", "recursive", "mixed"])
def test_analytic_transport_matches_autograd(dtype, parameterization, sampler, branch, schedule="temperature"):
    torch.manual_seed(421)
    options = dict(observation_dim=2, state_dim=2, action_feature_dim=2, hidden_dims=[7, 5],
                   score_parameterization=parameterization, particle_sampler=sampler,
                   score_clip=0.4, particle_clip=0.8, langevin_warm_start=True,
                   transition_proposal=True, observation_anchor=True, proposal_hidden=[4],
                   langevin_step_size_learnable=True,
                   langevin_step_size=0.03, langevin_steps=2,
                   **({"langevin_temperature_learnable": True} if sampler == "ula" and schedule == "temperature" else {"langevin_noise_scale": 0.1, "langevin_noise_scale_learnable": True} if sampler == "ula" else {}))
    reference = EnergyBelief(**options, score_implementation="autograd").to(dtype=dtype)
    with torch.no_grad():
        for module in (reference.observation_anchor, reference.transition_proposal):
            for parameter in module.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.1)
    analytic = EnergyBelief(**options, score_implementation="analytic").to(dtype=dtype)
    analytic.load_state_dict(reference.state_dict())
    initial = {"initial": [True, True], "recursive": [False, False],
               "mixed": [True, False]}[branch]
    data = [torch.randn(2, 2, dtype=dtype), torch.randn(2, 3, 2, dtype=dtype),
            torch.randn(2, 2, dtype=dtype)]
    initial_noise = torch.randn(2, 3, 2, dtype=dtype)
    step_noise = torch.randn(2, 2, 3, 2, dtype=dtype)
    results = []
    rng = torch.random.get_rng_state().clone()
    for model in (reference, analytic):
        inputs = [x.detach().clone().requires_grad_() for x in data]
        particles, scores = model.particles_from_noise(
            *inputs, torch.tensor(initial), initial_noise=initial_noise, langevin_noise=step_noise,
            step_size=0.03, track_grad=True, potential_branch=branch)
        (particles.square().sum() + scores.square().sum()).backward()
        results.append((particles.detach(), scores.detach(), [x.grad for x in inputs], _grads(model)))
        assert torch.equal(rng, torch.random.get_rng_state())
    tol = 1e-10 if dtype == torch.float64 else 1e-6
    for a, b in zip(results[0][:2], results[1][:2], strict=True):
        _close(a, b, tol)
    grad_tol = 1e-10 if dtype == torch.float64 else 1e-5
    for a, b in zip(results[0][2], results[1][2], strict=True):
        _close(a, b, grad_tol)
    for name in results[0][3]:
        _close(results[0][3][name], results[1][3][name], grad_tol)
    with torch.no_grad():
        outputs = [m.particles_from_noise(*data, torch.tensor(initial), initial_noise=initial_noise,
                   langevin_noise=step_noise, step_size=0.03, track_grad=False,
                   potential_branch=branch) for m in (reference, analytic)]
    for a, b in zip(*outputs, strict=True):
        _close(a, b, tol)
        assert not a.requires_grad and not b.requires_grad
    assert torch.equal(rng, torch.random.get_rng_state())


def test_learned_noise_schedule_matches_autograd():
    test_analytic_transport_matches_autograd(torch.float64, "energy", "ula", "mixed", "noise")


@pytest.mark.parametrize("warm_adam", [False, True])
@pytest.mark.parametrize("objective", ["ppo", "prediction"])
@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
def test_update_matches_autograd(objective, sampler, warm_adam):
    model, config, rollout = _fixture("energy", sampler, continuous=True)
    model.prediction_head = "belief_mixture"
    model.prediction_predictor = ObservationPredictor(2, 2, 2, [8])
    other = copy.deepcopy(model)
    other.belief.score_implementation = "analytic"
    optimizers = [torch.optim.Adam(m.parameters(), lr=0.00025) for m in (model, other)]
    # Identical nonzero moments model a resumed optimizer and avoid first-step
    # amplification of roundoff in near-zero gradients.
    if warm_adam:
        for m, optimizer in zip((model, other), optimizers, strict=True):
            for parameter in m.parameters():
                parameter.grad = torch.full_like(parameter, 0.01)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    settings = {"minibatch_size": 9, "sequence_microbatch_size": 9, "normalize_advantage": False,
                "epochs": 1, "clip_coef": 0.2, "value_coef": 0.5, "entropy_coef": 0.01,
                "max_grad_norm": 100.0}
    rng = torch.random.get_rng_state().clone()
    metrics = []
    generators = []
    for m, optimizer in zip((model, other), optimizers, strict=True):
        torch.random.set_rng_state(rng)
        generator = torch.Generator().manual_seed(31)
        metrics.append(ppo.update_ppo(m, optimizer, rollout, config, settings,
                                      generator=generator, objective=objective))
        generators.append(generator)
    assert torch.equal(generators[0].get_state(), generators[1].get_state())
    for field in fields(metrics[0]):
        _close(torch.tensor(getattr(metrics[0], field.name)),
               torch.tensor(getattr(metrics[1], field.name)), 1e-5)
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters(), strict=True):
        try:
            torch.testing.assert_close(p, q, atol=1e-4, rtol=1e-5)
        except AssertionError as error:
            raise AssertionError(f"{name}: grads {p.grad} vs {q.grad}") from error
        _close(p.grad, q.grad, 1e-5)
        for key in optimizers[0].state[p]:
            _close(optimizers[0].state[p][key], optimizers[1].state[q][key], 1e-5)






