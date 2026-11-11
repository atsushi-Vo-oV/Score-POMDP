from __future__ import annotations

import copy
import csv
from dataclasses import fields

import numpy as np
import pytest
import torch
from test_p3o_wml import OBS_DIM, PARTICLES, _rollout_for
from test_self_imitation import artifacts, assert_tree_equal, checkpoint, tiny_config

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, load_config
from sb_pomdp.policies import ScoreBeliefActorCritic
from sb_pomdp.ppo import PPOMetrics, replay_belief_sequence, update_ppo
from sb_pomdp.prediction_schedule import PredictionSchedule
from sb_pomdp.score_curl import (
    CurlReplaySamples,
    curl_coefficient,
    curl_generator,
    curl_metric_fields,
    score_curl_statistics,
)
from sb_pomdp.seedbatch.validation import validate as validate_seedbatch
from sb_pomdp.train import METRIC_FIELDS, build_optimizer, train_single


def _belief(**overrides):
    args = {"observation_dim": 3, "state_dim": 2, "action_feature_dim": 2, "hidden_dims": [8],
            "score_parameterization": "direct", "score_clip": 0.0, "particle_clip": 0.0,
            "langevin_steps": 3, "langevin_step_size": 0.2, "particle_sampler": "svgd"}
    args.update(overrides)
    torch.manual_seed(13)
    return EnergyBelief(**args).double()


def _field(belief, initial=True):
    g = torch.Generator().manual_seed(19)
    previous = torch.randn(1, 3, 2, generator=g, dtype=torch.double, requires_grad=True)
    observation = torch.randn(1, 3, generator=g, dtype=torch.double)
    action = torch.randn(1, 2, generator=g, dtype=torch.double)
    flag = torch.tensor([initial])
    return lambda x: belief.score_at(x, observation, previous.detach(), action, flag,
                                     create_graph=True, detach_query=False), previous


@pytest.mark.parametrize("estimator", ["exact", "hutchinson", "auto"])
@pytest.mark.parametrize("matrix,expected,ratio", [
    ([[2., 1.], [1., 3.]], 0., 0.),
    ([[0., -2.], [2., 0.]], 32., 2.),
    ([[0., 0.], [0., 0.]], 0., 0.),
])
def test_known_linear_fields(estimator, matrix, expected, ratio):
    a = torch.tensor(matrix, dtype=torch.double, requires_grad=True)
    x = torch.ones(1, 1024, 2, dtype=torch.double, requires_grad=True)
    loss, measured = score_curl_statistics(lambda q: q @ a.T, x, estimator=estimator,
                                         probes=8, generator=curl_generator(7, 2))
    assert float(loss) == pytest.approx(expected, rel=0.06, abs=1e-12)
    assert float(measured) == pytest.approx(ratio, abs=1e-12)
    loss.backward()
    assert x.grad is None
    assert a.grad is not None


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("bound", [10.0, None, "log"])
def test_exact_matches_finite_difference_of_actual_bounded_score(initial, bound):
    belief = _belief(direct_score_bound=bound, initial_score_prior=0.7)
    field, previous = _field(belief, initial)
    x = torch.tensor([[[0.3, -0.8], [1.2, 0.1]]], dtype=torch.double, requires_grad=True)
    columns = []
    for axis in range(2):
        delta = torch.zeros_like(x)
        delta[..., axis] = 1e-5
        columns.append((field(x + delta) - field(x - delta)) / 2e-5)
    j = torch.stack(columns, -1)
    expected = (j - j.transpose(-1, -2)).square().sum((-1, -2)).mean()
    loss, _ = score_curl_statistics(field, x)
    torch.testing.assert_close(loss, expected, atol=1e-10, rtol=1e-7)
    loss.backward()
    assert x.grad is None and previous.grad is None
    active = belief.initial_energy if initial else belief.observation_energy
    inactive = belief.observation_energy if initial else belief.initial_energy
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in active.parameters())
    assert all(p.grad is None for p in inactive.parameters())


@pytest.mark.parametrize("initial", [False, True])
def test_energy_field_curl_is_zero(initial):
    field, _ = _field(_belief(score_parameterization="energy"), initial)
    loss, ratio = score_curl_statistics(field, torch.ones(1, 3, 2, dtype=torch.double))
    assert loss.item() < 1e-25
    assert ratio.item() < 1e-10


def test_hutchinson_converges_in_value_and_parameter_gradient():
    g = curl_generator(12, 4)
    a = torch.randn(6, 6, generator=g, dtype=torch.double, requires_grad=True)
    x = torch.randn(1, 256, 6, generator=g, dtype=torch.double)
    exact, ratio = score_curl_statistics(lambda q: q @ a.T, x, estimator="exact")
    expected_grad = torch.autograd.grad(exact, a)[0]
    estimated, measured = score_curl_statistics(lambda q: q @ a.T, x, estimator="hutchinson",
                                              probes=128, generator=g)
    actual_grad = torch.autograd.grad(estimated, a)[0]
    torch.testing.assert_close(estimated, exact, rtol=0.035, atol=0)
    assert (actual_grad - expected_grad).norm() / expected_grad.norm() < 0.04
    torch.testing.assert_close(measured, ratio, rtol=0, atol=0)


def test_curl_training_reduces_ratio_on_both_score_networks():
    belief = _belief()
    fields_ = [_field(belief, initial)[0] for initial in (True, False)]
    positions = torch.randn(1, 8, 2, generator=curl_generator(1, 1), dtype=torch.double)
    optimizer = torch.optim.Adam(belief.parameters(), lr=0.02)
    before = [score_curl_statistics(f, positions)[1].item() for f in fields_]
    for _ in range(20):
        optimizer.zero_grad()
        sum(score_curl_statistics(f, positions)[0] for f in fields_).backward()
        optimizer.step()
    after = [score_curl_statistics(f, positions)[1].item() for f in fields_]
    assert all(a < b * 0.65 for a, b in zip(after, before))


def _model():
    config = load_config("config/local.json").with_overrides({
        "model.num_particles": PARTICLES, "model.langevin_steps": 2,
        "model.score_parameterization": "direct", "model.particle_sampler": "svgd",
        "model.prediction_interval_updates": 2, "model.prediction_phase_updates": 1,
        "model.belief_activation_checkpoint": False,
    })
    torch.manual_seed(0)
    model = ScoreBeliefActorCritic(observation_dim=OBS_DIM, state_dim=2,
        action_kind="discrete", action_feature_dim=2, discrete_actions=2,
        action_low=None, action_high=None, model_config=config["model"])
    ppo = config["ppo"]
    ppo.update(epochs=1, minibatch_size=8, sequence_microbatch_size=4)
    return model, config["model"], ppo


@pytest.mark.parametrize("objective", ["ppo", "prediction"])
def test_off_and_monitor_preserve_loss_gradients_parameters_and_rng(objective, monkeypatch):
    import sb_pomdp.ppo as module
    model, config, ppo = _model()
    rollout = _rollout_for(model, batch=2, steps=4, weights=None)
    outcomes = []
    for mode in ("missing", "off", "monitor"):
        current = copy.deepcopy(model)
        cfg = dict(config)
        if mode == "missing":
            cfg = {k: v for k, v in cfg.items() if not k.startswith("score_curl_")}
        elif mode == "monitor":
            cfg.update(score_curl_monitor=True, score_curl_samples=8)
        optimizer = build_optimizer(current, ppo)
        torch.manual_seed(91)
        generator = torch.Generator().manual_seed(23)
        with monkeypatch.context() as patch:
            if mode != "monitor":
                def forbidden(*args, **kwargs):
                    raise AssertionError("disabled curl must not be called")
                patch.setattr(module, "CurlReplaySamples", forbidden)
                patch.setattr(module, "curl_generator", forbidden)
            metrics = update_ppo(current, optimizer, rollout, cfg, ppo,
                                 objective=objective, generator=generator)
        outcomes.append((current, optimizer, metrics, torch.get_rng_state(), generator.get_state()))
    for candidate in outcomes[1:]:
        expected = outcomes[0]
        assert_tree_equal(expected[0].state_dict(), candidate[0].state_dict())
        assert_tree_equal(expected[1].state_dict(), candidate[1].state_dict())
        for p, q in zip(expected[0].parameters(), candidate[0].parameters()):
            assert_tree_equal(p.grad, q.grad)
        assert torch.equal(expected[3], candidate[3]) and torch.equal(expected[4], candidate[4])
        for f in fields(PPOMetrics):
            if not f.name.startswith("score_curl_"):
                assert getattr(expected[2], f.name) == getattr(candidate[2], f.name)


@pytest.mark.parametrize("objective", ["ppo", "prediction"])
def test_penalty_added_once_per_minibatch_outside_bptt(objective, monkeypatch):
    model, cfg, ppo = _model()
    rollout = _rollout_for(model, batch=2, steps=4, weights=None)
    reference = copy.deepcopy(model)
    ppo["max_grad_norm"] = 1e20
    cfg.update(score_curl_penalty=0.3, score_curl_samples=7)
    seen = []
    original = CurlReplaySamples.evaluate
    def observe(self, belief, **kwargs):
        assert sum(map(len, self.plan.values())) == 7
        assert all(v.grad_fn is None and not v.requires_grad for vs in self.contexts.values() for v in vs)
        seen.append(self)
        return original(self, belief, **kwargs)
    monkeypatch.setattr(CurlReplaySamples, "evaluate", observe)
    # lr=0 exposes accumulated gradients without mixing in optimiser state.
    torch.manual_seed(99)
    actual = update_ppo(model, torch.optim.SGD(model.parameters(), lr=0), rollout, cfg, ppo,
                        objective=objective, score_curl_seed=4, score_curl_update=2)
    torch.manual_seed(99)
    update_ppo(reference, torch.optim.SGD(reference.parameters(), lr=0), rollout,
               {**cfg, "score_curl_penalty": 0.0}, ppo, objective=objective)
    assert len(seen) == 1 and actual.score_curl_loss > 0 and actual.score_curl_coef == 0.3
    assert any(not torch.equal(p.grad, q.grad) for p, q in
               zip(model.belief.parameters(), reference.belief.parameters())
               if p.grad is not None and q.grad is not None)
    curl_loss, _ = original(seen[0], reference.belief, create_graph=True)
    (0.3 * curl_loss).backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        if p.grad is None or q.grad is None:
            assert p.grad is q.grad
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
def test_auxiliary_stage_is_actual_transport_input_and_bypasses_graphs(sampler, monkeypatch):
    belief = _belief(particle_sampler=sampler, langevin_step_size_learnable=True,
                     observation_anchor=True, belief_cuda_graphs=True,
                     score_implementation="analytic")
    g = curl_generator(8, 2)
    observation = torch.randn(1, 3, generator=g, dtype=torch.double)
    previous = torch.randn(1, 3, 2, generator=g, dtype=torch.double)
    action = torch.randn(1, 2, generator=g, dtype=torch.double)
    initial = torch.tensor([True])
    noise = torch.randn(1, 3, 2, generator=g, dtype=torch.double)
    increments = torch.randn(1, 3, 3, 2, generator=g, dtype=torch.double)
    actual_inputs = []
    method = "_score_at_for_branch" if sampler == "ula" else "_stein_step"
    original = getattr(belief, method)
    def record(particles, *args, **kwargs):
        actual_inputs.append(particles.detach().clone())
        return original(particles, *args, **kwargs)
    monkeypatch.setattr(belief, method, record)
    kwargs = {"initial_noise": noise, "langevin_noise": increments, "step_size": 0.2,
              "track_grad": False, "need_scores": False}
    with torch.no_grad():
        belief.particles_from_noise(observation, previous, action, initial, **kwargs)
    monkeypatch.setattr(belief, method, original)
    def forbidden(*args, **kwargs):
        raise AssertionError("auxiliary replay must not enter CUDA Graphs or compile")
    monkeypatch.setattr(belief, "_execute_transport", forbidden)
    for stage in range(3):
        with torch.no_grad():
            point, _ = belief.particles_from_noise(observation, previous, action, initial,
                                                   stop_before_step=stage, **kwargs)
        torch.testing.assert_close(point, actual_inputs[stage], rtol=0, atol=0)


def test_prefix_initial_and_recursive_samples_and_microbatch_invariance():
    from dataclasses import replace

    from sb_pomdp.buffer import BeliefReplayPrefix
    model, cfg, _ = _model()
    rollout = _rollout_for(model, batch=2, steps=4, weights=None)
    prefix = BeliefReplayPrefix(observations=rollout.observations[:2, 0],
        previous_actions=rollout.previous_actions[:2, 0], initial=torch.tensor([True, False]),
        ula_initial_noise=rollout.ula_initial_noise[:2, 0],
        ula_step_noises=rollout.ula_step_noises[:2, 0])
    initial = rollout.initial.clone()
    initial[0, 0] = False
    rollout = replace(rollout, initial=initial,
                      belief_prefixes=(prefix, rollout.belief_prefixes[1]))
    cfg.update(score_curl_penalty=0.1, score_curl_samples=128)
    samples = []
    for chunks in ([torch.tensor([0, 1])], [torch.tensor([0]), torch.tensor([1])]):
        collector = CurlReplaySamples(rollout, torch.tensor([0, 1]), cfg, curl_generator(3, 2))
        for chunk in chunks:
            replay_belief_sequence(model, rollout, chunk, cfg, curl_samples=collector)
        samples.append(collector)
    assert samples[0].plan == samples[1].plan
    assert (0, -2) in samples[0].contexts and (0, -1) in samples[0].contexts
    assert samples[0].contexts[(0, -2)][3].item() is True
    assert samples[0].contexts[(0, -1)][3].item() is False
    assert len(samples[0].plan) == 10
    for key in samples[0].contexts:
        for left, right in zip(samples[0].contexts[key], samples[1].contexts[key]):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)


def test_ramp_uses_ppo_progress_not_prediction_or_segment_progress():
    schedule = PredictionSchedule(2, 2, 1, 5)
    cfg = {"score_curl_penalty": 1.0, "score_curl_schedule": "ramp"}
    values = [curl_coefficient(cfg, ppo_updates=schedule.ppo_updates_done(u),
                               total_ppo_updates=5) for u in range(1, schedule.total_updates + 1)]
    assert values == [0, 0, 0.2, 0.4, 0.4, 0.6, 0.8, 0.8, 1.0]
    cfg["score_curl_schedule"] = "constant"
    assert curl_coefficient(cfg, ppo_updates=0, total_ppo_updates=5) == 1.0


@pytest.mark.parametrize("overrides", [
    {"score_curl_penalty": -1}, {"score_curl_penalty": float("nan")},
    {"score_curl_penalty": float("inf")}, {"score_curl_penalty": True},
    {"score_curl_samples": 0}, {"score_curl_probes": 0}, {"score_curl_samples": 1.2},
    {"score_curl_estimator": "bad"}, {"score_curl_schedule": "bad"},
    {"score_curl_monitor": 1},
    {"score_curl_penalty": 0.1, "score_parameterization": "energy"},
    {"score_curl_monitor": True, "score_parameterization": "energy"},
])
def test_bad_config_is_rejected(overrides):
    with pytest.raises(ConfigError):
        load_config("config/trysvgdlight_mirror2_tf-svgd.json").with_overrides(
            {f"model.{key}": value for key, value in overrides.items()})


@pytest.mark.parametrize("setting", ["score_curl_penalty", "score_curl_monitor"])
def test_seedbatch_fails_explicitly(setting):
    from sb_pomdp.seedbatch.backend import SeedBatchBackend
    cfg = load_config("config/trysvgdlight_mirror2_tf-svgd.json").to_dict()
    cfg["model"][setting] = True if setting.endswith("monitor") else 0.1
    with pytest.raises(ConfigError, match="seedbatch does not support score_curl"):
        validate_seedbatch(cfg)
    with pytest.raises(ConfigError, match="seedbatch does not support score_curl"):
        SeedBatchBackend([], [], cfg["model"], cfg["ppo"])


def test_monitor_csv_keeps_legacy_columns_and_identical_training(tmp_path):
    cfg = tiny_config(tmp_path, enabled=False, horizon=5)
    states, rows = [], []
    for monitor in (False, True):
        current = cfg.with_overrides({"model.score_curl_monitor": monitor})
        output = artifacts(tmp_path / str(monitor))
        train_single(current, task="disk_tag", seed=3, artifacts=output,
                     evaluate_at_end=False, verbose=False)
        states.append(checkpoint(output))
        with output.metrics_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows.append(list(reader))
            assert reader.fieldnames == list(METRIC_FIELDS + curl_metric_fields(current["model"]))
    for key in ("model", "optimizer", "torch_rng_state", "python_rng_state"):
        assert_tree_equal(states[0][key], states[1][key])
    assert np.array_equal(states[0]["numpy_rng_state"][1], states[1]["numpy_rng_state"][1])
    for left, right in zip(*rows):
        for key in METRIC_FIELDS:
            if key != "steps_per_second":
                assert left[key] == right[key]
        assert float(right["score_curl_ratio"]) > 0


def test_ramp_training_prediction_and_segment_resume(tmp_path):
    cfg = tiny_config(tmp_path, enabled=False, total=2, horizon=7).with_overrides({
        "model.score_curl_penalty": 0.1, "model.score_curl_schedule": "ramp",
        "model.score_curl_samples": 3, "model.prediction_pretrain_steps": 4,
        "model.prediction_interval_updates": 1, "model.prediction_phase_updates": 1,
        "model.score_implementation": "analytic", "model.belief_cuda_graphs": True,
    })
    full, split = artifacts(tmp_path / "full"), artifacts(tmp_path / "split")
    args = {"task": "disk_tag", "seed": 3, "evaluate_at_end": False, "verbose": False}
    train_single(cfg, artifacts=full, **args)
    train_single(cfg, artifacts=split, segment_updates=2, **args)
    train_single(cfg, artifacts=split, resume_checkpoint=split.checkpoint_path(tag="latest"), **args)
    expected, actual = checkpoint(full), checkpoint(split)
    for key in ("model", "optimizer", "torch_rng_state"):
        assert_tree_equal(expected[key], actual[key])
    rows = expected["trainer_state"]["metric_rows"]
    assert [r["score_curl_coef"] for r in rows] == [0, 0.05, 0.05, 0.1]
    for a, b in zip(rows, actual["trainer_state"]["metric_rows"]):
        for key in curl_metric_fields(cfg["model"]):
            assert np.isfinite(a[key]) and a[key] == b[key]
