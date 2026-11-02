"""Full-graph transport and execution-only configuration checks."""
import copy

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError, load_config, resume_comparable_config, validate_config


@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
@pytest.mark.parametrize("branch", ["initial", "recursive", "mixed"])
def test_compiled_fullgraph_matches_eager_and_reuses_graph(monkeypatch, sampler, branch):
    torch.manual_seed(10)
    eager = EnergyBelief(2, 2, 2, [4], particle_sampler=sampler, score_implementation="analytic")
    compiled = copy.deepcopy(eager)
    compiled.belief_compile = "inductor"
    builds = []
    def backend(graph, example_inputs):
        builds.append(graph)
        return torch._dynamo.lookup_backend("aot_eager")(graph, example_inputs)
    original = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: original(fn, backend=backend, fullgraph=True, dynamic=False))
    flags = {"initial": [True, True], "recursive": [False, False], "mixed": [True, False]}[branch]
    obs = torch.randn(2, 2)
    prev = torch.randn(2, 3, 2)
    action = torch.randn(2, 2)
    initial_noise = torch.randn(2, 3, 2)
    noise = torch.randn(2, 2, 3, 2)
    def run(model):
        model.zero_grad(set_to_none=True)
        x = prev.clone().requires_grad_()
        particles, scores = model.particles_from_noise(obs, x, action, torch.tensor(flags), initial_noise=initial_noise,
            langevin_noise=noise, step_size=0.03, track_grad=True, potential_branch=branch)
        (particles.square().sum() + scores.square().sum()).backward()
        return particles, scores, x.grad, [p.grad for p in model.parameters()]
    ref = run(eager)
    actual = run(compiled)
    for a, b in zip(ref[:2], actual[:2]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ref[2], actual[2], atol=1e-5, rtol=1e-5)
    for a, b in zip(ref[3], actual[3]):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    count = len(builds)
    counters = dict(torch._dynamo.utils.counters["stats"])
    assert count == 1
    run(compiled)
    assert len(builds) == count
    assert dict(torch._dynamo.utils.counters["stats"]) == counters


def test_compile_config_validation_and_resume():
    config = load_config("config/core6_mirror_b02.json").to_dict()
    assert config["model"]["score_implementation"] == "autograd"
    assert config["model"]["belief_compile"] == "off"
    assert config["model"]["belief_cuda_graphs"] is False
    changed = copy.deepcopy(config)
    changed["model"].update(score_implementation="analytic", belief_compile="inductor")
    validate_config(changed)
    assert resume_comparable_config(config) == resume_comparable_config(changed)
    changed["model"]["belief_cuda_graphs"] = True
    with pytest.raises(ConfigError, match="alternatives.*parameter gradients"):
        validate_config(changed)
    changed["model"]["belief_compile"] = "off"
    validate_config(changed)
    assert resume_comparable_config(config) == resume_comparable_config(changed)
    changed["model"].update(belief_compile="inductor", belief_cuda_graphs=False)
    changed["model"]["score_implementation"] = "autograd"
    with pytest.raises(ConfigError, match="requires"):
        validate_config(changed)
    for key, value in [("score_implementation", "bad"), ("belief_compile", "bad"), ("belief_cuda_graphs", "true")]:
        changed = copy.deepcopy(config)
        changed["model"][key] = value
        with pytest.raises(ConfigError):
            validate_config(changed)


def test_cache_distinguishes_incoming_gradient_state(monkeypatch):
    model = EnergyBelief(2, 2, 2, [4], score_implementation="analytic", belief_compile="inductor")
    builds = []
    original = torch.compile
    def backend(graph, example_inputs):
        builds.append(graph)
        return torch._dynamo.lookup_backend("aot_eager")(graph, example_inputs)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: original(fn, backend=backend, fullgraph=True, dynamic=False))
    obs = torch.randn(1, 2)
    prev = torch.randn(1, 3, 2)
    noise = torch.randn(1, 1, 3, 2)
    def call(value):
        return model.particles_from_noise(obs, value, obs, torch.tensor([False]), initial_noise=prev,
                                          langevin_noise=noise, step_size=0.03, track_grad=True)
    call(prev)
    next_state = prev.clone().requires_grad_()
    call(next_state)
    count = len(builds)
    call(next_state)
    call(prev)
    assert len(builds) == count
    assert len(model._compiled_chains) == 2


def test_inductor_cpu_or_explicit_compiler_skip():
    model = EnergyBelief(2, 2, 2, [4], score_implementation="analytic", belief_compile="inductor")
    obs = torch.randn(1, 2)
    prev = torch.randn(1, 2, 2)
    noise = torch.randn(1, 1, 2, 2)
    import warnings
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        out = model.particles_from_noise(obs, prev, obs, torch.tensor([False]), initial_noise=prev,
                                         langevin_noise=noise, step_size=0.03, track_grad=True)
    if model._compile_unavailable:
        assert any("C++ compiler unavailable" in str(item.message) for item in captured)
        pytest.skip("Inductor CPU C++ compiler unavailable (InvalidCxxCompiler)")
    (out[0].sum() + out[1].sum()).backward()
    reference = copy.deepcopy(model)
    reference.belief_compile = "off"
    reference.zero_grad(set_to_none=True)
    expected = reference.particles_from_noise(obs, prev, obs, torch.tensor([False]), initial_noise=prev,
                                             langevin_noise=noise, step_size=0.03, track_grad=True)
    (expected[0].sum() + expected[1].sum()).backward()
    for x, y in zip(out, expected):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)
    for x, y in zip(model.parameters(), reference.parameters()):
        if x.grad is None:
            assert y.grad is None
        else:
            torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("parameterization", ["energy", "direct", "direct_joint"])
@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
@pytest.mark.parametrize("track_grad,need_scores", [(True, True), (False, False)])
def test_aot_eager_all_score_paths(monkeypatch, parameterization, sampler, track_grad, need_scores):
    torch.manual_seed(77)
    eager = EnergyBelief(2, 2, 2, [4], particle_sampler=sampler,
                         score_parameterization=parameterization, score_implementation="analytic")
    compiled = copy.deepcopy(eager)
    compiled.belief_compile = "inductor"
    original = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: original(fn, backend="aot_eager", fullgraph=True, dynamic=False))
    obs = torch.randn(2, 2)
    prev = torch.randn(2, 3, 2)
    noise = torch.randn(2, 2, 3, 2)
    def run(model):
        x = prev.clone().requires_grad_(track_grad)
        out = model.particles_from_noise(obs, x, obs, torch.tensor([True, False]), initial_noise=prev,
             langevin_noise=noise, step_size=0.03, track_grad=track_grad, need_scores=need_scores)
        if track_grad:
            (out[0].sum() + out[1].sum()).backward()
        return out, x.grad, [p.grad for p in model.parameters()]
    a, b = run(eager), run(compiled)
    torch.testing.assert_close(a[0][0], b[0][0], atol=1e-6, rtol=1e-6)
    if need_scores:
        torch.testing.assert_close(a[0][1], b[0][1], atol=1e-6, rtol=1e-6)
    else:
        assert b[0][1] is None
    if track_grad:
        torch.testing.assert_close(a[1], b[1], atol=1e-5, rtol=1e-5)
        for x, y in zip(a[2], b[2]):
            if x is None:
                assert y is None
            else:
                torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("parameterization", ["energy", "direct", "direct_joint"])
@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
def test_inductor_cuda_matches_eager(parameterization, sampler):
    torch.manual_seed(9)
    eager = EnergyBelief(2, 2, 2, [4], particle_sampler=sampler,
                         score_parameterization=parameterization, score_implementation="analytic").cuda()
    compiled = copy.deepcopy(eager)
    compiled.belief_compile = "inductor"
    obs, prev, noise = torch.randn(2, 2, device="cuda"), torch.randn(2, 3, 2, device="cuda"), torch.randn(2, 2, 3, 2, device="cuda")
    def run(model):
        x = prev.detach().clone().requires_grad_()
        p, s = model.particles_from_noise(obs, x, obs, torch.tensor([True, False], device="cuda"), initial_noise=prev,
                                          langevin_noise=noise, step_size=0.03, track_grad=True)
        (p.square().sum() + s.square().sum()).backward()
        return p, s, x.grad, [p.grad for p in model.parameters()]
    a, b = run(eager), run(compiled)
    for x, y in zip(a[:2], b[:2]):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a[2], b[2], atol=1e-5, rtol=1e-5)
    for x, y in zip(a[3], b[3]):
        if x is None:
            assert y is None
        else:
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


def test_compare_two_updates_inductor_matches_eager(tmp_path):
    import csv
    import warnings

    from sb_pomdp.compare import run_comparisons
    # Probe the real backend; systems without a supported compiler explicitly skip.
    probe = EnergyBelief(2, 2, 2, [4], score_implementation="analytic", belief_compile="inductor")
    obs, prev, noise = torch.randn(1, 2), torch.randn(1, 2, 2), torch.randn(1, 1, 2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        probe.particles_from_noise(obs, prev, obs, torch.tensor([False]), initial_noise=prev,
                                   langevin_noise=noise, step_size=0.03, track_grad=True)
    if probe._compile_unavailable:
        pytest.skip("Inductor CPU C++ compiler unavailable (InvalidCxxCompiler)")
    base = {"experiment.eval_episodes": 1, "experiment.eval_interval_updates": 999,
            "experiment.checkpoint_interval_updates": 999, "environment.tasks.masked_cartpole.horizon": 20,
            "model.num_particles": 2, "model.langevin_steps": 1, "model.energy_hidden": [4],
            "model.d_model": 4, "model.num_heads": 1, "model.num_transformer_layers": 1,
            "model.transformer_ff_dim": 8, "model.policy_hidden": [4], "model.diffusion_steps": 1,
            "model.score_implementation": "analytic", "ppo.total_steps": 8, "ppo.num_envs": 1,
            "ppo.rollout_steps": 4, "ppo.epochs": 1, "ppo.minibatch_size": 4,
            "ppo.sequence_microbatch_size": 4}
    rows = []
    for mode in ("off", "inductor"):
        cfg = load_config("config/local.json").with_overrides({**base, "model.belief_compile": mode,
                                                               "experiment.output_dir": str(tmp_path / mode)})
        run, summaries, _ = run_comparisons(cfg, methods=["score_transformer"], tasks=["masked_cartpole"],
            seeds=[0], gradient_modes=["full"], max_updates=None, segment_updates=None,
            run_dir=tmp_path / mode / "run", resume=False, label="compile-test", campaign_id="test",
            bulk_index=1, verbose=False)
        with (run.root / summaries[0]["result_dir"] / "metrics.csv").open(newline="", encoding="utf-8") as f:
            rows.append(list(csv.DictReader(f)))
    assert len(rows[0]) == len(rows[1]) == 2
    for left, right in zip(*rows):
        for key in left:
            if any(word in key.lower() for word in ("time", "second", "fps", "elapsed")):
                continue
            try:
                a, b = float(left[key]), float(right[key])
            except (ValueError, TypeError):
                assert left[key] == right[key]
            else:
                assert a == pytest.approx(b, abs=1e-4, rel=1e-4, nan_ok=True)


def test_missing_cpu_compiler_warns_once_and_falls_back(monkeypatch):
    class InvalidCxxCompiler(Exception):
        pass
    model = EnergyBelief(2, 2, 2, [4], score_implementation="analytic", belief_compile="inductor")
    eager = copy.deepcopy(model)
    eager.belief_compile = "off"
    def fake_compile(fn, **kwargs):
        def fail(*args):
            raise InvalidCxxCompiler("compiler unavailable")
        return fail
    monkeypatch.setattr(torch, "compile", fake_compile)
    obs, prev, noise = torch.randn(1, 2), torch.randn(1, 2, 2), torch.randn(1, 1, 2, 2)
    def run(belief):
        return belief.particles_from_noise(obs, prev, obs, torch.tensor([False]), initial_noise=prev,
                                            langevin_noise=noise, step_size=0.03, track_grad=True)
    expected = run(eager)
    with pytest.warns(RuntimeWarning, match=r"C\+\+ compiler unavailable"):
        actual = run(model)
    torch.testing.assert_close(expected[0], actual[0])
    torch.testing.assert_close(expected[1], actual[1])
    assert model._compile_unavailable
    run(model)


def test_compare_two_updates_aot_eager(tmp_path, monkeypatch):
    """Exercise the complete compare harness even when the CPU has no C++ compiler."""
    original = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: original(
        fn, backend="aot_eager", fullgraph=True, dynamic=False,
    ))
    test_compare_two_updates_inductor_matches_eager(tmp_path)
