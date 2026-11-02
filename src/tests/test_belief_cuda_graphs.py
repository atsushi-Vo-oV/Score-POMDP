"""Run this file on the CUDA server before enabling belief_cuda_graphs."""

import copy
import warnings
from contextlib import contextmanager

import pytest
import torch

from sb_pomdp.belief import EnergyBelief
from sb_pomdp.config import ConfigError
from sb_pomdp.cuda_graphs import CudaGraphStep, _CaptureModule, _RecomputedStep


@pytest.fixture
def fake_cuda(monkeypatch):
    """Exercise runner decisions on CPU; this does not emulate CUDA kernels."""
    state = {"stream": "caller", "device": None, "builds": []}

    @contextmanager
    def context(key, value):
        previous = state[key]
        state[key] = value
        try:
            yield
        finally:
            state[key] = previous

    def make_graphed(module, inputs, **kwargs):
        state["builds"].append((module, inputs, kwargs, state["device"]))
        assert torch.is_grad_enabled()
        return module

    monkeypatch.setattr(torch.cuda, "device", lambda device: context("device", device))
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: context("stream", stream))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: state["stream"])
    monkeypatch.setattr(torch.cuda, "make_graphed_callables", make_graphed)
    return state


def test_capture_parameter_leaves_are_private_but_storage_is_live():
    model = torch.nn.Linear(2, 2)
    original_parameters = tuple(model.parameters())
    x = torch.randn(3, 2, requires_grad=True)
    # Keep eager AccumulateGrad nodes alive, as in an outstanding sequence.
    eager = model(x)
    captured = _CaptureModule(model, model.forward, track_grad=True)
    actual = captured(x)
    torch.testing.assert_close(actual, eager)
    aliases = tuple(captured.parameters())
    for original, alias in zip(original_parameters, aliases):
        assert original is not alias
        assert original.data_ptr() == alias.data_ptr()
        assert torch.autograd.graph.get_gradient_edge(original).node is not (
            torch.autograd.graph.get_gradient_edge(alias).node
        )
    grads = torch.autograd.grad(actual.sum(), aliases + original_parameters, allow_unused=True)
    assert all(value is not None for value in grads[:len(aliases)])
    assert all(value is None for value in grads[len(aliases):])
    assert all(a is b for a, b in zip(model.parameters(), original_parameters))
    with torch.no_grad():
        model.weight.add_(0.5)
    torch.testing.assert_close(captured(x), model(x))


@pytest.mark.parametrize("bound_method", [False, True])
def test_capture_rejects_compiled_callable(bound_method):
    model = torch.nn.Linear(2, 2)
    function = torch.compile(model.forward if bound_method else model,
                             backend="aot_eager", fullgraph=True)
    function(torch.randn(3, 2))
    with pytest.raises(ValueError, match="eager callable"):
        _CaptureModule(model, function, track_grad=True)


@pytest.mark.parametrize("mutate_flags", [False, True])
def test_compile_and_cuda_graphs_are_rejected_before_execution(mutate_flags, monkeypatch):
    def unexpected_compile(*args, **kwargs):
        pytest.fail("unsupported combination reached compilation")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    with pytest.raises(ConfigError, match="alternatives.*parameter gradients"):
        model = EnergyBelief(2, 2, 1, [8], score_implementation="analytic",
                             belief_compile="inductor", belief_cuda_graphs=not mutate_flags)
        model.belief_cuda_graphs = True
        model.particles_from_noise(
            torch.randn(1, 2), torch.randn(1, 3, 2), torch.randn(1, 1), torch.tensor([False]),
            initial_noise=torch.randn(1, 3, 2), langevin_noise=torch.randn(1, 2, 3, 2),
            step_size=0.02, track_grad=True,
        )


@pytest.mark.parametrize("length", [2, 16, 128])
def test_library_runner_deferred_backward_on_cpu(fake_cuda, length):
    model = torch.nn.Linear(1, 1, bias=False)
    model.register_parameter("unused", torch.nn.Parameter(torch.tensor(0.5)))
    model.weight.data.fill_(0.9)
    initial = torch.tensor([[0.3]], requires_grad=True)
    constant = torch.tensor(0.001)

    def function(x, shift):
        return (model(x + shift).sin(),)

    runner = CudaGraphStep(function, (initial, constant), model, track_grad=True)
    _, samples, options, device = fake_cuda["builds"][0]
    assert options == {"num_warmup_iters": 3, "allow_unused_input": True}
    assert device == initial.device
    assert [value.requires_grad for value in samples] == [True, False]
    assert samples[0].data_ptr() != initial.data_ptr()
    for _ in range(2):
        state = initial
        outputs = []
        for _ in range(length):
            state, = runner(state, constant)
            outputs.append(state)
        actual = torch.autograd.grad(sum(outputs).sum(), (initial, *model.parameters()),
                                     allow_unused=True)
        state = initial
        eager = []
        for _ in range(length):
            state, = function(state, constant)
            eager.append(state)
        expected = torch.autograd.grad(sum(eager).sum(), (initial, *model.parameters()),
                                       allow_unused=True)
        for a, b in zip(actual, expected):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b)
        with torch.no_grad():
            model.weight.add_(0.01)


def test_runner_rejects_wrong_stream_and_changed_parameters(fake_cuda):
    model = torch.nn.Linear(1, 1)
    x = torch.ones(1, 1, requires_grad=True)
    runner = CudaGraphStep(lambda x: (model(x),), (x,), model, track_grad=True)
    output, = runner(x)
    fake_cuda["stream"] = "other"
    with pytest.raises(RuntimeError, match="caller's stream"):
        runner(x)
    fake_cuda["stream"] = "caller"
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


def test_capture_failure_restores_stream_and_chains_error(fake_cuda, monkeypatch):
    failure = RuntimeError("operation failed due to a previous error during capture")

    def fail(*args, **kwargs):
        # Mimic graph.__exit__ failing before it restores the stream.
        fake_cuda["stream"] = "leaked capture stream"
        raise failure

    monkeypatch.setattr(torch.cuda, "make_graphed_callables", fail)
    model = torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="restart this process") as caught:
        CudaGraphStep(lambda x: (model(x),), (torch.ones(1, 1),), model, track_grad=True)
    assert caught.value.__cause__ is failure
    assert fake_cuda["stream"] == "caller"
    assert fake_cuda["device"] is None


def test_no_grad_uses_forward_only_capture(fake_cuda, monkeypatch):
    events = []

    @contextmanager
    def graph_context(graph):
        # The production call must let torch.cuda.graph select its default
        # capture stream, just like make_graphed_callables does.
        with torch.cuda.stream("capture"):
            yield

    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: "warmup")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    model = torch.nn.Linear(1, 1)

    def function(x):
        assert not torch.is_grad_enabled()
        events.append(fake_cuda["stream"])
        return (model(x),)

    runner = CudaGraphStep(function, (torch.ones(1, 1, requires_grad=True),),
                           model, track_grad=False)
    assert not runner.differentiable
    assert events == ["sync", "warmup", "warmup", "warmup", "sync", "capture"]
    assert fake_cuda["stream"] == "caller"
    assert not fake_cuda["builds"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable: real stream validation")
@pytest.mark.parametrize("side_stream", [False, True])
@pytest.mark.parametrize("track_grad", [False, True])
def test_cuda_capture_with_live_eager_parameter_nodes(side_stream, track_grad):
    caller = torch.cuda.current_stream()
    stream = torch.cuda.Stream() if side_stream else caller
    stream.wait_stream(caller)
    with torch.cuda.stream(stream):
        model = torch.nn.Linear(2, 2).cuda()
        reference = copy.deepcopy(model)
        x = torch.randn(3, 2, device="cuda", requires_grad=True)
        expected_x = x.detach().clone().requires_grad_()
        # These nodes must remain alive through capture, including an existing
        # AccumulateGrad on the legacy default stream in the default-stream case.
        eager = model(x).square().sum()
        expected_eager = reference(expected_x).square().sum()
        rng = torch.cuda.get_rng_state().clone()
        with torch.set_grad_enabled(track_grad):
            runner = CudaGraphStep(lambda value: (model(value).sin(),), (x,), model,
                                   track_grad=track_grad)
            first, = runner(x)
            saved = first.clone()
            second, = runner(x + 0.1)
            torch.testing.assert_close(first, saved, atol=0, rtol=0)
            expected_first = reference(expected_x).sin()
            expected_second = reference(expected_x + 0.1).sin()
            torch.testing.assert_close(first, expected_first)
            torch.testing.assert_close(second, expected_second)
        if track_grad:
            (eager + first.sum() + second.sum()).backward()
            (expected_eager + expected_first.sum() + expected_second.sum()).backward()
            for a, b in zip((x, *model.parameters()), (expected_x, *reference.parameters())):
                torch.testing.assert_close(a.grad, b.grad)
        else:
            assert not first.requires_grad
        assert torch.equal(rng, torch.cuda.get_rng_state())
    caller.wait_stream(stream)


def test_cuda_graph_flag_is_warned_cpu_noop():
    model = EnergyBelief(2, 2, 1, [8], belief_cuda_graphs=True)
    reference = copy.deepcopy(model)
    reference.belief_cuda_graphs = False
    args = (torch.randn(1, 2), torch.randn(1, 3, 2), torch.randn(1, 1), torch.tensor([False]))
    kwargs = {"initial_noise": torch.randn(1, 3, 2), "langevin_noise": torch.randn(1, 2, 3, 2),
              "step_size": 0.02, "track_grad": True}
    expected = reference.particles_from_noise(*args, **kwargs)
    with pytest.warns(RuntimeWarning, match="no-op on CPU"):
        actual = model.particles_from_noise(*args, **kwargs)
    with warnings.catch_warnings(record=True) as caught:
        model.particles_from_noise(*args, **kwargs)
    assert not caught
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert not model._cuda_graph_chains


def test_recomputed_autograd_node_preserves_temporal_inputs_on_cpu():
    # CPU-testable lifetime contract. The CUDA runner supplies the same interface
    # using captures; this runner deliberately overwrites its workspace each call.
    class Runner:
        def forward(self, inputs):
            x, weight = inputs
            self.workspace = (x * weight).sin()
            return (self.workspace.clone(),)

        def backward(self, inputs, grads):
            x, weight = inputs
            self.forward(inputs)
            return (grads[0] * (x * weight).cos() * weight,
                    grads[0] * (x * weight).cos() * x)

    for length in (2, 16, 128):
        initial = torch.tensor(0.3, requires_grad=True)
        weight = torch.tensor(0.9, requires_grad=True)
        runner = Runner()
        state = initial
        captured = []
        for index in range(length):
            state, = _RecomputedStep.apply(runner, state + index * 0.001, weight)
            captured.append(state)
        actual = torch.autograd.grad(sum(captured), (initial, weight))
        state = initial
        eager = []
        for index in range(length):
            state = ((state + index * 0.001) * weight).sin()
            eager.append(state)
        expected = torch.autograd.grad(sum(eager), (initial, weight))
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable: capture validation requires the CUDA server")
@pytest.mark.parametrize("length", [2, 16])
@pytest.mark.parametrize("sampler", ["ula", "svgd", "smi"])
@pytest.mark.parametrize("parameterization", ["energy", "direct", "direct_joint"])
def test_cuda_graph_deferred_sequence_backward(length, sampler, parameterization):
    _check_cuda_sequence(length, sampler, parameterization, compile_mode="off", need_scores=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable: capture validation requires the CUDA server")
@pytest.mark.parametrize("need_scores", [False, True])
def test_cuda_graph_sequence_and_optional_scores(need_scores):
    _check_cuda_sequence(16, "ula", "energy", compile_mode="off", need_scores=need_scores)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable: full replay validation requires the CUDA server")
def test_cuda_graph_128_steps():
    # Capture can select different cuBLAS kernels. Across 128 float32 steps,
    # roundoff accumulates: the server observed 8.65e-5 absolute / 2.82e-5
    # relative gradient error. Like the Adam check in test_speedup_exactness,
    # use a measured absolute allowance for amplified roundoff. Keep rtol and
    # output checks strict, and retain 1e-5 atol for every 2/16-step test.
    _check_cuda_sequence(128, "ula", "energy", compile_mode="off", need_scores=True,
                         production=True, gradient_atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable: checkpoint/capture validation requires the CUDA server")
def test_cuda_graph_with_activation_checkpoint():
    _check_cuda_sequence(2, "ula", "energy", compile_mode="off", need_scores=True, checkpoint=True)


def _check_cuda_sequence(length, sampler, parameterization, *, compile_mode, need_scores,
                         production=False, checkpoint=False, gradient_atol=1e-5):
    torch.manual_seed(51)
    torch.backends.cuda.matmul.allow_tf32 = False
    state_dim, particles, steps = (6, 16, 12) if production else (2, 3, 3)
    reference = EnergyBelief(
        2, state_dim, 1, [8], score_implementation="analytic", particle_sampler=sampler,
        score_parameterization=parameterization, langevin_warm_start=True,
        observation_anchor=True, transition_proposal=True, proposal_hidden=[4],
        langevin_step_size_learnable=True, langevin_step_size=0.02, langevin_steps=steps,
    ).cuda()
    captured = copy.deepcopy(reference)
    captured.belief_cuda_graphs = True
    captured.belief_compile = compile_mode
    noise0 = torch.randn(length, 2, particles, state_dim, device="cuda")
    noise = torch.randn(length, 2, steps, particles, state_dim, device="cuda")
    observations = torch.randn(length, 2, 2, device="cuda")
    actions = torch.randn(length, 2, 1, device="cuda")
    previous = torch.randn(2, particles, state_dim, device="cuda")
    rng = torch.cuda.get_rng_state().clone()

    def run(model):
        model.zero_grad(set_to_none=True)
        inputs = [value.clone().requires_grad_() for value in (observations, actions, previous)]
        obs, act, state = inputs
        outputs = []
        for index in range(length):
            initial = torch.tensor([index == 0, index % 5 == 0], device="cuda")
            branch = model.classify_potential_branch(initial)

            def step(observation, previous, action, flags, start_noise, step_noise,
                     branch=branch):
                return model.particles_from_noise(
                    observation, previous, action, flags, initial_noise=start_noise,
                    langevin_noise=step_noise, step_size=0.02, track_grad=True,
                    need_scores=need_scores, potential_branch=branch,
                )

            step_inputs = (obs[index], state, act[index], initial, noise0[index], noise[index])
            if checkpoint:
                from torch.utils.checkpoint import checkpoint as activation_checkpoint
                state, scores = activation_checkpoint(step, *step_inputs, use_reentrant=False)
            else:
                state, scores = step(*step_inputs)
            outputs.append(state)
            if scores is not None:
                outputs.append(scores)
        # All forwards precede a single backward: detects intermediate overwrite.
        sum(value.square().mean() for value in outputs).backward()
        return outputs, [value.grad for value in inputs] + [value.grad for value in model.parameters()]

    for _ in range(2):
        expected, expected_grads = run(reference)
        actual, actual_grads = run(captured)
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)
        for x, y in zip(expected_grads, actual_grads):
            assert (x is None) == (y is None)
            if x is not None:
                torch.testing.assert_close(x, y, atol=gradient_atol, rtol=1e-5)
        assert torch.equal(rng, torch.cuda.get_rng_state())
        # Replay must read updated parameter storage, not values frozen at capture.
        with torch.no_grad():
            for model in (reference, captured):
                for parameter in model.parameters():
                    parameter.add_(0.0001)
    assert captured._cuda_graph_chains
