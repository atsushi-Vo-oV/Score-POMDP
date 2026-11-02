"""CUDA capture with private autograd leaves and deferred-backward recomputation.

Each outer autograd node owns its inputs and outputs. The shared capture
workspace is restored immediately before backward, so outstanding sequence
steps cannot overwrite each other's saved activations. First-order grads only.
"""

from __future__ import annotations

import threading

import torch
from torch.autograd.function import once_differentiable


class _RecomputedStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runner, *inputs):
        ctx.runner = runner
        # Includes original parameters, preserving optimizer version checks.
        ctx.save_for_backward(*inputs)
        return runner.forward(inputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        return (None, *ctx.runner.backward(ctx.saved_tensors, grad_outputs))


class _TransportModule(torch.nn.Module):
    def __init__(self, model, function):
        super().__init__()
        self.model = model
        # A callable may itself be a Module; registering it again would create
        # duplicate tied paths and interfere with functional_call restoration.
        object.__setattr__(self, "function", function)

    def forward(self, *inputs):
        return self.function(*inputs)


class _CaptureModule(torch.nn.Module):
    """Share parameter storage, but never eager AccumulateGrad nodes/streams.

    The callable must be eager. functional_call temporarily substitutes independent
    leaves without copying weights or permanently changing the original model.
    Compiled bound methods can retain the original leaves in Dynamo/AOT and are
    unsupported. Only aliases enter make_graphed_callables' parameter surface.
    """

    def __init__(self, model, function, *, track_grad):
        super().__init__()
        if hasattr(function, "_torchdynamo_orig_callable"):
            raise ValueError("CUDA graph capture requires an eager callable; compiled callables are unsupported")
        named = tuple(model.named_parameters())
        self.names = tuple("model." + name for name, _ in named)
        self.aliases = torch.nn.ParameterList(
            torch.nn.Parameter(value.detach(), requires_grad=track_grad and value.requires_grad)
            for _, value in named
        )
        # Do not register the original parameters a second time.
        object.__setattr__(self, "source", _TransportModule(model, function))

    def forward(self, *inputs):
        return torch.func.functional_call(
            self.source, dict(zip(self.names, self.aliases)), inputs,
        )


class CudaGraphStep:
    """One private capture workspace per caller stream and shape/grad signature."""

    def __init__(self, function, inputs, model, *, track_grad):
        self.parameters = tuple(model.parameters())
        self.num_inputs = len(inputs)
        device = inputs[0].device
        self.stream = torch.cuda.current_stream(device)
        self.lock = threading.Lock()
        self.static_inputs = tuple(
            value.detach().clone().requires_grad_(track_grad and value.requires_grad)
            for value in inputs
        )
        self.module = _CaptureModule(model, function, track_grad=track_grad)
        self.surface = self.static_inputs + tuple(self.module.parameters())
        self.differentiable = tuple(value for value in self.surface if value.requires_grad)
        self.forward_graph = None
        # The outer stream context also restores the caller if torch 2.3.1's
        # graph.__exit__ throws at capture_end before exiting its own stream ctx.
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            try:
                if track_grad and self.differentiable:
                    with torch.enable_grad():
                        self.graphed = torch.cuda.make_graphed_callables(
                            self.module, self.static_inputs,
                            num_warmup_iters=3, allow_unused_input=True,
                        )
                else:
                    self._capture_forward_only(device)
            except Exception as exc:
                # PyTorch 2.3.1 capture_end can throw before its C++ generator
                # capture_epilogue. Neither reset() nor set_rng_state() clears
                # that internal flag through the supported Python API. Do not
                # silently fall back to eager or cache an incomplete runner.
                raise RuntimeError(
                    "Belief CUDA graph capture failed. PyTorch 2.3.1 may leave "
                    "the CUDA generator/allocator in capture state; restart this "
                    "process before further CUDA work (or disable belief_cuda_graphs "
                    "in a fresh process). The original capture error is chained."
                ) from exc

    def _capture_forward_only(self, device):
        # make_graphed_callables in 2.3.1 calls autograd.grad unconditionally,
        # so inference needs a forward-only capture.
        torch.cuda.synchronize(device)
        with torch.cuda.stream(torch.cuda.Stream(device=device)), torch.no_grad():
            for _ in range(3):
                self.module(*self.static_inputs)
        torch.cuda.synchronize(device)
        self.forward_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.forward_graph), torch.no_grad():
            self.outputs = self.module(*self.static_inputs)

    def _restore_inputs(self, inputs):
        if torch.cuda.current_stream(self.static_inputs[0].device) != self.stream:
            raise RuntimeError("belief CUDA graph must replay on its capture caller's stream")
        with torch.no_grad():
            for static, value in zip(self.static_inputs, inputs[:self.num_inputs]):
                static.copy_(value)

    def forward(self, inputs):
        with self.lock, torch.no_grad():
            self._restore_inputs(inputs)
            if self.forward_graph is not None:
                self.forward_graph.replay()
                outputs = self.outputs
            else:
                outputs = self.graphed(*self.static_inputs)
            # Static outputs must not escape: future forwards reuse this storage.
            return tuple(value.clone() for value in outputs)

    def backward(self, inputs, grad_outputs):
        with self.lock:
            self._restore_inputs(inputs)
            if self.forward_graph is not None:
                return (None,) * len(self.surface)
            # Make a fresh library autograd node and immediately consume it.
            # No library node survives across another forward replay.
            with torch.enable_grad():
                outputs = self.graphed(*self.static_inputs)
                gradients = torch.autograd.grad(
                    outputs, self.differentiable,
                    tuple(torch.zeros_like(output) if grad is None else grad
                          for output, grad in zip(outputs, grad_outputs)),
                    allow_unused=True,
                )
            iterator = iter(gradients)
            with torch.no_grad():
                result = tuple(next(iterator) if value.requires_grad else None
                               for value in self.surface)
                # A later backward may reuse the library's gradient storage.
                return tuple(None if value is None else value.clone() for value in result)

    def __call__(self, *inputs):
        return _RecomputedStep.apply(self, *inputs, *self.parameters)
