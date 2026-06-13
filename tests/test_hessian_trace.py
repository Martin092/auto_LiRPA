"""Checks for DirectHessianTraceOP, the forward-mode Hessian trace.

Three properties are tested on small CPU models. The expanded graph's forward
value must equal the autograd Hessian trace exactly, because the forward
recursion is just the chain rule evaluated at a point. The IBP and CROWN
bounds must enclose the true trace everywhere on a dense sample of the input
box, including for networks deeper than the two-hidden-layer nets the
hardcoded experiment wrapper supports. And on those two-hidden-layer nets the
general graph must produce bounds close to the hand-rolled analytical trace
recursion, since both encode the same math.
"""

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.hessian_trace import DirectHessianTraceOP
from auto_LiRPA.operators.s_shaped import SigmoidGradOp, SigmoidSecondGradOp
from auto_LiRPA.perturbations import PerturbationLpNorm


class _TraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return DirectHessianTraceOP.apply(self.model(x), x)


def _mlp(activation, dims, seed=0):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


MODEL_CASES = {
    'sigmoid': lambda: _mlp(nn.Sigmoid, [2, 4, 4, 1]),
    'softplus': lambda: _mlp(nn.Softplus, [2, 4, 4, 1]),
    'tanh': lambda: _mlp(nn.Tanh, [2, 4, 4, 1]),
    'sigmoid_deep': lambda: _mlp(nn.Sigmoid, [3, 5, 4, 5, 4, 1]),
    'sigmoid_multi_output': lambda: _mlp(nn.Sigmoid, [2, 4, 4, 3]),
}


def _autograd_traces(model, x):
    """True tr(d^2 out_k / d input^2), shape [batch, out_dim], via autograd."""
    traces = []
    for b in range(x.shape[0]):
        xb = x[b].detach()
        out_dim = model(xb.unsqueeze(0)).numel()
        row = []
        for k in range(out_dim):
            def scalar_forward(inp):
                return model(inp.unsqueeze(0)).reshape(-1)[k]

            hessian = torch.autograd.functional.hessian(scalar_forward, xb)
            row.append(torch.diagonal(hessian).sum())
        traces.append(torch.stack(row))
    return torch.stack(traces)


def _bounded_trace_module(model, x0):
    return BoundedModule(_TraceWrapper(model), x0, device='cpu')


def _box(x0, eps):
    return BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), x_L=x0 - eps, x_U=x0 + eps))


@pytest.mark.parametrize('name', list(MODEL_CASES), ids=list(MODEL_CASES))
def test_forward_value_matches_autograd(name):
    model = MODEL_CASES[name]()
    dim = model[0].in_features
    torch.manual_seed(1)
    x0 = torch.randn(3, dim)
    bounded = _bounded_trace_module(model, x0)
    forward_trace = bounded(x0)
    expected = _autograd_traces(model, x0)
    assert forward_trace.shape == expected.shape
    torch.testing.assert_close(forward_trace, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('method', ['IBP', 'backward'])
@pytest.mark.parametrize('name', list(MODEL_CASES), ids=list(MODEL_CASES))
def test_bounds_enclose_true_trace(name, method):
    model = MODEL_CASES[name]()
    dim = model[0].in_features
    torch.manual_seed(2)
    x0 = torch.randn(2, dim).clamp(-0.5, 0.5)
    eps = 0.1
    bounded = _bounded_trace_module(model, x0)
    lower, upper = bounded.compute_hessian_trace_bounds(_box(x0, eps), method=method)
    assert (lower <= upper + 1e-6).all()

    torch.manual_seed(3)
    offsets = torch.rand(200, dim) * 2 - 1
    for b in range(x0.shape[0]):
        samples = x0[b] + eps * offsets
        true_traces = _autograd_traces(model, samples)
        slack = 1e-5 + 1e-5 * true_traces.abs()
        assert (true_traces >= lower[b] - slack).all(), (
            f'{name}/{method}: lower bound crossed by '
            f'{(lower[b] - true_traces).max().item():.2e}')
        assert (true_traces <= upper[b] + slack).all(), (
            f'{name}/{method}: upper bound crossed by '
            f'{(true_traces - upper[b]).max().item():.2e}')


class _HardcodedTrace(nn.Module):
    """The two-hidden-layer trace recursion from the experiments, unrolled by
    hand the same way as experiments/common/trace.py, as a reference graph."""

    def __init__(self, model, d1, d2):
        super().__init__()
        self.linear0, self.linear1, self.linear2 = model[0], model[2], model[4]
        self.d1, self.d2 = d1, d2

    def forward(self, x):
        p1 = self.linear0(x)
        a1 = torch.sigmoid(p1)
        p2 = self.linear1(a1)

        w0 = self.linear0.weight.unsqueeze(0)
        w1 = self.linear1.weight.unsqueeze(0)

        d1_p1, d1_p2 = self.d1(p1), self.d1(p2)
        d2_p1, d2_p2 = self.d2(p1), self.d2(p2)

        j1_rowsq = (self.linear0.weight ** 2).sum(dim=1)
        j2 = (w1 * d1_p1.unsqueeze(1)) @ w0
        j2_rowsq = (j2 ** 2).sum(dim=2)

        t_a1 = d2_p1 * j1_rowsq
        t2 = nn.functional.linear(t_a1, self.linear1.weight)
        t_a2 = d1_p2 * t2 + d2_p2 * j2_rowsq
        return nn.functional.linear(t_a2, self.linear2.weight)


@pytest.mark.parametrize('method', ['IBP', 'backward'])
def test_close_to_hardcoded_trace_recursion(method):
    model = _mlp(nn.Sigmoid, [2, 4, 4, 1])
    reference = _HardcodedTrace(
        model, SigmoidGradOp.apply, SigmoidSecondGradOp.apply)
    torch.manual_seed(4)
    x0 = torch.randn(2, 2).clamp(-0.5, 0.5)
    eps = 0.1

    bounded = _bounded_trace_module(model, x0)
    lower, upper = bounded.compute_hessian_trace_bounds(_box(x0, eps), method=method)

    bounded_ref = BoundedModule(reference, x0, device='cpu')
    ref_lower, ref_upper = bounded_ref.compute_bounds(
        x=(_box(x0, eps),), method=method)

    width = (ref_upper - ref_lower).clamp(min=1e-6)
    assert ((lower - ref_lower).abs() <= 0.05 * width + 1e-5).all(), (
        f'lower bounds differ from the hardcoded recursion by '
        f'{(lower - ref_lower).abs().max().item():.2e}')
    assert ((upper - ref_upper).abs() <= 0.05 * width + 1e-5).all(), (
        f'upper bounds differ from the hardcoded recursion by '
        f'{(upper - ref_upper).abs().max().item():.2e}')
