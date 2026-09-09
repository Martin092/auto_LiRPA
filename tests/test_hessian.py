"""Tests for the Hessian trace and diagonal graphs.

DirectHessianTraceOP and DirectHessianDiagOP use the same recursion, the
diagonal being the trace without the sum over input dimensions. They are
tested together so both run on the same models against the same autograd
ground truth.

What is checked:

* the forward value of the expanded graph equals autograd exactly, since the
  recursion is just the chain rule at a point
* IBP and CROWN bounds contain the true values on many samples of the input
  box, over several activations, depths, a multi-output model, and a composite
  model that uses the Add, Sub and Mul builders
* summing the diagonal gives the same value as the trace graph
* a box of radius zero makes the IBP bounds exact
* on two-hidden-layer nets the trace graph is close to a recursion written out
  by hand, since both compute the same thing

Bounds are only compared against autograd. They are not compared against the
full-Hessian graphs, because two different relaxations do not have to be
ordered.
"""

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.hessian_diag import DirectHessianDiagOP
from auto_LiRPA.hessian_trace import DirectHessianTraceOP
from auto_LiRPA.operators.s_shaped import SigmoidGradOp, SigmoidSecondGradOp
from auto_LiRPA.perturbations import PerturbationLpNorm


class _Wrapper(nn.Module):
    """Marks the model output for one of the forward Hessian state graphs."""

    def __init__(self, model, op):
        super().__init__()
        self.model = model
        self.op = op

    def forward(self, x):
        return self.op.apply(self.model(x), x)


class _Composite(nn.Module):
    """g(x) * h(x) + g(x) - h(x): exercises Mul (both operands perturbed),
    Add, Sub, and the fan-out of g and h into two consumers each."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.g = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 1))
        self.h = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))

    def forward(self, x):
        g, h = self.g(x), self.h(x)
        return g * h + g - h


def _mlp(activation, dims, seed=0):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


# name -> (model factory, input dimension)
MODEL_CASES = {
    'sigmoid': (lambda: _mlp(nn.Sigmoid, [2, 4, 4, 1]), 2),
    'softplus': (lambda: _mlp(nn.Softplus, [2, 4, 4, 1]), 2),
    'tanh': (lambda: _mlp(nn.Tanh, [2, 4, 4, 1]), 2),
    'sigmoid_deep': (lambda: _mlp(nn.Sigmoid, [3, 5, 4, 5, 4, 1]), 3),
    'sigmoid_multi_output': (lambda: _mlp(nn.Sigmoid, [2, 4, 4, 3]), 2),
    'composite': (_Composite, 2),
}


def _case(name):
    make, in_dim = MODEL_CASES[name]
    return make(), in_dim


def _autograd_diags(model, x):
    """True diag(d^2 out_k / d input^2), shape [batch, out_dim, in_dim]."""
    rows = []
    for b in range(x.shape[0]):
        xb = x[b].detach()
        out_dim = model(xb.unsqueeze(0)).numel()
        per_out = []
        for k in range(out_dim):
            def scalar_forward(inp):
                return model(inp.unsqueeze(0)).reshape(-1)[k]
            hessian = torch.autograd.functional.hessian(scalar_forward, xb)
            per_out.append(torch.diagonal(hessian))
        rows.append(torch.stack(per_out))
    return torch.stack(rows)


def _autograd_traces(model, x):
    """True tr(d^2 out_k / d input^2), shape [batch, out_dim]."""
    return _autograd_diags(model, x).sum(dim=-1)


def _bounded_trace(model, x0):
    return BoundedModule(
        _Wrapper(model, DirectHessianTraceOP), x0, device='cpu')


def _bounded_diag(model, x0):
    return BoundedModule(
        _Wrapper(model, DirectHessianDiagOP), x0, device='cpu')


def _box(x0, eps):
    return BoundedTensor(x0, PerturbationLpNorm(norm=float('inf'), eps=eps))


@pytest.mark.parametrize('name', list(MODEL_CASES), ids=list(MODEL_CASES))
def test_trace_forward_value_matches_autograd(name):
    model, in_dim = _case(name)
    torch.manual_seed(1)
    x0 = torch.randn(3, in_dim)
    forward_trace = _bounded_trace(model, x0)(x0)
    expected = _autograd_traces(model, x0)
    assert forward_trace.shape == expected.shape
    torch.testing.assert_close(forward_trace, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('name', list(MODEL_CASES), ids=list(MODEL_CASES))
def test_diag_forward_value_matches_autograd(name):
    model, in_dim = _case(name)
    torch.manual_seed(1)
    x0 = torch.randn(3, in_dim)
    value = _bounded_diag(model, x0)(x0)
    expected = _autograd_diags(model, x0)
    assert value.shape == expected.shape
    torch.testing.assert_close(value, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('method', ['IBP', 'backward'])
@pytest.mark.parametrize('name', list(MODEL_CASES), ids=list(MODEL_CASES))
def test_bounds_enclose_true_trace(name, method):
    model, in_dim = _case(name)
    torch.manual_seed(2)
    x0 = torch.randn(2, in_dim).clamp(-0.5, 0.5)
    eps = 0.1
    bounded = _bounded_trace(model, x0)
    lower, upper = bounded.compute_hessian_trace_bounds(
        _box(x0, eps), method=method)
    assert (lower <= upper + 1e-6).all()

    torch.manual_seed(3)
    offsets = torch.rand(200, in_dim) * 2 - 1
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


@pytest.mark.parametrize('method', ['IBP', 'backward'])
@pytest.mark.parametrize('name', list(MODEL_CASES), ids=list(MODEL_CASES))
def test_bounds_enclose_true_diag(name, method):
    model, in_dim = _case(name)
    torch.manual_seed(2)
    x0 = torch.randn(2, in_dim)
    eps = 0.15
    bounded = _bounded_diag(model, x0)
    lower, upper = bounded.compute_hessian_diag_bounds(
        _box(x0, eps), method=method)

    torch.manual_seed(3)
    for _ in range(100):
        samples = x0 + (torch.rand_like(x0) * 2 - 1) * eps
        diag = _autograd_diags(model, samples)
        assert (diag >= lower - 1e-5).all(), (
            f'lower bound violated ({name}, {method})')
        assert (diag <= upper + 1e-5).all(), (
            f'upper bound violated ({name}, {method})')


@pytest.mark.parametrize(
    'name', ['sigmoid', 'sigmoid_deep', 'composite'])
def test_diag_sum_equals_trace_forward(name):
    # the diagonal is the trace with the input-dim reduction deferred, so the
    # two expanded graphs must agree exactly at any evaluation point
    model, in_dim = _case(name)
    torch.manual_seed(4)
    x0 = torch.randn(3, in_dim)
    diag_value = _bounded_diag(model, x0)(x0)
    trace_value = _bounded_trace(model, x0)(x0)
    torch.testing.assert_close(
        diag_value.sum(dim=-1), trace_value, atol=1e-6, rtol=1e-5)


def test_zero_radius_box_is_tight():
    # at eps = 0 IBP degenerates to the exact forward pass, so the bounds must
    # pinch onto the autograd diagonal
    model, in_dim = _case('sigmoid')
    torch.manual_seed(5)
    x0 = torch.randn(2, in_dim)
    bounded = _bounded_diag(model, x0)
    lower, upper = bounded.compute_hessian_diag_bounds(
        _box(x0, 0.0), method='IBP')
    expected = _autograd_diags(model, x0)
    torch.testing.assert_close(lower, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(upper, expected, atol=1e-5, rtol=1e-4)


def test_bounds_shape_and_ordering():
    model, in_dim = _case('sigmoid_multi_output')
    torch.manual_seed(6)
    x0 = torch.randn(2, in_dim)
    bounded = _bounded_diag(model, x0)
    lower, upper = bounded.compute_hessian_diag_bounds(
        _box(x0, 0.05), method='IBP')
    assert lower.shape == upper.shape == (2, 3, 2)
    assert (lower <= upper + 1e-8).all()


class _ConstMatMul(nn.Module):
    """W @ Y with a constant W and a perturbed Y: the pattern of the W @ J and
    W @ D steps of the trace/diag graphs, where the constant is on the LEFT,
    which BoundLinear alone does not recognize as a constant operand."""

    def __init__(self, out_f, in_f, cols, seed=0):
        torch.manual_seed(seed)
        super().__init__()
        self.register_buffer('W', torch.randn(out_f, in_f))
        self.cols = cols

    def forward(self, x):
        return self.W.matmul(x.view(x.shape[0], -1, self.cols))


def test_const_left_matmul_ibp_is_exact():
    # the exact interval image of A @ [y_l, y_u] is A@mid -+ |A|@diff; the
    # generic bilinear fallback that used to run here is looser and blows up
    # in memory for wide layers
    model = _ConstMatMul(out_f=7, in_f=5, cols=3)
    torch.manual_seed(7)
    x0 = torch.randn(2, 15)
    eps = 0.2
    bounded = BoundedModule(model, x0, device='cpu')
    lower, upper = bounded.compute_bounds(x=(_box(x0, eps),), method='IBP')

    y_l, y_u = (x0 - eps).view(2, 5, 3), (x0 + eps).view(2, 5, 3)
    mid, diff = (y_l + y_u) / 2, (y_u - y_l) / 2
    center, dev = model.W.matmul(mid), model.W.abs().matmul(diff)
    torch.testing.assert_close(lower, center - dev, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(upper, center + dev, atol=1e-6, rtol=1e-5)


def test_wide_net_diag_ibp_scales():
    # regression for the memory blowup: before the constant-left matmul fast
    # path, IBP on this graph broadcast [batch, width, width, d] tensors
    model = _mlp(nn.Sigmoid, [4, 512, 512, 1], seed=8)
    torch.manual_seed(8)
    x0 = torch.rand(4, 4) * 2 - 1
    bounded = _bounded_diag(model, x0)
    lower, upper = bounded.compute_hessian_diag_bounds(
        _box(x0, 0.05), method='IBP')
    assert lower.shape == (4, 1, 4)
    diag = _autograd_diags(model, x0)
    assert (diag >= lower - 1e-5).all() and (diag <= upper + 1e-5).all()


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
    model, in_dim = _case('sigmoid')
    reference = _HardcodedTrace(
        model, SigmoidGradOp.apply, SigmoidSecondGradOp.apply)
    torch.manual_seed(4)
    x0 = torch.randn(2, in_dim).clamp(-0.5, 0.5)
    eps = 0.1

    bounded = _bounded_trace(model, x0)
    lower, upper = bounded.compute_hessian_trace_bounds(
        _box(x0, eps), method=method)

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
