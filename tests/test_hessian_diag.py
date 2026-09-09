"""Checks for DirectHessianDiagOP, the forward-mode Hessian diagonal.

The expanded graph's forward value must equal the autograd Hessian diagonal
exactly (the recursion is the chain rule evaluated at a point). IBP and CROWN
bounds must enclose the true diagonal on dense samples of the input box,
across activations, depths, a multi-output head, and a composite model that
exercises the Add/Sub/Mul builders. Summing the diagonal must reproduce the
trace graph's forward value exactly, and a zero-radius box must pinch the IBP
bounds onto the exact diagonal. Soundness against autograd is the ground
truth throughout; no ordering against the full-Hessian constructions is
asserted, since different relaxations need not dominate each other.
"""

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.hessian_diag import DirectHessianDiagOP
from auto_LiRPA.hessian_trace import DirectHessianTraceOP
from auto_LiRPA.perturbations import PerturbationLpNorm


class _DiagWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return DirectHessianDiagOP.apply(self.model(x), x)


class _TraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return DirectHessianTraceOP.apply(self.model(x), x)


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


MODEL_CASES = {
    'sigmoid': lambda: _mlp(nn.Sigmoid, [2, 4, 4, 1]),
    'softplus': lambda: _mlp(nn.Softplus, [2, 4, 4, 1]),
    'tanh': lambda: _mlp(nn.Tanh, [2, 4, 4, 1]),
    'sigmoid_deep': lambda: _mlp(nn.Sigmoid, [3, 5, 4, 5, 4, 1]),
    'sigmoid_multi_output': lambda: _mlp(nn.Sigmoid, [2, 4, 4, 3]),
    'composite': _Composite,
}


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
            H = torch.autograd.functional.hessian(scalar_forward, xb)
            per_out.append(torch.diagonal(H))
        rows.append(torch.stack(per_out))
    return torch.stack(rows)


def _bounded_diag_module(model, x0):
    return BoundedModule(_DiagWrapper(model), x0, device='cpu')


def _box(x0, eps):
    return BoundedTensor(x0, PerturbationLpNorm(norm=float('inf'), eps=eps))


@pytest.mark.parametrize('name', list(MODEL_CASES))
def test_forward_value_matches_autograd(name):
    model = MODEL_CASES[name]()
    in_dim = 3 if name == 'sigmoid_deep' else 2
    torch.manual_seed(1)
    x0 = torch.randn(3, in_dim)
    value = _bounded_diag_module(model, x0)(x0)
    expected = _autograd_diags(model, x0)
    assert value.shape == expected.shape
    torch.testing.assert_close(value, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('method', ['IBP', 'backward'])
@pytest.mark.parametrize('name', list(MODEL_CASES))
def test_bounds_enclose_true_diag(name, method):
    model = MODEL_CASES[name]()
    in_dim = 3 if name == 'sigmoid_deep' else 2
    torch.manual_seed(2)
    x0 = torch.randn(2, in_dim)
    eps = 0.15
    bounded = _bounded_diag_module(model, x0)
    lb, ub = bounded.compute_hessian_diag_bounds(_box(x0, eps), method=method)

    torch.manual_seed(3)
    for _ in range(100):
        xs = x0 + (torch.rand_like(x0) * 2 - 1) * eps
        diag = _autograd_diags(model, xs)
        assert (diag >= lb - 1e-5).all(), f'lower bound violated ({name}, {method})'
        assert (diag <= ub + 1e-5).all(), f'upper bound violated ({name}, {method})'


@pytest.mark.parametrize('name', ['sigmoid', 'sigmoid_deep', 'composite'])
def test_diag_sum_equals_trace_forward(name):
    # the diagonal is the trace with the input-dim reduction deferred, so the
    # two expanded graphs must agree exactly at any evaluation point
    model = MODEL_CASES[name]()
    in_dim = 3 if name == 'sigmoid_deep' else 2
    torch.manual_seed(4)
    x0 = torch.randn(3, in_dim)
    diag_val = _bounded_diag_module(model, x0)(x0)
    trace_val = BoundedModule(_TraceWrapper(model), x0, device='cpu')(x0)
    torch.testing.assert_close(diag_val.sum(dim=-1), trace_val,
                               atol=1e-6, rtol=1e-5)


def test_zero_radius_box_is_tight():
    # at eps = 0 IBP degenerates to the exact forward pass, so the bounds must
    # pinch onto the autograd diagonal
    model = MODEL_CASES['sigmoid']()
    torch.manual_seed(5)
    x0 = torch.randn(2, 2)
    bounded = _bounded_diag_module(model, x0)
    lb, ub = bounded.compute_hessian_diag_bounds(_box(x0, 0.0), method='IBP')
    expected = _autograd_diags(model, x0)
    torch.testing.assert_close(lb, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(ub, expected, atol=1e-5, rtol=1e-4)


def test_bounds_shape_and_ordering():
    model = MODEL_CASES['sigmoid_multi_output']()
    torch.manual_seed(6)
    x0 = torch.randn(2, 2)
    bounded = _bounded_diag_module(model, x0)
    lb, ub = bounded.compute_hessian_diag_bounds(_box(x0, 0.05), method='IBP')
    assert lb.shape == ub.shape == (2, 3, 2)
    assert (lb <= ub + 1e-8).all()


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
    lb, ub = bounded.compute_bounds(x=(_box(x0, eps),), method='IBP')

    y_l, y_u = (x0 - eps).view(2, 5, 3), (x0 + eps).view(2, 5, 3)
    mid, diff = (y_l + y_u) / 2, (y_u - y_l) / 2
    center, dev = model.W.matmul(mid), model.W.abs().matmul(diff)
    torch.testing.assert_close(lb, center - dev, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ub, center + dev, atol=1e-6, rtol=1e-5)


def test_wide_net_diag_ibp_scales():
    # regression for the memory blowup: before the constant-left matmul fast
    # path, IBP on this graph broadcast [batch, width, width, d] tensors
    model = _mlp(nn.Sigmoid, [4, 512, 512, 1], seed=8)
    torch.manual_seed(8)
    x0 = torch.rand(4, 4) * 2 - 1
    bounded = _bounded_diag_module(model, x0)
    lb, ub = bounded.compute_hessian_diag_bounds(_box(x0, 0.05), method='IBP')
    assert lb.shape == (4, 1, 4)
    diag = _autograd_diags(model, x0)
    assert (diag >= lb - 1e-5).all() and (diag <= ub + 1e-5).all()
