import itertools

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.hessian import DirectHessianOP
from auto_LiRPA.operators.hessian import DoubleJacobianOP
from auto_LiRPA.operators.s_shaped import (
    BoundSigmoidSecondGrad, d2sigmoid, d3sigmoid)
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import logger


def _scalar_hessian(model, x):
    def scalar_forward(inp):
        return model(inp).reshape(-1)[0]

    return torch.autograd.functional.hessian(scalar_forward, x).reshape(
        x.numel(), x.numel())


def _output_hessians(model, x):
    hessians = []
    output = model(x).reshape(-1)
    for output_idx in range(output.numel()):
        def scalar_forward(inp):
            return model(inp).reshape(-1)[output_idx]

        hessian = torch.autograd.functional.hessian(
            scalar_forward, x).reshape(x.numel(), x.numel())
        hessians.append(hessian)
    return torch.stack(hessians).reshape(
        x.shape[0], -1, *x.shape[1:], *x.shape[1:])


def _make_grid(x0, eps, steps):
    return [
        torch.linspace(
            x0[0, i] - eps, x0[0, i] + eps, steps=steps)
        for i in range(x0.numel())
    ]


class _HessianWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return DirectHessianOP.apply(self.model(x), x)


class _DoubleJacobianWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return DoubleJacobianOP.apply(self.model(x), x)


def test_native_linear_hessian_bounds_are_zero():
    model = nn.Linear(3, 2)
    x0 = torch.zeros(1, 3)
    bounded = BoundedModule(_HessianWrapper(model), x0)

    forward_hessian = bounded(x0)
    assert forward_hessian.shape == (1, 2, 3, 3)
    assert torch.equal(forward_hessian, torch.zeros_like(forward_hessian))

    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=0.1))
    lower, upper = bounded.compute_hessian_bounds(x)

    assert torch.equal(lower, torch.zeros_like(lower))
    assert torch.equal(upper, torch.zeros_like(upper))


def test_native_relu_hessian_is_explicitly_unsupported():
    model = nn.ReLU()
    x0 = torch.zeros(1, 3)

    with pytest.raises(NotImplementedError):
        BoundedModule(_HessianWrapper(model), x0)


def test_native_direct_softplus_hessian_bounds():
    model = nn.Softplus()
    x0 = torch.tensor([[-0.2, 0.0, 0.4]])
    bounded = BoundedModule(_HessianWrapper(model), x0)

    forward_hessian = bounded(x0)
    expected_forward = _output_hessians(model, x0)
    assert torch.allclose(forward_hessian, expected_forward)

    eps = 0.1
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    lower, upper = bounded.compute_hessian_bounds(x)

    grids = [
        torch.linspace(
            x0[0, i] - eps, x0[0, i] + eps, steps=9)
        for i in range(x0.numel())
    ]
    for point in itertools.product(*grids):
        hessian = _output_hessians(
            model, torch.tensor([point]))
        assert torch.all(hessian >= lower - 1e-5)
        assert torch.all(hessian <= upper + 1e-5)


def test_native_direct_sigmoid_hessian_bounds():
    model = nn.Sigmoid()
    x0 = torch.tensor([[-0.2, 0.0, 0.4]])
    bounded = BoundedModule(_HessianWrapper(model), x0)

    forward_hessian = bounded(x0)
    expected_forward = _output_hessians(model, x0)
    assert torch.allclose(forward_hessian, expected_forward)

    eps = 0.1
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    grids = [
        torch.linspace(
            x0[0, i] - eps, x0[0, i] + eps, steps=9)
        for i in range(x0.numel())
    ]
    for method in ['IBP', 'backward']:
        lower, upper = bounded.compute_hessian_bounds(x, method=method)
        for point in itertools.product(*grids):
            hessian = _output_hessians(
                model, torch.tensor([point]))
            assert torch.all(hessian >= lower - 1e-5)
            assert torch.all(hessian <= upper + 1e-5)


def test_sigmoid_second_grad_precompute_masks_and_symmetry():
    op = BoundSigmoidSecondGrad(attr={'device': torch.device('cpu')})
    op.precompute_relaxation(x_limit=5)

    x = torch.tensor([[-3., -1., 1., 3.]])
    upper_d_lower, upper_d_upper, upper_has_d_lower, upper_has_d_upper = (
        op.retrieve_from_precompute(x))
    lower_d_lower, lower_d_upper, lower_has_d_lower, lower_has_d_upper = (
        op.retrieve_from_precompute(x, flip=True))

    assert torch.equal(
        upper_has_d_lower, torch.tensor([[False, True, False, True]]))
    assert torch.equal(
        upper_has_d_upper, torch.tensor([[False, False, True, True]]))
    assert torch.equal(
        lower_has_d_lower, torch.tensor([[True, True, False, False]]))
    assert torch.equal(
        lower_has_d_upper, torch.tensor([[True, False, True, False]]))

    # Lower-endpoint thresholds are recovered by odd symmetry, which negates
    # tangent points and swaps lower/upper roles.
    assert torch.allclose(lower_d_lower[:, 0], -upper_d_upper[:, 3])
    assert torch.allclose(lower_d_upper[:, 0], -upper_d_lower[:, 3])
    assert torch.allclose(lower_d_lower[:, 1], -upper_d_upper[:, 2])
    assert torch.allclose(lower_d_upper[:, 2], -upper_d_lower[:, 1])

    # The lower-endpoint view exposes the positive-branch lower tangents for
    # negative endpoints that are not part of the upper-endpoint table itself.
    assert torch.all(lower_d_lower[:, :2] > 0)
    assert torch.all(
        lower_d_lower[:, :2] <= op.extreme_point + 1e-5)

    upper_lower_line_at_neg_middle = (
        d3sigmoid(upper_d_lower[:, 1]) * (x[:, 1] - upper_d_lower[:, 1])
        + d2sigmoid(upper_d_lower[:, 1]))
    upper_upper_line_at_pos_middle = (
        d3sigmoid(upper_d_upper[:, 2]) * (x[:, 2] - upper_d_upper[:, 2])
        + d2sigmoid(upper_d_upper[:, 2]))
    upper_lower_line_at_pos_tail = (
        d3sigmoid(upper_d_lower[:, 3]) * (x[:, 3] - upper_d_lower[:, 3])
        + d2sigmoid(upper_d_lower[:, 3]))
    lower_lower_line_at_neg_tail = (
        d3sigmoid(lower_d_lower[:, 0]) * (x[:, 0] - lower_d_lower[:, 0])
        + d2sigmoid(lower_d_lower[:, 0]))
    lower_upper_line_at_neg_tail = (
        d3sigmoid(lower_d_upper[:, 0]) * (x[:, 0] - lower_d_upper[:, 0])
        + d2sigmoid(lower_d_upper[:, 0]))

    assert torch.all(
        upper_lower_line_at_neg_middle <= d2sigmoid(x[:, 1]) + 1e-5)
    assert torch.all(
        upper_upper_line_at_pos_middle >= d2sigmoid(x[:, 2]) - 1e-5)
    assert torch.all(
        upper_lower_line_at_pos_tail <= d2sigmoid(x[:, 3]) + 1e-5)
    assert torch.all(
        lower_lower_line_at_neg_tail <= d2sigmoid(x[:, 0]) + 1e-5)
    assert torch.all(
        lower_upper_line_at_neg_tail >= d2sigmoid(x[:, 0]) - 1e-5)


def test_sigmoid_second_grad_piecewise_case_relaxations_are_sound():
    """Representative intervals from the curvature/crossing case table."""
    op = BoundSigmoidSecondGrad(
        attr={'device': torch.device('cpu')},
        options={'sigmoid_second_grad_relaxation': 'piecewise'})

    lower = torch.tensor([
        -4., -2., 0.2, 2.5,   # fully convex / concave regions
        -4., -1., -1., 0.5,   # lower-bound crossing cases
        -4., -1., -4., 1.5,   # upper-bound crossing cases
    ])
    upper = torch.tensor([
        -3., -1., 1.0, 4.0,
        -1., 0.5, 3.0, 3.0,
        -1., 0.5, 1.0, 3.0,
    ])

    class SimpleBoundedInput:
        def __init__(self, l, u):
            self.lower = l
            self.upper = u

    x = SimpleBoundedInput(lower, upper)
    op.init_linear_relaxation(x)
    op.bound_relax(x, init=False)

    grid_t = torch.linspace(0., 1., steps=257).unsqueeze(1)
    grid = lower.unsqueeze(0) + grid_t * (upper - lower).unsqueeze(0)
    y = d2sigmoid(grid)
    lower_line = op.lw.unsqueeze(0) * grid + op.lb.unsqueeze(0)
    upper_line = op.uw.unsqueeze(0) * grid + op.ub.unsqueeze(0)

    assert torch.all(lower_line <= y + 1e-5)
    assert torch.all(upper_line >= y - 1e-5)
    # These intervals are chosen so that each representative case should get
    # an actual linear relaxation, not merely retain the initial IBP constant.
    assert torch.all(op.lw.abs() > 1e-6)
    assert torch.all(op.uw.abs() > 1e-6)


def test_sigmoid_second_grad_piecewise_relaxations_cover_boundary_grid():
    """Check soundness across intervals around all curvature breakpoints."""
    op = BoundSigmoidSecondGrad(
        attr={'device': torch.device('cpu')},
        options={'sigmoid_second_grad_relaxation': 'piecewise'})
    endpoints = torch.tensor([
        -5., -3., -2.3, -2.2, -1., -0.1,
        0., 0.1, 1., 2.2, 2.3, 3., 5.,
    ])
    lower, upper = zip(*[
        (endpoints[i], endpoints[j])
        for i in range(endpoints.numel())
        for j in range(i + 1, endpoints.numel())
    ])
    lower = torch.stack(lower)
    upper = torch.stack(upper)

    class SimpleBoundedInput:
        def __init__(self, l, u):
            self.lower = l
            self.upper = u

    x = SimpleBoundedInput(lower, upper)
    op.init_linear_relaxation(x)
    op.bound_relax(x, init=False)

    grid_t = torch.linspace(0., 1., steps=257).unsqueeze(1)
    grid = lower.unsqueeze(0) + grid_t * (upper - lower).unsqueeze(0)
    y = d2sigmoid(grid)
    lower_line = op.lw.unsqueeze(0) * grid + op.lb.unsqueeze(0)
    upper_line = op.uw.unsqueeze(0) * grid + op.ub.unsqueeze(0)

    assert torch.all(lower_line <= y + 1e-5)
    assert torch.all(upper_line >= y - 1e-5)


def test_sigmoid_second_grad_tangent_relaxation_is_fixed():
    tangent = BoundSigmoidSecondGrad(
        attr={'device': torch.device('cpu')},
        options={'sigmoid_second_grad_relaxation': 'tangent'})
    same_slope = BoundSigmoidSecondGrad(
        attr={'device': torch.device('cpu')},
        options={'sigmoid_second_grad_relaxation': 'same-slope'})
    piecewise = BoundSigmoidSecondGrad(
        attr={'device': torch.device('cpu')},
        options={'sigmoid_second_grad_relaxation': 'piecewise'})

    assert tangent.sigmoid_second_grad_relaxation == 'tangent'
    assert same_slope.sigmoid_second_grad_relaxation == 'tangent'
    assert not tangent.optimizable
    assert not same_slope.optimizable
    assert piecewise.optimizable


def test_native_direct_sigmoid_hessian_alpha_crown_piecewise_bounds():
    """Piecewise sigmoid'' relaxations should be optimizable and sound."""
    model = nn.Sigmoid()
    # Exercise intervals around each outer inflection and around zero.
    x0 = torch.tensor([[-2.3, 0.0, 2.3]])
    bounded = BoundedModule(
        _HessianWrapper(model), x0,
        bound_opts={
            'optimize_bound_args': {'iteration': 2},
            'sigmoid_second_grad_relaxation': 'piecewise',
        })

    eps = 0.2
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    lower, upper = bounded.compute_hessian_bounds(x, method='alpha-CROWN')

    second_grad_nodes = [
        node for node in bounded.nodes()
        if type(node).__name__ == 'BoundSigmoidSecondGrad']
    assert second_grad_nodes
    assert any(node.alpha for node in second_grad_nodes)

    grids = [
        torch.linspace(
            x0[0, i] - eps, x0[0, i] + eps, steps=9)
        for i in range(x0.numel())
    ]
    for point in itertools.product(*grids):
        hessian = _output_hessians(
            model, torch.tensor([point]))
        assert torch.all(hessian >= lower - 1e-5)
        assert torch.all(hessian <= upper + 1e-5)


@pytest.mark.parametrize("wrapper_cls", [_HessianWrapper, _DoubleJacobianWrapper])
@pytest.mark.parametrize("method", ['IBP', 'backward'])
def test_sigmoid_linear_network_soundness(wrapper_cls, method):
    torch.manual_seed(5)
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Sigmoid(),
        nn.Linear(3, 1),
    )
    x0 = torch.tensor([[0.1, -0.2]])
    bounded = BoundedModule(wrapper_cls(model), x0)

    eps = 0.05
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    lower, upper = bounded.compute_hessian_bounds(x, method=method)
    for point in itertools.product(*_make_grid(x0, eps, 9)):
        hessian = _scalar_hessian(
            model, torch.tensor([point]))
        assert torch.all(hessian >= lower[0, 0] - 1e-5)
        assert torch.all(hessian <= upper[0, 0] + 1e-5)


def test_native_sigmoid_linear_network_alpha_crown_piecewise_hessian_contains_sampled_points():
    """Optimized sigmoid'' bounds should stay sound inside a sigmoid network."""
    torch.manual_seed(5)
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Sigmoid(),
        nn.Linear(3, 1),
    )
    x0 = torch.tensor([[0.1, -0.2]])
    bounded = BoundedModule(
        _HessianWrapper(model), x0,
        bound_opts={
            'optimize_bound_args': {'iteration': 2},
            'sigmoid_second_grad_relaxation': 'piecewise',
        })

    eps = 0.05
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    lower, upper = bounded.compute_hessian_bounds(x, method='alpha-CROWN')

    second_grad_nodes = [
        node for node in bounded.nodes()
        if type(node).__name__ == 'BoundSigmoidSecondGrad']
    assert second_grad_nodes
    assert any(node.alpha for node in second_grad_nodes)

    grids = [
        torch.linspace(
            x0[0, i] - eps, x0[0, i] + eps, steps=9)
        for i in range(x0.numel())
    ]
    for point in itertools.product(*grids):
        hessian = _scalar_hessian(
            model, torch.tensor([point]))
        assert torch.all(hessian >= lower[0, 0] - 1e-5)
        assert torch.all(hessian <= upper[0, 0] + 1e-5)


@pytest.mark.parametrize("wrapper_cls", [_HessianWrapper, _DoubleJacobianWrapper])
@pytest.mark.parametrize("method", ['IBP', 'backward'])
def test_softplus_linear_network_soundness(wrapper_cls, method):
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Softplus(),
        nn.Linear(3, 1),
    )
    x0 = torch.tensor([[0.1, -0.2]])
    bounded = BoundedModule(wrapper_cls(model), x0)

    eps = 0.05
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    lower, upper = bounded.compute_hessian_bounds(x, method=method)

    for point in itertools.product(*_make_grid(x0, eps, 9)):
        hessian = _scalar_hessian(
            model, torch.tensor([point]))
        assert torch.all(hessian >= lower[0, 0] - 1e-5)
        assert torch.all(hessian <= upper[0, 0] + 1e-5)


@pytest.mark.parametrize("wrapper_cls", [_HessianWrapper, _DoubleJacobianWrapper])
def test_stacked_softplus_soundness(wrapper_cls):
    torch.manual_seed(3)
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Softplus(),
        nn.Linear(2, 2),
        nn.Softplus(),
    )
    x0 = torch.tensor([[0.05, -0.1]])
    bounded = BoundedModule(wrapper_cls(model), x0)

    forward_hessian = bounded(x0)
    expected_forward = _scalar_hessian(model, x0)
    assert torch.allclose(forward_hessian[0, 0], expected_forward)

    eps = 0.3
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    lower, upper = bounded.compute_hessian_bounds(x)
    for point in itertools.product(*_make_grid(x0, eps, 7)):
        hessian = _scalar_hessian(
            model, torch.tensor([point]))
        assert torch.all(hessian >= lower[0, 0] - 1e-5)
        assert torch.all(hessian <= upper[0, 0] + 1e-5)


def test_direct_and_double_jacobian_agree_on_forward_value():
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Sigmoid(),
        nn.Linear(3, 1),
    )
    x0 = torch.tensor([[0.2, -0.3]])

    direct_bounded = BoundedModule(_HessianWrapper(model), x0)
    double_bounded = BoundedModule(_DoubleJacobianWrapper(model), x0)

    direct_fwd = direct_bounded(x0)
    double_fwd = double_bounded(x0)

    assert torch.allclose(direct_fwd, double_fwd)
