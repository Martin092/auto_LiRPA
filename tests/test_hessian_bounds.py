import itertools

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.hessian import HessianOP
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


def test_native_linear_hessian_bounds_are_zero():
    model = nn.Linear(3, 2).double()
    x0 = torch.zeros(1, 3, dtype=torch.double)
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
    model = nn.ReLU().double()
    x0 = torch.zeros(1, 3, dtype=torch.double)

    with pytest.raises(NotImplementedError):
        BoundedModule(_HessianWrapper(model), x0)


def test_native_direct_softplus_hessian_bounds():
    model = nn.Softplus().double()
    x0 = torch.tensor([[-0.2, 0.0, 0.4]], dtype=torch.double)
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
            x0[0, i] - eps, x0[0, i] + eps, steps=9, dtype=torch.double)
        for i in range(x0.numel())
    ]
    for point in itertools.product(*grids):
        hessian = _output_hessians(
            model, torch.tensor([point], dtype=torch.double))
        assert torch.all(hessian >= lower - 1e-10)
        assert torch.all(hessian <= upper + 1e-10)


def test_native_direct_sigmoid_hessian_bounds():
    model = nn.Sigmoid().double()
    x0 = torch.tensor([[-0.2, 0.0, 0.4]], dtype=torch.double)
    bounded = BoundedModule(_HessianWrapper(model), x0)

    forward_hessian = bounded(x0)
    expected_forward = _output_hessians(model, x0)
    assert torch.allclose(forward_hessian, expected_forward)

    eps = 0.1
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    grids = [
        torch.linspace(
            x0[0, i] - eps, x0[0, i] + eps, steps=9, dtype=torch.double)
        for i in range(x0.numel())
    ]
    for method in ['IBP', 'backward']:
        lower, upper = bounded.compute_hessian_bounds(x, method=method)
        for point in itertools.product(*grids):
            hessian = _output_hessians(
                model, torch.tensor([point], dtype=torch.double))
            assert torch.all(hessian >= lower - 1e-10)
            assert torch.all(hessian <= upper + 1e-10)


def test_native_sigmoid_linear_network_hessian_contains_sampled_points():
    torch.manual_seed(5)
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Sigmoid(),
        nn.Linear(3, 1),
    ).double()
    x0 = torch.tensor([[0.1, -0.2]], dtype=torch.double)
    bounded = BoundedModule(_HessianWrapper(model), x0)

    eps = 0.05
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    grids = [
        torch.linspace(
            x0[0, i] - eps, x0[0, i] + eps, steps=9, dtype=torch.double)
        for i in range(x0.numel())
    ]
    for method in ['IBP', 'backward']:
        lower, upper = bounded.compute_hessian_bounds(x, method=method)
        for point in itertools.product(*grids):
            hessian = _scalar_hessian(
                model, torch.tensor([point], dtype=torch.double))
            assert torch.all(hessian >= lower[0, 0] - 1e-10)
            assert torch.all(hessian <= upper[0, 0] + 1e-10)


def test_native_softplus_linear_network_hessian_contains_sampled_points():
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Softplus(),
        nn.Linear(3, 1),
    ).double()
    x0 = torch.tensor([[0.1, -0.2]], dtype=torch.double)
    bounded = BoundedModule(_HessianWrapper(model), x0)

    eps = 0.05
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    lower, upper = bounded.compute_hessian_bounds(x)

    print("Lower: ", lower)
    print("Upper: ", upper)

    grids = [
        torch.linspace(
            x0[0, i] - eps, x0[0, i] + eps, steps=9, dtype=torch.double)
        for i in range(x0.numel())
    ]
    for point in itertools.product(*grids):
        hessian = _scalar_hessian(
            model, torch.tensor([point], dtype=torch.double))
        assert torch.all(hessian >= lower[0, 0] - 1e-10)
        assert torch.all(hessian <= upper[0, 0] + 1e-10)


def test_native_stacked_softplus_hessian_contains_sampled_points():
    torch.manual_seed(3)
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Softplus(),
        nn.Softplus(),
        nn.Linear(2, 1),
    ).double()
    x0 = torch.tensor([[0.05, -0.1]], dtype=torch.double)
    bounded = BoundedModule(_HessianWrapper(model), x0)

    forward_hessian = bounded(x0)
    expected_forward = _scalar_hessian(model, x0)
    assert torch.allclose(forward_hessian[0, 0], expected_forward)

    eps = 0.3
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float('inf'), eps=eps))
    lower, upper = bounded.compute_hessian_bounds(x)
    grids = [
        torch.linspace(
            x0[0, i] - eps, x0[0, i] + eps, steps=7, dtype=torch.double)
        for i in range(x0.numel())
    ]
    #print("Bounds: ", lower, upper)
    for point in itertools.product(*grids):
        hessian = _scalar_hessian(
            model, torch.tensor([point], dtype=torch.double))
        #print("hessian: ", hessian)
        assert torch.all(hessian >= lower[0, 0] - 1e-10)
        assert torch.all(hessian <= upper[0, 0] + 1e-10)


class _HessianWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return HessianOP.apply(self.model(x), x)
