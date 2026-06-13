"""BoundMatMul with a constant left operand and a perturbed right operand.

This combination routes through the swap_x_and_weight trick: x @ y is
rewritten as (y^T x^T)^T so the perturbed operand plays the role of the
input. The final output bounds were always fine, but the intermediate
bounds stored on the matmul node used to come out in transposed layout.
The backward pass for an intermediate node starts from an eyeC identity
spec over the flattened (m, n) output, and the swap branch kept it as an
identity after transposing the output, even though entry (i, j) of the
transposed output lives at flattened position j*m + i, not i*n + j. Any
relaxation built on those intermediate bounds, like a square following
the matmul, then becomes unsound.
"""

import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundMatMul
from auto_LiRPA.perturbations import PerturbationLpNorm


class ConstFirstMatMul(nn.Module):
    def __init__(self, const):
        super().__init__()
        self.register_buffer('const', const)

    def forward(self, x):
        return self.const.matmul(x)


class SquaredAfterMatMul(nn.Module):
    """The square forces CROWN to compute intermediate bounds for the
    matmul node, which is where the transposed layout used to surface."""

    def __init__(self, const):
        super().__init__()
        self.matmul = ConstFirstMatMul(const)

    def forward(self, x):
        y = self.matmul(x)
        return (y ** 2).flatten(1).sum(dim=1, keepdim=True)


def _setup(seed=0, batch=2, k=3, m=2, n=4, eps=0.1):
    torch.manual_seed(seed)
    const = torch.randn(k, m)
    x0 = torch.randn(batch, m, n)
    box = BoundedTensor(x0, PerturbationLpNorm(
        norm=float('inf'), x_L=x0 - eps, x_U=x0 + eps))
    return const, x0, box, eps


def _samples(x0, eps, count=500, seed=1):
    torch.manual_seed(seed)
    offsets = torch.rand(count, *x0.shape) * 2 - 1
    return x0.unsqueeze(0) + eps * offsets


def test_final_output_bounds_sound():
    const, x0, box, eps = _setup()
    bounded = BoundedModule(ConstFirstMatMul(const), x0, device='cpu')
    lower, upper = bounded.compute_bounds(x=(box,), method='backward')
    outs = const.matmul(_samples(x0, eps))
    assert (outs >= lower - 1e-5).all(), (
        f'lower bound crossed by {(lower - outs).max().item():.2e}')
    assert (outs <= upper + 1e-5).all(), (
        f'upper bound crossed by {(outs - upper).max().item():.2e}')


def test_intermediate_bounds_enclose_samples():
    const, x0, box, eps = _setup()
    bounded = BoundedModule(SquaredAfterMatMul(const), x0, device='cpu')
    bounded.compute_bounds(x=(box,), method='backward')
    node = next(n for n in bounded.nodes() if isinstance(n, BoundMatMul))
    outs = const.matmul(_samples(x0, eps))
    assert node.lower.shape == outs.shape[1:]
    assert (outs >= node.lower - 1e-5).all(), (
        'matmul intermediate lower bound crossed by '
        f'{(node.lower - outs).max().item():.2e}')
    assert (outs <= node.upper + 1e-5).all(), (
        'matmul intermediate upper bound crossed by '
        f'{(outs - node.upper).max().item():.2e}')


def test_downstream_of_matmul_sound():
    const, x0, box, eps = _setup()
    model = SquaredAfterMatMul(const)
    bounded = BoundedModule(model, x0, device='cpu')
    lower, upper = bounded.compute_bounds(x=(box,), method='backward')
    for sample in _samples(x0, eps):
        out = model(sample)
        assert (out >= lower - 1e-5).all(), (
            f'lower bound crossed by {(lower - out).max().item():.2e}')
        assert (out <= upper + 1e-5).all(), (
            f'upper bound crossed by {(out - upper).max().item():.2e}')
