"""Compare primal-target and direct Jacobian-target alpha-CROWN.

This experiment is intentionally small and scalar-valued so the resulting
bounds are easy to interpret. The nested sigmoid model

    y = sigmoid(scale_outer * sigmoid(scale_inner * x + bias_inner)
                + bias_outer)

has a Jacobian graph with a product of two input-dependent local gradients. The
historical Jacobian path optimizes the primal graph first and then runs ordinary
CROWN on that derivative graph. The experimental path optimizes the expanded
Jacobian graph itself.  The tables include a ``jacobian-alpha-fixed-grad``
control that disables ``BoundTanhGrad``/``BoundSigmoidGrad`` alpha parameters,
so the incremental effect of directly optimizing gradient activations is visible.

The script contains two cases:

* a 1D nested-sigmoid control case with almost no useful primal-bound slack,
  which isolates looseness created inside the derivative graph;
* a small 2D mixed network where hidden-state interval quality can matter in
  addition to derivative-graph relaxations.
"""

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.jacobian import JacobianOP
from auto_LiRPA.perturbations import PerturbationLpNorm


class NestedSigmoid(nn.Module):
    """A tiny smooth model whose Jacobian contains a nontrivial product."""

    def __init__(
            self,
            scale_inner: float = 3.0,
            bias_inner: float = -0.4,
            scale_outer: float = 2.0,
            bias_outer: float = 0.2):
        super().__init__()
        self.scale_inner = scale_inner
        self.bias_inner = bias_inner
        self.scale_outer = scale_outer
        self.bias_outer = bias_outer

    def forward(self, x):
        hidden = torch.sigmoid(self.scale_inner * x + self.bias_inner)
        return torch.sigmoid(self.scale_outer * hidden + self.bias_outer)


class MixedSigmoidNetwork(nn.Module):
    """A small dense network with coupled hidden states."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(3, 2)
        self.linear3 = nn.Linear(2, 1)
        with torch.no_grad():
            self.linear1.weight.copy_(torch.tensor([
                [2.2, -1.4],
                [-1.5, 2.0],
                [1.0, 1.3],
            ]))
            self.linear1.bias.copy_(torch.tensor([0.1, -0.2, 0.05]))
            self.linear2.weight.copy_(torch.tensor([
                [1.4, -1.2, 0.8],
                [-0.9, 1.1, 1.3],
            ]))
            self.linear2.bias.copy_(torch.tensor([-0.1, 0.2]))
            self.linear3.weight.copy_(torch.tensor([[1.0, -1.1]]))
            self.linear3.bias.zero_()

    def forward(self, x):
        hidden1 = torch.sigmoid(self.linear1(x))
        hidden2 = torch.sigmoid(self.linear2(hidden1))
        return self.linear3(hidden2)


class JacobianWrapper(nn.Module):
    """Expose the Jacobian of a scalar model as the bounded output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return JacobianOP.apply(self.model(x), x)


@dataclass
class BoundSummary:
    method: str
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass
class VectorBoundSummary:
    method: str
    lower: torch.Tensor
    upper: torch.Tensor

    @property
    def widths(self) -> torch.Tensor:
        return self.upper - self.lower

    @property
    def total_width(self) -> float:
        return self.widths.sum().item()

    @property
    def max_width(self) -> float:
        return self.widths.max().item()


def empirical_1d_jacobian_bounds(
        model: nn.Module,
        center: torch.Tensor,
        eps: float,
        samples: int) -> BoundSummary:
    """Approximate the exact Jacobian range on a 1D interval by sampling."""
    grid = torch.linspace(
        center.item() - eps, center.item() + eps,
        steps=samples, device=center.device, dtype=center.dtype)
    jacobians = []
    for point in grid:
        sample = point.reshape_as(center).detach().requires_grad_(True)
        output = model(sample)
        jacobians.append(torch.autograd.grad(output.sum(), sample)[0].item())
    return BoundSummary(
        method='sampled',
        lower=min(jacobians),
        upper=max(jacobians))


def empirical_box_jacobian_bounds(
        model: nn.Module,
        center: torch.Tensor,
        eps: float,
        grid_size: int) -> VectorBoundSummary:
    """Approximate a Jacobian box on a 2D Linf ball using a dense grid."""
    assert center.numel() == 2
    axes = [
        torch.linspace(
            center[0, i].item() - eps, center[0, i].item() + eps,
            steps=grid_size, device=center.device, dtype=center.dtype)
        for i in range(center.numel())
    ]
    jacobians = []
    for x0 in axes[0]:
        for x1 in axes[1]:
            sample = torch.stack([x0, x1]).reshape_as(center)
            sample = sample.detach().requires_grad_(True)
            output = model(sample)
            jacobians.append(torch.autograd.grad(output.sum(), sample)[0])
    jacobians = torch.stack(jacobians)
    return VectorBoundSummary(
        method='sampled',
        lower=jacobians.min(dim=0).values.flatten(),
        upper=jacobians.max(dim=0).values.flatten())


def make_bounded_model(
        model_factory: Callable[[], nn.Module],
        center: torch.Tensor,
        iterations: int,
        device: str,
        refine_intermediate_bounds: bool = False,
        disable_gradient_optimization: bool = False) -> BoundedModule:
    """Build a fresh bounded Jacobian graph for one measurement."""
    model = model_factory().to(device=device, dtype=center.dtype)
    bound_opts = {'optimize_bound_args': {
        'iteration': iterations,
        'fix_interm_bounds': not refine_intermediate_bounds,
    }}
    if disable_gradient_optimization:
        bound_opts['disable_optimization'] = [
            'BoundTanhGrad', 'BoundSigmoidGrad']
    return BoundedModule(
        JacobianWrapper(model),
        center,
        bound_opts=bound_opts,
        device=device)


def compute_scalar_lirpa_bounds(
        model_factory: Callable[[], nn.Module],
        center: torch.Tensor,
        eps: float,
        iterations: int,
        device: str) -> List[BoundSummary]:
    """Run the three Jacobian bound modes we want to compare."""
    bounded_input = BoundedTensor(
        center, PerturbationLpNorm(norm=float('inf'), eps=eps))

    summaries = []
    bounded_model = make_bounded_model(
        model_factory, center, iterations, device)
    lower, upper = bounded_model.compute_jacobian_bounds(
        bounded_input, optimize=False)
    summaries.append(BoundSummary('crown', lower.item(), upper.item()))

    bounded_model = make_bounded_model(
        model_factory, center, iterations, device)
    lower, upper = bounded_model.compute_jacobian_bounds(
        bounded_input, optimize_target='primal')
    summaries.append(BoundSummary(
        'primal-alpha', lower.item(), upper.item()))

    bounded_model = make_bounded_model(
        model_factory, center, iterations, device,
        refine_intermediate_bounds=True)
    lower, upper = bounded_model.compute_jacobian_bounds(
        bounded_input, optimize_target='primal')
    summaries.append(BoundSummary(
        'primal-alpha-refine', lower.item(), upper.item()))

    bounded_model = make_bounded_model(
        model_factory, center, iterations, device,
        disable_gradient_optimization=True)
    lower, upper = bounded_model.compute_jacobian_bounds(
        bounded_input, optimize_target='jacobian')
    summaries.append(BoundSummary(
        'jacobian-alpha-fixed-grad', lower.item(), upper.item()))

    bounded_model = make_bounded_model(
        model_factory, center, iterations, device)
    lower, upper = bounded_model.compute_jacobian_bounds(
        bounded_input, optimize_target='jacobian')
    summaries.append(BoundSummary(
        'jacobian-alpha', lower.item(), upper.item()))

    return summaries


def compute_vector_lirpa_bounds(
        model_factory: Callable[[], nn.Module],
        center: torch.Tensor,
        eps: float,
        iterations: int,
        device: str) -> List[VectorBoundSummary]:
    """Run the same comparison for a vector-valued Jacobian output."""
    bounded_input = BoundedTensor(
        center, PerturbationLpNorm(norm=float('inf'), eps=eps))

    summaries = []
    configs: List[Tuple[str, bool, str, bool]] = [
        ('crown', False, 'primal', False),
        ('primal-alpha', False, 'primal', False),
        ('primal-alpha-refine', True, 'primal', False),
        ('jacobian-alpha-fixed-grad', False, 'jacobian', True),
        ('jacobian-alpha', False, 'jacobian', False),
    ]
    for (name, refine_intermediate_bounds, optimize_target,
         disable_gradient_optimization) in configs:
        bounded_model = make_bounded_model(
            model_factory, center, iterations, device,
            refine_intermediate_bounds=refine_intermediate_bounds,
            disable_gradient_optimization=disable_gradient_optimization)
        if name == 'crown':
            lower, upper = bounded_model.compute_jacobian_bounds(
                bounded_input, optimize=False)
        else:
            lower, upper = bounded_model.compute_jacobian_bounds(
                bounded_input, optimize_target=optimize_target)
        summaries.append(VectorBoundSummary(
            name, lower.flatten().detach(), upper.flatten().detach()))
    return summaries


def print_scalar_table(eps: float, empirical: BoundSummary,
                       summaries: Iterable[BoundSummary]) -> None:
    """Print one compact comparison table for a perturbation radius."""
    print(f'\neps = {eps:g}')
    print(f'{"method":<22} {"lower":>12} {"upper":>12} {"width":>12} '
          f'{"excess width":>14}')
    print('-' * 76)
    rows = [empirical, *summaries]
    for row in rows:
        excess_width = row.width - empirical.width
        print(f'{row.method:<22} {row.lower:>12.6f} {row.upper:>12.6f} '
              f'{row.width:>12.6f} {excess_width:>14.6f}')


def print_vector_table(eps: float, empirical: VectorBoundSummary,
                       summaries: Iterable[VectorBoundSummary]) -> None:
    """Print componentwise and aggregate widths for the 2D case."""
    print(f'\neps = {eps:g}')
    print(f'{"method":<22} {"dx0 width":>12} {"dx1 width":>12} '
          f'{"total width":>12} {"excess total":>14}')
    print('-' * 78)
    rows = [empirical, *summaries]
    for row in rows:
        excess_total = row.total_width - empirical.total_width
        widths = row.widths
        print(f'{row.method:<22} {widths[0].item():>12.6f} '
              f'{widths[1].item():>12.6f} {row.total_width:>12.6f} '
              f'{excess_total:>14.6f}')


def run_1d_control_case(
        eps_values: Iterable[float],
        center_value: float,
        iterations: int,
        samples: int,
        device: str) -> None:
    """Run the comparison over several perturbation radii."""
    torch.manual_seed(0)
    center = torch.tensor([[center_value]], dtype=torch.float32, device=device)
    model = NestedSigmoid().to(device=device, dtype=torch.float32)

    print('Model: y = sigmoid(2 * sigmoid(3 * x - 0.4) + 0.2)')
    print(f'center = {center_value:g}, alpha iterations = {iterations}, '
          f'sampled points = {samples}')
    for eps in eps_values:
        empirical = empirical_1d_jacobian_bounds(model, center, eps, samples)
        summaries = compute_scalar_lirpa_bounds(
            NestedSigmoid, center, eps, iterations=iterations, device=device)
        print_scalar_table(eps, empirical, summaries)


def run_2d_mixed_case(
        eps_values: Iterable[float],
        iterations: int,
        grid_size: int,
        device: str) -> None:
    """Run a coupled 2D model where primal intervals can also matter."""
    center = torch.tensor([[0.1, -0.2]], dtype=torch.float32, device=device)
    model = MixedSigmoidNetwork().to(device=device, dtype=torch.float32)
    print('\n\nModel: 2D mixed sigmoid network '
          '(Linear 2->3, Sigmoid, Linear 3->2, Sigmoid, Linear 2->1)')
    print(f'center = {center.flatten().tolist()}, alpha iterations = {iterations}, '
          f'grid size = {grid_size} x {grid_size}')
    for eps in eps_values:
        empirical = empirical_box_jacobian_bounds(
            model, center, eps, grid_size)
        summaries = compute_vector_lirpa_bounds(
            MixedSigmoidNetwork, center, eps,
            iterations=iterations, device=device)
        print_vector_table(eps, empirical, summaries)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare alpha-CROWN targets for Jacobian bounds.')
    parser.add_argument(
        '--eps', type=float, nargs='+', default=[0.1, 0.3, 0.5, 1.0],
        help='Perturbation radii to evaluate.')
    parser.add_argument(
        '--center', type=float, default=0.1,
        help='Center of the one-dimensional input interval.')
    parser.add_argument(
        '--iterations', type=int, default=20,
        help='Number of alpha-CROWN optimization iterations.')
    parser.add_argument(
        '--samples', type=int, default=2001,
        help='Grid samples for the empirical Jacobian range.')
    parser.add_argument(
        '--mixed-grid-size', type=int, default=41,
        help='Samples per input dimension for the 2D mixed-network grid.')
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
        help='PyTorch device.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_1d_control_case(
        eps_values=args.eps,
        center_value=args.center,
        iterations=args.iterations,
        samples=args.samples,
        device=args.device)
    run_2d_mixed_case(
        eps_values=args.eps,
        iterations=args.iterations,
        grid_size=args.mixed_grid_size,
        device=args.device)
