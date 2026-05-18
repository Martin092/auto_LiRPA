import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, List

import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.operators.s_shaped import SigmoidSecondGrad


class SecondGradLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.second = SigmoidSecondGrad()

    def forward(self, x):
        preact = self.linear(x)
        g = torch.ones_like(preact).unsqueeze(1)
        out = self.second(g, preact)
        return out.squeeze(1)


class NestedSigmoidSecondGrad(nn.Module):
    def __init__(self,
                 scale_inner: float = 3.0,
                 bias_inner: float = -0.4,
                 scale_outer: float = 2.0,
                 bias_outer: float = 0.2):
        super().__init__()
        self.inner = SecondGradLayer(1, 1)
        self.outer = SecondGradLayer(1, 1)
        # initialize linear weights to match the original scalar scales
        with torch.no_grad():
            self.inner.linear.weight.fill_(scale_inner)
            self.inner.linear.bias.fill_(bias_inner)
            self.outer.linear.weight.fill_(scale_outer)
            self.outer.linear.bias.fill_(bias_outer)

    def forward(self, x):
        h = self.inner(x)
        return self.outer(h)


class MixedSigmoidSecondGrad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = SecondGradLayer(2, 3)
        self.layer2 = SecondGradLayer(3, 2)
        self.layer3 = SecondGradLayer(2, 1)
        # copy previous weight matrices into the internal linears
        with torch.no_grad():
            self.layer1.linear.weight.copy_(torch.tensor([
                [2.2, -1.4],
                [-1.5, 2.0],
                [1.0, 1.3],
            ]))
            self.layer1.linear.bias.copy_(torch.tensor([0.1, -0.2, 0.05]))
            self.layer2.linear.weight.copy_(torch.tensor([
                [1.4, -1.2, 0.8],
                [-0.9, 1.1, 1.3],
            ]))
            self.layer2.linear.bias.copy_(torch.tensor([-0.1, 0.2]))
            self.layer3.linear.weight.copy_(torch.tensor([[1.0, -1.1]]))
            self.layer3.linear.bias.zero_()

    def forward(self, x):
        hidden1 = self.layer1(x)
        hidden2 = self.layer2(hidden1)
        return self.layer3(hidden2)


@dataclass
class BoundSummary:
    method: str
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


def empirical_1d_jacobian_bounds(model: nn.Module, center: torch.Tensor, eps: float, samples: int) -> BoundSummary:
    grid = torch.linspace(
        center.item() - eps, center.item() + eps,
        steps=samples, device=center.device, dtype=center.dtype)
    values = []
    for point in grid:
        sample = point.reshape_as(center)
        with torch.no_grad():
            out = model(sample)
        values.append(out.reshape(-1)[0].item())
    return BoundSummary(method='sampled', lower=min(values), upper=max(values))


def empirical_box_secondgrad_bounds(model: nn.Module, center: torch.Tensor, eps: float, grid_size: int) -> BoundSummary:
    assert center.numel() == 2
    axes = [
        torch.linspace(
            center[0, i].item() - eps, center[0, i].item() + eps,
            steps=grid_size, device=center.device, dtype=center.dtype)
        for i in range(center.numel())
    ]
    values = []
    for x0 in axes[0]:
        for x1 in axes[1]:
            sample = torch.stack([x0, x1]).reshape_as(center)
            with torch.no_grad():
                out = model(sample)
            values.append(float(out.reshape(-1)[0].item()))
    return BoundSummary(method='sampled', lower=min(values), upper=max(values))


def make_bounded_model(model_factory: Callable[[], nn.Module], center: torch.Tensor, iterations: int, device: str, relax_method: str = 'tangent', disable_second_grad_opt: bool = False) -> BoundedModule:
    model = model_factory().to(device=device, dtype=center.dtype)
    bound_opts = {
        'optimize_bound_args': {'iteration': iterations},
        'sigmoid_second_grad_relaxation': relax_method,
    }
    if disable_second_grad_opt:
        bound_opts['disable_optimization'] = ['BoundSigmoidSecondGrad']
    return BoundedModule(model, center, bound_opts=bound_opts, device=device)


def compute_scalar_lirpa_bounds(model_factory: Callable[[], nn.Module], center: torch.Tensor, eps: float, iterations: int, device: str) -> List[BoundSummary]:
    bounded_input = BoundedTensor(center, PerturbationLpNorm(norm=float('inf'), eps=eps))

    summaries: List[BoundSummary] = []

    bounded_model = make_bounded_model(model_factory, center, iterations, device)
    lower, upper = bounded_model.compute_bounds(x=(bounded_input,), method='backward')
    summaries.append(BoundSummary('CROWN', lower.reshape(-1)[0].item(), upper.reshape(-1)[0].item()))

    bounded_model = make_bounded_model(model_factory, center, iterations, device)
    lower, upper = bounded_model.compute_bounds(x=(bounded_input,), method='CROWN-Optimized')
    summaries.append(BoundSummary('CROWN-Optimized', lower.reshape(-1)[0].item(), upper.reshape(-1)[0].item()))

    return summaries


def print_scalar_table(eps: float, empirical: BoundSummary, summaries: Iterable[BoundSummary]) -> None:
    print(f'\neps = {eps:g}')
    print(f'{"method":<28} {"lower":>12} {"upper":>12} {"width":>12} {"excess width":>14}')
    print('-' * 80)
    rows = [empirical, *summaries]
    for row in rows:
        excess_width = row.width - empirical.width
        print(f'{row.method:<28} {row.lower:>12.6f} {row.upper:>12.6f} {row.width:>12.6f} {excess_width:>14.6f}')


def run_1d_control_case(eps_values: Iterable[float], center_value: float, iterations: int, samples: int, device: str) -> None:
    torch.manual_seed(0)
    center = torch.tensor([[center_value]], dtype=torch.float32, device=device)
    model = NestedSigmoidSecondGrad().to(device=device, dtype=center.dtype)

    print('Model: second-derivative through nested sigmoid composition')
    print(f'center = {center_value:g}, alpha iterations = {iterations}, sampled points = {samples}')
    for eps in eps_values:
        empirical = empirical_1d_jacobian_bounds(model, center, eps, samples)
        summaries = compute_scalar_lirpa_bounds(NestedSigmoidSecondGrad, center, eps, iterations=iterations, device=device)
        print_scalar_table(eps, empirical, summaries)


def run_2d_mixed_case(eps_values: Iterable[float], iterations: int, device: str) -> None:
    center = torch.tensor([[0.1, -0.2]], dtype=torch.float32, device=device)
    model = MixedSigmoidSecondGrad().to(device=device, dtype=center.dtype)
    print('\nModel: 2D mixed network, second-derivative of final preactivation')
    print(f'center = {center.flatten().tolist()}, alpha iterations = {iterations}')
    for eps in eps_values:
        empirical = empirical_box_secondgrad_bounds(model, center, eps, grid_size=41)
        summaries = compute_scalar_lirpa_bounds(MixedSigmoidSecondGrad, center, eps, iterations=iterations, device=device)
        print_scalar_table(eps, empirical, summaries)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare alpha-CROWN on Sigmoid second-derivative outputs.')
    parser.add_argument('--eps', type=float, nargs='+', default=[0.1, 0.3, 0.5], help='Perturbation radii to evaluate.')
    parser.add_argument('--center', type=float, default=0.1, help='Center of the one-dimensional input interval.')
    parser.add_argument('--iterations', type=int, default=20, help='Number of alpha-CROWN optimization iterations.')
    parser.add_argument('--samples', type=int, default=2001, help='Grid samples for the empirical value range.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='PyTorch device.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_1d_control_case(eps_values=args.eps, center_value=args.center, iterations=args.iterations, samples=args.samples, device=args.device)
    run_2d_mixed_case(eps_values=args.eps, iterations=args.iterations, device=args.device)
