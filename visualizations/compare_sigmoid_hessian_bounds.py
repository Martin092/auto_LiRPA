"""Compare sigmoid Hessian bounds across sigmoid'' relaxation choices.

This is a small, deterministic showcase for the three practically interesting
settings of :class:`BoundSigmoidSecondGrad`:

1. ``tangent`` with ordinary CROWN/backward bounds:
   the historical fixed tangent-to-secant relaxation;
2. ``piecewise`` with ordinary CROWN/backward bounds:
   the new case-table relaxation with fixed midpoint tangent choices;
3. ``piecewise`` with ``alpha-CROWN``:
   the same case-table relaxation with optimized tangent points.

The network is intentionally one-dimensional so its output Hessian is a scalar,
but it has several sigmoid hidden units whose pre-activation intervals cross
the relevant curvature regions of sigmoid''.  That keeps the comparison easy
to read while still exercising the real Hessian-bound path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.hessian import HessianOP
from auto_LiRPA.perturbations import PerturbationLpNorm


DEFAULT_OUTPUT = Path(__file__).with_name("sigmoid_hessian_bound_comparison.png")


class HessianWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HessianOP.apply(self.model(x), x)


def build_demo_model() -> nn.Module:
    """Build a fixed 1D sigmoid network with diverse pre-activation regions."""
    model = nn.Sequential(
        nn.Linear(1, 4),
        nn.Sigmoid(),
        nn.Linear(4, 1),
    ).double()

    with torch.no_grad():
        # With x in [-1, 1], these hidden pre-activation intervals are:
        # [-3.6, -1.6], [-1.5, 0.5], [-0.5, 1.5], [1.6, 3.6].
        # Together they cross the left outer inflection, zero, and the right
        # outer inflection of sigmoid''.
        model[0].weight[:] = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
        model[0].bias[:] = torch.tensor([-2.6, -0.5, 0.5, 2.6])
        # Positive output weights make the aggregate Hessian width easy to
        # interpret: tighter neuronwise second-derivative bounds translate
        # directly into a tighter scalar Hessian interval.
        model[2].weight[:] = torch.tensor([[1.0, 0.8, 0.8, 1.0]])
        model[2].bias.zero_()

    return model


@dataclass(frozen=True)
class ExperimentSetting:
    label: str
    relaxation: str
    method: str
    optimize_iterations: int | None = None


@dataclass(frozen=True)
class BoundResult:
    label: str
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


def compute_bounds(
        model: nn.Module,
        x0: torch.Tensor,
        eps: float,
        setting: ExperimentSetting) -> BoundResult:
    """Compute one scalar Hessian interval for the requested setting."""
    bound_opts = {
        "sigmoid_second_grad_relaxation": setting.relaxation,
    }
    if setting.optimize_iterations is not None:
        bound_opts["optimize_bound_args"] = {
            "iteration": setting.optimize_iterations,
        }

    bounded = BoundedModule(HessianWrapper(model), x0, bound_opts=bound_opts)
    x = BoundedTensor(
        x0, PerturbationLpNorm(norm=float("inf"), eps=eps))
    lower, upper = bounded.compute_hessian_bounds(x, method=setting.method)
    return BoundResult(
        label=setting.label,
        lower=lower.detach().reshape(-1)[0].item(),
        upper=upper.detach().reshape(-1)[0].item(),
    )


def sample_hessian_range(
        model: nn.Module,
        x0: torch.Tensor,
        eps: float,
        num_samples: int = 1001) -> tuple[float, float]:
    """Densely sample the true scalar Hessian over the 1D input interval."""
    samples = torch.linspace(
        x0.item() - eps, x0.item() + eps,
        steps=num_samples, dtype=x0.dtype).view(-1, 1)

    values = []
    for sample in samples:
        sample = sample.view_as(x0).detach().clone().requires_grad_(True)
        output = model(sample).reshape(())
        grad = torch.autograd.grad(output, sample, create_graph=True)[0]
        hessian = torch.autograd.grad(grad.reshape(()), sample)[0]
        values.append(hessian.detach().reshape(()))

    stacked = torch.stack(values)
    return stacked.min().item(), stacked.max().item()


def print_summary(
        results: list[BoundResult],
        sampled_lower: float,
        sampled_upper: float) -> None:
    print("Sigmoid Hessian bound comparison")
    print("=" * 72)
    print(f"Sampled true Hessian range: [{sampled_lower:+.8f}, {sampled_upper:+.8f}]")
    print()
    print(f"{'setting':<28} {'lower':>13} {'upper':>13} {'width':>13}")
    print("-" * 72)
    for result in results:
        print(
            f"{result.label:<28} "
            f"{result.lower:+13.8f} "
            f"{result.upper:+13.8f} "
            f"{result.width:13.8f}")


def save_plot(
        results: list[BoundResult],
        sampled_lower: float,
        sampled_upper: float,
        output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y_positions = list(range(len(results)))

    for y, result in zip(y_positions, results):
        ax.plot(
            [result.lower, result.upper], [y, y],
            linewidth=6, solid_capstyle="round", label=result.label)
        ax.plot([result.lower, result.upper], [y, y], "ko", markersize=5)

    ax.axvspan(
        sampled_lower, sampled_upper,
        color="black", alpha=0.12, label="sampled true range")
    ax.axvline(sampled_lower, color="black", linewidth=1, linestyle="--")
    ax.axvline(sampled_upper, color="black", linewidth=1, linestyle="--")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([result.label for result in results])
    ax.set_xlabel("scalar Hessian bound")
    ax.set_title("Sigmoid network Hessian bounds by sigmoid'' relaxation")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sigmoid Hessian bounds across relaxation settings.")
    parser.add_argument(
        "--eps", type=float, default=1.0,
        help="L-infinity radius around x0=0 for the 1D input interval.")
    parser.add_argument(
        "--samples", type=int, default=1001,
        help="Number of dense samples used to estimate the true Hessian range.")
    parser.add_argument(
        "--alpha-iterations", type=int, default=20,
        help="Optimization iterations for the alpha-CROWN piecewise run.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Where to save the comparison plot.")
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Only print the numeric comparison table.")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = build_demo_model()
    x0 = torch.zeros(1, 1, dtype=torch.double)

    settings = [
        ExperimentSetting(
            label="tangent + backward",
            relaxation="tangent",
            method="backward"),
        ExperimentSetting(
            label="piecewise + backward",
            relaxation="piecewise",
            method="backward"),
        ExperimentSetting(
            label="piecewise + alpha-CROWN",
            relaxation="piecewise",
            method="alpha-CROWN",
            optimize_iterations=args.alpha_iterations),
    ]

    results = [
        compute_bounds(model, x0, args.eps, setting)
        for setting in settings
    ]
    sampled_lower, sampled_upper = sample_hessian_range(
        model, x0, args.eps, args.samples)

    print_summary(results, sampled_lower, sampled_upper)
    if not args.no_plot:
        save_plot(results, sampled_lower, sampled_upper, args.output)


if __name__ == "__main__":
    main()
