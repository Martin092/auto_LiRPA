import numpy as np
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.hessian import DirectHessianOP
from auto_LiRPA.perturbations import PerturbationLpNorm
import matplotlib.pyplot as plt


class HessianWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return DirectHessianOP.apply(self.model(x), x)


def build_model(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Softplus(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Softplus(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Softplus(),
        nn.Linear(hidden_dim, output_dim),
    ).double()


def run_experiment(runs: int, seed: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_dim = 1
    hidden_dims = np.arange(1, 1501, 100)

    diffs = []
    for run_idx in range(runs):
        if seed is not None:
            torch.manual_seed(seed + run_idx)
            np.random.seed(seed + run_idx)

        lower_bounds = []
        upper_bounds = []

        for h in hidden_dims:
            model = build_model(input_dim, h, 1)
            x0 = torch.randn(1, input_dim, dtype=torch.double)

            bounded = BoundedModule(HessianWrapper(model), x0)
            x = BoundedTensor(x0, PerturbationLpNorm(norm=float("inf"), eps=1))
            lower, upper = bounded.compute_hessian_bounds(x, method="backward")

            lower_bounds.append(lower.detach().flatten().numpy()[0])
            upper_bounds.append(upper.detach().flatten().numpy()[0])

        diffs.append(np.array(upper_bounds) - np.array(lower_bounds))

    diffs_array = np.stack(diffs, axis=0)
    mean_diffs = diffs_array.mean(axis=0)
    std_diffs = diffs_array.std(axis=0)
    return hidden_dims, mean_diffs, std_diffs





def main() -> None:
    runs = 10
    seed = 88083
    hidden_dims, mean_diffs, std_diffs = run_experiment(runs, seed)

    plt.plot(hidden_dims, mean_diffs, label="Mean bound width")
    plt.fill_between(
        hidden_dims,
        mean_diffs - std_diffs,
        mean_diffs + std_diffs,
        alpha=0.2,
        label="±1 std",
    )
    plt.title("Difference between bounds (mean ± std), epsilon=1")
    plt.xlabel("Hidden dimensions")
    plt.ylabel("Bound width")
    plt.legend()

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
