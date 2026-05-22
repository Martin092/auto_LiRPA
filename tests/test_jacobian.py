# pylint: disable=wrong-import-position
"""Test Jacobian bounds."""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1] / 'examples' / 'vision'))
from jacobian import compute_jacobians
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
from auto_LiRPA.jacobian import JacobianOP
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class TestJacobian(TestCase):
    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName, seed=1, ref_name='jacobian_test_data',
            generate=generate,
            device=device, dtype=dtype)

    def test(self):
        in_dim, linear_size = 8, 100
        model = nn.Sequential(
            Flatten(),
            nn.Linear(3*in_dim**2, linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, linear_size),
            nn.Tanh(),
            nn.Linear(linear_size, linear_size),
            nn.Sigmoid(),
            nn.Linear(linear_size, 10),
        )
        model = model.to(device=self.default_device, dtype=self.default_dtype)
        x0 = torch.randn(1, 3, in_dim, in_dim,
                         device=self.default_device, dtype=self.default_dtype)
        self.result = compute_jacobians(model, x0)
        self.check()

    def test_concat_jacobian(self):
        '''
        Test JacobianOP with Concat operation. This needs some special handling
        in auto_LiRPA to make it work properly. (See parse_graph.py for details.)
        '''
        class ConcatModule(nn.Module):
            def forward(self, x):
                return JacobianOP.apply(torch.cat([x, x], dim=1), x)
        concatmodel = ConcatModule().to(device=self.default_device, dtype=self.default_dtype)
        x0 = torch.randn(1, 5, device=self.default_device, dtype=self.default_dtype)
        BoundedModule(concatmodel, x0)
        print('Concat JacobianOP test passed.')

    def test_direct_optimized_jacobian_bounds(self):
        """Direct Jacobian-graph optimization should be sound and runnable."""

        class NestedSigmoidJacobian(nn.Module):
            def forward(self, x):
                y = torch.sigmoid(torch.sigmoid(x))
                return JacobianOP.apply(y, x)

        model = NestedSigmoidJacobian().to(
            device=self.default_device, dtype=self.default_dtype)
        x0 = torch.tensor([[0.1]], device=self.default_device,
                          dtype=self.default_dtype)
        bounded = BoundedModule(
            model, x0,
            bound_opts={'optimize_bound_args': {'iteration': 2}})
        eps = 0.2
        x = BoundedTensor(
            x0, PerturbationLpNorm(norm=float('inf'), eps=eps))

        lower, upper = bounded.compute_jacobian_bounds(
            x, optimize_target='jacobian')

        for point in torch.linspace(
                x0.item() - eps, x0.item() + eps, steps=9,
                device=self.default_device, dtype=self.default_dtype):
            sample = point.reshape_as(x0).detach().requires_grad_(True)
            output = torch.sigmoid(torch.sigmoid(sample))
            jacobian = torch.autograd.grad(output.sum(), sample)[0]
            jacobian = jacobian.reshape_as(lower)
            assert torch.all(jacobian >= lower - 1e-5)
            assert torch.all(jacobian <= upper + 1e-5)

        # The historical primal-optimization path should remain usable after
        # directly optimizing the expanded Jacobian graph.
        bounded.compute_jacobian_bounds(x, optimize_target='primal')


if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestJacobian(generate=False)
    testcase.setUp()
    testcase.test()
