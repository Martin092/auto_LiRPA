"""Bounds on the Hessian diagonal, computed forwards.

Same recursion as the Hessian trace (see hessian_trace.py), but without the
final sum over input dimensions. Each node carries the Jacobian and the
diagonal diag(d^2 out_k / d input^2), both of shape
(batch, numel, input_dim). Every rule is the trace rule with the sum dropped:

    input x:           D = 0
    linear  Wu + b:    D' = W D
    activation s(u):   D' = s'(u) . D + s''(u) . (J . J)      (elementwise)
    add  u + v:        D' = D_u + D_v
    mul  u . v:        D' = v.D_u + u.D_v + 2 (J_u . J_v)     (elementwise)

The full Hessian is never built, so memory stays at O(numel * input_dim)
instead of O(numel * input_dim^2).

    class DiagWrapper(nn.Module):
        def forward(self, x):
            return DirectHessianDiagOP.apply(self.model(x), x)

    bounded = BoundedModule(DiagWrapper(model), x0)
    lower, upper = bounded.compute_hessian_diag_bounds(x)

Bounds have shape (batch, out_numel, input_dim).
"""
import torch

from auto_LiRPA.bound_ops import (
    BoundDirectHessianDiagOP, BoundHessianDiagInit, DirectHessianDiagOP)
from auto_LiRPA.hessian_trace import build_forward_state_graph


def compute_hessian_diag_bounds(
    self,
    x,
    bound_lower: bool = True,
    bound_upper: bool = True,
    method: str = 'backward',
):
    """Compute bounds for a graph expanded from DirectHessianDiagOP."""
    if isinstance(x, torch.Tensor):
        x = (x,)
    if not getattr(self, 'hessian_diag_node_pairs', None):
        raise RuntimeError('No Hessian diag nodes found in this BoundedModule')
    return self.compute_bounds(
        method=method, x=x,
        bound_lower=bound_lower, bound_upper=bound_upper)


def _expand_hessian_diag(self):
    self.hessian_diag_node_pairs = []
    for node in list(self.nodes()):
        if isinstance(node, BoundDirectHessianDiagOP):
            self.hessian_diag_node_pairs.append((node.inputs[0], node.inputs[1]))
            replacement = build_hessian_diag_graph(
                self, node.inputs[0], node.inputs[1])
            self.replace_node(node, replacement)
    if self.hessian_diag_node_pairs:
        self._optimize_graph()
        self.forward(*self.global_input)


def build_hessian_diag_graph(self, output_node, input_node, prefix=None):
    prefix = f'/hessian_diag{output_node.name}' if prefix is None else prefix
    return build_forward_state_graph(
        self, output_node, input_node, prefix=prefix,
        kind='diag',
        state_init_cls=BoundHessianDiagInit,
        state_dummy_shape=lambda batch, dim: (batch, dim, dim))


__all__ = [
    'DirectHessianDiagOP',
    'compute_hessian_diag_bounds',
]
