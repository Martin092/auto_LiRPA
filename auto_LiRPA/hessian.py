"""Hessian bounds for native ``BoundedModule`` graphs.

``HessianOP`` marks a Hessian request in the parsed graph. During
``BoundedModule`` construction, the marker is expanded as a Jacobian of a
Jacobian, reusing the existing Jacobian graph construction machinery.

Native ``BoundedModule`` usage:

    class HessianWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return HessianOP.apply(self.model(x), x)

    bounded_model = BoundedModule(HessianWrapper(model), x0)
    lower, upper = bounded_model.compute_hessian_bounds(x)
"""

import torch

from auto_LiRPA.bound_ops import (
    BoundHessianOP, BoundHessianOutputReshape, HessianOP)
from auto_LiRPA.jacobian import build_jacobian_graph


def compute_hessian_bounds(
    self,
    x,
    bound_lower: bool = True,
    bound_upper: bool = True,
):
    """Compute IBP bounds for a Hessian graph expanded from ``HessianOP``."""

    if isinstance(x, torch.Tensor):
        x = (x,)
    if not getattr(self, 'hessian_node_pairs', None):
        raise RuntimeError('No Hessian nodes found in this BoundedModule')
    return self.compute_bounds(
        method='IBP', x=x,
        bound_lower=bound_lower, bound_upper=bound_upper)


def _expand_hessian(self):
    self.hessian_node_pairs = []
    for node in list(self.nodes()):
        if isinstance(node, BoundHessianOP):
            self.hessian_node_pairs.append((node.inputs[0], node.inputs[1]))
            expand_hessian_node(self, node)
    if self.hessian_node_pairs:
        self._optimize_graph()
        self.forward(*self.global_input)


def expand_hessian_node(self, hessian_node):
    output_node = hessian_node.inputs[0]
    input_node = hessian_node.inputs[1]
    prefix = f'/hessian{output_node.name}{input_node.name}'

    jacobian_node = build_jacobian_graph(
        self, output_node, input_node,
        prefix=f'{prefix}/jacobian1', allow_unused=True)
    # The second Jacobian expansion needs shapes for the first Jacobian graph.
    self.forward(*self.global_input, final_node_name=jacobian_node.name)
    print("**********")
    print("**********")
    jacobian_of_jacobian_node = build_jacobian_graph(
        self, jacobian_node, input_node,
        prefix=f'{prefix}/jacobian2', allow_unused=True)
    hessian_expanded_node = BoundHessianOutputReshape(
        attr={'order': 2},
        inputs=[jacobian_of_jacobian_node, output_node, input_node],
        output_index=0,
        options=self.bound_opts)
    hessian_expanded_node.name = f'{prefix}/hessian_output'
    self.add_nodes([hessian_expanded_node])
    self.replace_node(hessian_node, hessian_expanded_node)


__all__ = [
    'HessianOP',
    'compute_hessian_bounds',
]
