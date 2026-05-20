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
from collections import deque

import torch

from auto_LiRPA.bound_ops import (
    BoundHessianOP, BoundHessianOutputReshape, HessianOP)
from auto_LiRPA.jacobian import build_jacobian_graph
from auto_LiRPA.operators import BoundInput, BoundJacobianZero, BoundJacobianInit, BoundAdd, BoundHessianInit
from auto_LiRPA.utils import logger, prod


def compute_hessian_bounds(
    self,
    x,
    bound_lower: bool = True,
    bound_upper: bool = True,
    method: str = 'backward',
):
    """Compute IBP or backward bound propagation bounds for a Hessian graph expanded from ``HessianOP``."""

    if isinstance(x, torch.Tensor):
        x = (x,)
    if not getattr(self, 'hessian_node_pairs', None):
        raise RuntimeError('No Hessian nodes found in this BoundedModule')
    return self.compute_bounds(
        method=method, x=x,
        bound_lower=bound_lower, bound_upper=bound_upper)


def _expand_hessian(self):
    self.hessian_node_pairs = []
    for node in list(self.nodes()):
        # print()
        # print(node)
        if isinstance(node, BoundHessianOP):
            self.hessian_node_pairs.append((node.inputs[0], node.inputs[1]))
            expand_hessian_node(self, node)
    if self.hessian_node_pairs:
        self._optimize_graph()
        print("Global input: ", self.global_input)
        self.forward(*self.global_input)

def build_hessian_graph(
        self, output_node, input_node, prefix=None, allow_unused=False):
    batch_size = output_node.output_shape[0]
    output_dim = prod(output_node.output_shape[1:])
    prefix = f'/hessian{output_node.name}' if prefix is None else prefix

    # Gradient values in `grad` may not be accurate. We do not consider gradient
    # accumulation from multiple succeeding nodes. We only want the shapes but
    # not the accurate values.
    grad = {}
    # Dummy values in grad_start
    forward_value = getattr(output_node, 'forward_value', None)
    dtype = forward_value.dtype if isinstance(forward_value, torch.Tensor) else None
    grad_start = torch.ones(batch_size, output_dim,
                            *output_node.output_shape[1:],
                            dtype=dtype, device=self.device)

    hessian_start = torch.ones(batch_size, output_dim,
                            *output_node.output_shape[1:], *output_node.output_shape[1:],
                            dtype=dtype, device=self.device)

    # first position of the tuple is the grad, second is the hessian.
    grad[output_node.name] = (grad_start, hessian_start)
    input_node_found = False

    # First BFS pass: traverse the graph, count degrees, and build gradient
    # layers.
    # Degrees of nodes.
    degree = {}
    # Original layer for gradient computation.
    node_grad_ori = {}

    degree[output_node.name] = 0
    queue = deque([output_node])
    while len(queue) > 0:
        node = queue.popleft()

        if node == input_node:
            input_node_found = True
            continue
        elif node.no_jacobian or not node.from_input or node.no_hessian:
            continue
        else:

            node_grad_ori[node.name] = node.build_hessian_node(*grad[node.name])
            # if 'jacobian2' in prefix:

            print(node)
            print("START")
            for i in range(len(node_grad_ori[node.name])):
                print("\t", node_grad_ori[node.name][i])
            print("END")
            print()

            node_grad_ori[node.name] += [None] * (
                len(node.inputs) - len(node_grad_ori[node.name]))
            # print(node_grad_ori[node.name])
        logger.debug(f'Building hessian node for {node}')
        if not isinstance(node, BoundInput):
            for i in range(len(node.inputs)):
                if node_grad_ori[node.name][i] is None:
                    continue
                entry = node_grad_ori[node.name][i]
                grad_module, grad_args, deps = entry

                def describe_arg(arg):
                    if hasattr(arg, "shape"):
                        return tuple(arg.shape)
                    return repr(arg)
                #T
                # print(
                #     "Node:", node,
                #     "i:", i,
                #     "target:", node.inputs[i].name,
                #     "grad_module:", type(grad_module).__name__,
                #     "arg_shapes:", [describe_arg(arg) for arg in grad_args],
                #     "deps:", [dep.name for dep in deps],
                #     "input_shape:", getattr(grad_module, "input_shape", None),
                # )
                #
                # print("Grad moduls is: ", grad_module)
                # print("Input is ", grad_args)
                grad[node.inputs[i].name] = grad_module(*grad_args)
                if not node.inputs[i].name in degree:
                    degree[node.inputs[i].name] = 0
                    queue.append(node.inputs[i])
                degree[node.inputs[i].name] += 1

    if not input_node_found:
        if not allow_unused:
            raise RuntimeError('Input node not found')
        zero_node = BoundJacobianZero(
            attr=None, inputs=[output_node, input_node],
            output_index=0, options=self.bound_opts)
        zero_node.name = f'{prefix}{input_node.name}/jacobian_zero'
        self.add_nodes([zero_node])
        return zero_node

    # Second BFS pass: build the backward computational graph
    hess_node = {}
    initial_name = f'{prefix}{output_node.name}'
    hess_node[output_node.name] = BoundHessianInit(inputs=[input_node, output_node])
    hess_node[output_node.name].name = initial_name
    self.add_nodes([hess_node[output_node.name]])
    queue = deque([output_node])
    while len(queue) > 0:
        node = queue.popleft()

        if node == input_node:
            return hess_node[node.name]
        if node.no_jacobian or not node.from_input or node.no_hessian:
            continue

        logger.debug(f'Converting gradient node for {node}')
        print(node)
        for k in range(len(node.inputs)):
            if node_grad_ori[node.name][k] is None:
                continue
            # print("AAAAAAAAAAA")
            # print(node_grad_ori[node.name][k][0])
            # print(tuple(item.detach()
            #           for item in node_grad_ori[node.name][k][1]))
            print("Looking at ", node_grad_ori[node.name][k][0])
            print("Passing a tuple: ", tuple(item.detach()
                      for item in node_grad_ori[node.name][k][1])
                  )
            nodes_op, nodes_in, nodes_out, _ = self._convert_nodes(
                node_grad_ori[node.name][k][0],
                tuple(item.detach()
                      for item in node_grad_ori[node.name][k][1])
            )

            # print("Node op: ", nodes_op)
            # print("Node in: ", nodes_in)
            # print("Node out: ", nodes_out)
            # print()
            logger.debug(f'Converting node operators for: {node}')
            logger.debug(f'Generated backwards ops: {nodes_op}')
            rename_dict = {}
            assert isinstance(nodes_in[0], BoundInput)
            rename_dict[nodes_in[0].name] = hess_node[node.name].name
            for i in range(1, len(nodes_in)):
                # Assume it's a parameter here
                new_name = f'{prefix}{node.name}/{k}/params{nodes_in[i].name}'
                rename_dict[nodes_in[i].name] = new_name
            for i in range(len(nodes_op)):
                # intermediate nodes
                if not nodes_op[i].name in rename_dict:
                    new_name = f'{prefix}{node.name}/{k}/tmp{nodes_op[i].name}'
                    rename_dict[nodes_op[i].name] = new_name

            # assert len(nodes_out) == 1

            # print("NODES OUT: ", nodes_out)
            nodes_out = nodes_out[0]
            rename_dict[nodes_out.name] = f'{prefix}{node.name}/{k}/output'

            self.rename_nodes(nodes_op, nodes_in, rename_dict)
            input_nodes_replace = (
                [self._modules[nodes_in[0].name]] + node_grad_ori[node.name][k][2])
            for i in range(len(input_nodes_replace)):
                for n in nodes_op:
                    for j in range(len(n.inputs)):
                        if n.inputs[j].name == nodes_in[i].name:
                            n.inputs[j] = input_nodes_replace[i]
            self.add_nodes(nodes_op + nodes_in[len(input_nodes_replace):])

            if node.inputs[k].name in hess_node:
                node_cur = hess_node[node.inputs[k].name]
                print("ADDING ", node_cur, " and ", nodes_out)
                node_add = BoundAdd(
                    attr=None, inputs=[node_cur, nodes_out],
                    output_index=0, options={})
                node_add.name = f'{nodes_out.name}/add'
                hess_node[node.inputs[k].name] = node_add
                self.add_nodes([node_add])
            else:
                hess_node[node.inputs[k].name] = nodes_out
            degree[node.inputs[k].name] -= 1
            if degree[node.inputs[k].name] == 0:
                queue.append(node.inputs[k])

    raise RuntimeError('Input node not found')

def expand_hessian_node(self, hessian_node):
    logger.info(f'Expanding Jacobian node {hessian_node}')

    output_node = hessian_node.inputs[0]
    input_node = hessian_node.inputs[1]
    replacement_node = build_hessian_graph(self, output_node, input_node)
    self.replace_node(hessian_node, replacement_node)


# def expand_hessian_node(self, hessian_node):
#     output_node = hessian_node.inputs[0]
#     input_node = hessian_node.inputs[1]
#     prefix = f'/hessian{output_node.name}{input_node.name}'
#
#     logger.info(f'Expanding Hessian node {hessian_node.name}')
#     jacobian_node = build_jacobian_graph(
#         self, output_node, input_node,
#         prefix=f'{prefix}/jacobian1', allow_unused=True)
#     # The second Jacobian expansion needs shapes for the first Jacobian graph.
#     self.forward(*self.global_input, final_node_name=jacobian_node.name)
#     logger.debug('Hessian Jacobian expansion checkpoint')
#     jacobian_of_jacobian_node = build_jacobian_graph(
#         self, jacobian_node, input_node,
#         prefix=f'{prefix}/jacobian2', allow_unused=True)
#     hessian_expanded_node = BoundHessianOutputReshape(
#         attr={'order': 2},
#         inputs=[jacobian_of_jacobian_node, output_node, input_node],
#         output_index=0,
#         options=self.bound_opts)
#     hessian_expanded_node.name = f'{prefix}/hessian_output'
#     self.add_nodes([hessian_expanded_node])
#     self.replace_node(hessian_node, hessian_expanded_node)


__all__ = [
    'HessianOP',
    'compute_hessian_bounds',
]
