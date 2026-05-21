#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Handle Jacobian bounds."""

import torch
from auto_LiRPA.bound_ops import JacobianOP, GradNorm  # pylint: disable=unused-import
from auto_LiRPA.bound_ops import (
    BoundInput, BoundAdd, BoundRelu, BoundJacobianInit, BoundJacobianZero,
    BoundJacobianOP)
from auto_LiRPA.utils import logger, prod
from collections import deque

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def _expand_jacobian(self):
    self.jacobian_start_nodes = []
    for node in list(self.nodes()):
        if isinstance(node, BoundJacobianOP):
            self.jacobian_start_nodes.append(node.inputs[0])
            expand_jacobian_node(self, node)
    if self.jacobian_start_nodes:
        # Disable unstable options
        self.bound_opts.update({
            'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,
            'sparse_intermediate_bounds_with_ibp': False,
            'sparse_features_alpha': False,
            'sparse_spec_alpha': False,
        })
        # Optimize new nodes if possible
        self._optimize_graph()
        for node in self.nodes():
            if isinstance(node, BoundRelu):
                node.use_sparse_spec_alpha = node.use_sparse_features_alpha = False
        # If Jacobian nodes are added, we need to redo the forward pass to update the
        # properties of newly added nodes (e.g., output shape, forward value, etc.)
        self.forward(*self.global_input)


def expand_jacobian_node(self, jacobian_node):
    logger.info(f'Expanding Jacobian node {jacobian_node}')

    output_node = jacobian_node.inputs[0]
    input_node = jacobian_node.inputs[1]
    replacement_node = build_jacobian_graph(self, output_node, input_node)
    self.replace_node(jacobian_node, replacement_node)


def build_jacobian_graph(
        self, output_node, input_node, prefix=None, allow_unused=False):
    batch_size = output_node.output_shape[0]
    output_dim = prod(output_node.output_shape[1:])
    prefix = f'/jacobian{output_node.name}' if prefix is None else prefix

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
    grad[output_node.name] = grad_start
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
        elif node.no_jacobian or not node.from_input:
            continue
        else:
            node_grad_ori[node.name] = node.build_gradient_node(grad[node.name])
            # if 'jacobian2' in prefix:

            #print(node)
            #print("START")
            #for i in range(len(node_grad_ori[node.name])):
            #    print("\t", node_grad_ori[node.name][i])
            #print("END")
            #print()

            node_grad_ori[node.name] += [None] * (
                len(node.inputs) - len(node_grad_ori[node.name]))
            # print(node_grad_ori[node.name])
        logger.debug(f'Building gradient node for {node}')
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
                #
                # print(
                #     "Node:", node,
                #     "i:", i,
                #     "target:", node.inputs[i].name,
                #     "grad_module:", type(grad_module).__name__,
                #     "arg_shapes:", [describe_arg(arg) for arg in grad_args],
                #     "deps:", [dep.name for dep in deps],
                #     "input_shape:", getattr(grad_module, "input_shape", None),
                # )

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
    grad_node = {}
    initial_name = f'{prefix}{output_node.name}'
    grad_node[output_node.name] = BoundJacobianInit(inputs=[output_node])
    grad_node[output_node.name].name = initial_name
    self.add_nodes([grad_node[output_node.name]])
    queue = deque([output_node])
    while len(queue) > 0:
        node = queue.popleft()

        if node == input_node:
            return grad_node[node.name]
        if node.no_jacobian or not node.from_input:
            continue

        logger.debug(f'Converting gradient node for {node}')
        #print(node)
        for k in range(len(node.inputs)):
            if node_grad_ori[node.name][k] is None:
                continue
            nodes_op, nodes_in, nodes_out, _ = self._convert_nodes(
                node_grad_ori[node.name][k][0],
                tuple(item.detach()
                      for item in node_grad_ori[node.name][k][1]))

            #print("Node op: ", nodes_op)
            #print("Node in: ", nodes_in)
            #print("Node out: ", nodes_out)
            #print()
            logger.debug(f'Converting node operators for: {node}')
            logger.debug(f'Generated backwards ops: {nodes_op}')
            rename_dict = {}
            assert isinstance(nodes_in[0], BoundInput)
            rename_dict[nodes_in[0].name] = grad_node[node.name].name
            for i in range(1, len(nodes_in)):
                # Assume it's a parameter here
                new_name = f'{prefix}{node.name}/{k}/params{nodes_in[i].name}'
                rename_dict[nodes_in[i].name] = new_name
            for i in range(len(nodes_op)):
                # intermediate nodes
                if not nodes_op[i].name in rename_dict:
                    new_name = f'{prefix}{node.name}/{k}/tmp{nodes_op[i].name}'
                    rename_dict[nodes_op[i].name] = new_name
            assert len(nodes_out) == 1
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

            if node.inputs[k].name in grad_node:
                node_cur = grad_node[node.inputs[k].name]
                print("ADDING ", node_cur, " and ", nodes_out)
                node_add = BoundAdd(
                    attr=None, inputs=[node_cur, nodes_out],
                    output_index=0, options={})
                node_add.name = f'{nodes_out.name}/add'
                grad_node[node.inputs[k].name] = node_add
                self.add_nodes([node_add])
            else:
                grad_node[node.inputs[k].name] = nodes_out
            degree[node.inputs[k].name] -= 1
            if degree[node.inputs[k].name] == 0:
                queue.append(node.inputs[k])

    raise RuntimeError('Input node not found')


def compute_jacobian_bounds(self: 'BoundedModule', x, optimize=True,
                            optimize_output_node=None,
                            bound_lower=True, bound_upper=True):
    """Compute jacobian bounds on the pre-augmented graph (new API)."""

    if isinstance(x, torch.Tensor):
        x = (x,)

    if optimize:
        if optimize_output_node is None:
            if len(self.jacobian_start_nodes) == 1:
                optimize_output_node = self.jacobian_start_nodes[0]
            else:
                raise NotImplementedError(
                    'Multiple Jacobian nodes found.'
                    'An output node for optimizable bounds (optimize_output_node) '
                    'must be specified explicitly')
        self.compute_bounds(
            method='CROWN-Optimized',
            C=None, x=x, bound_upper=False,
            final_node_name=optimize_output_node.name)
        intermediate_bounds = {}
        for node in self._modules.values():
            if node.is_lower_bound_current():
                intermediate_bounds[node.name] = (node.lower, node.upper)
    else:
        intermediate_bounds = None
    lb, ub = self.compute_bounds(
        method='CROWN', x=x,
        bound_lower=bound_lower, bound_upper=bound_upper,
        interm_bounds=intermediate_bounds)
    return lb, ub
