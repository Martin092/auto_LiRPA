"""Bounds on the Hessian trace, computed forwards.

The trace cannot be computed backwards. Pulling tr(H) through a linear layer
needs diag(W^T H W), which reads every entry of H. Forwards it works, because
each node only needs two things from its inputs: the Jacobian of its flattened
output with respect to the model input, shape (batch, numel, input_dim), and
the trace per output, shape (batch, numel). Each operator gives its own rule:

    input x:           J = I                 t = 0
    linear  Wu + b:    J' = W J              t' = W t
    activation s(u):   J' = s'(u) . J        t' = s'(u) . t + s''(u) . rowsq(J)
    add  u + v:        J' = J_u + J_v        t' = t_u + t_v
    mul  u . v:        J' = v.J_u + u.J_v    t' = v.t_u + u.t_v + 2 (J_u . J_v) 1

where rowsq(J)_k is the sum of squares of row k. Going forwards also means a
node used by several consumers needs no accumulation, since they share its
state.

The Hessian diagonal uses the same recursion, see hessian_diag.py.

    class TraceWrapper(nn.Module):
        def forward(self, x):
            return DirectHessianTraceOP.apply(self.model(x), x)

    bounded = BoundedModule(TraceWrapper(model), x0)
    lower, upper = bounded.compute_hessian_trace_bounds(x)
"""
from collections import deque

import torch

from auto_LiRPA.bound_ops import (
    BoundDirectHessianTraceOP, BoundHessianTraceInit, DirectHessianTraceOP)
from auto_LiRPA.operators import (
    BoundFlatten, BoundInput, BoundJacobianInit, BoundReshape)
from auto_LiRPA.utils import logger, prod


def compute_hessian_trace_bounds(
    self,
    x,
    bound_lower: bool = True,
    bound_upper: bool = True,
    method: str = 'backward',
):
    """Compute bounds for a graph expanded from DirectHessianTraceOP."""
    if isinstance(x, torch.Tensor):
        x = (x,)
    if not getattr(self, 'hessian_trace_node_pairs', None):
        raise RuntimeError('No Hessian trace nodes found in this BoundedModule')
    return self.compute_bounds(
        method=method, x=x,
        bound_lower=bound_lower, bound_upper=bound_upper)


def _expand_hessian_trace(self):
    self.hessian_trace_node_pairs = []
    for node in list(self.nodes()):
        if isinstance(node, BoundDirectHessianTraceOP):
            self.hessian_trace_node_pairs.append((node.inputs[0], node.inputs[1]))
            replacement = build_hessian_trace_graph(
                self, node.inputs[0], node.inputs[1])
            self.replace_node(node, replacement)
    if self.hessian_trace_node_pairs:
        self._optimize_graph()
        self.forward(*self.global_input)


def build_hessian_trace_graph(self, output_node, input_node, prefix=None):
    prefix = f'/hessian_trace{output_node.name}' if prefix is None else prefix
    return build_forward_state_graph(
        self, output_node, input_node, prefix=prefix,
        kind='trace',
        state_init_cls=BoundHessianTraceInit,
        state_dummy_shape=lambda batch, dim: (batch, dim))


def build_forward_state_graph(self, output_node, input_node, prefix, kind,
                              state_init_cls, state_dummy_shape):
    """Build the forward propagation graph from input_node to output_node.

    Each node carries a Jacobian and a state, where the state is the Hessian
    trace or diagonal depending on kind. Every operator gives its own rule
    through build_hessian_state_node. Returns the state node of the output.
    """

    # Everything the output depends on; state spreads from the input node
    # through this cone only.
    cone = set()
    queue = deque([output_node])
    while queue:
        node = queue.popleft()
        if node.name in cone:
            continue
        cone.add(node.name)
        queue.extend(node.inputs)
    if input_node.name not in cone:
        raise RuntimeError('Input node not found')

    # Topological order over the cone, inputs before consumers. Duplicate
    # edges (a node consuming the same input twice) are counted per edge.
    indegree = {name: 0 for name in cone}
    children = {name: [] for name in cone}
    for name in cone:
        for inp in self._modules[name].inputs:
            if inp.name in cone:
                indegree[name] += 1
                children[inp.name].append(name)
    order = deque(name for name in cone if indegree[name] == 0)
    topological = []
    while order:
        name = order.popleft()
        topological.append(name)
        for child in children[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                order.append(child)

    # Initial state at the input node: identity Jacobian, zero trace/diagonal.
    jacobian_init = BoundJacobianInit(inputs=[input_node])
    jacobian_init.name = f'{prefix}{input_node.name}/jacobian'
    state_init = state_init_cls(inputs=[input_node])
    state_init.name = f'{prefix}{input_node.name}/{kind}'
    self.add_nodes([jacobian_init, state_init])

    batch_size = input_node.output_shape[0]
    input_dim = prod(input_node.output_shape[1:])
    forward_value = getattr(input_node, 'forward_value', None)
    dtype = forward_value.dtype if isinstance(forward_value, torch.Tensor) else None
    jacobian_dummy = torch.ones(
        batch_size, input_dim, input_dim, dtype=dtype, device=self.device)
    state_dummy = torch.ones(
        *state_dummy_shape(batch_size, input_dim),
        dtype=dtype, device=self.device)

    # name -> (jacobian node, state node, jacobian dummy, state dummy)
    state = {input_node.name: (jacobian_init, state_init,
                               jacobian_dummy, state_dummy)}

    for name in topological:
        node = self._modules[name]
        if name in state or not any(
                inp.name in state for inp in node.inputs):
            continue

        # The state lives on the flattened output, and a pure reshape leaves
        # the flattened order untouched, so the producer's state passes
        # through as is. The ONNX trace likes to slip a Flatten between a
        # model output and its consumers, which is how these show up here.
        if isinstance(node, (BoundFlatten, BoundReshape)):
            source = next(inp.name for inp in node.inputs if inp.name in state)
            state[name] = state[source]
            continue

        logger.debug(f'Building {kind} propagation node for {node}')

        input_states = [
            state[inp.name][2:] if inp.name in state else None
            for inp in node.inputs]
        module, args, deps = node.build_hessian_state_node(input_states, kind)

        with torch.no_grad():
            jacobian_dummy, state_dummy = module(*args)

        nodes_op, nodes_in, nodes_out, _ = self._convert_nodes(
            module, tuple(arg.detach() for arg in args))
        if len(nodes_out) != 2:
            raise RuntimeError(
                f'{kind} propagation node for {node} must return '
                f'(jacobian, {kind}), got {len(nodes_out)} outputs.')

        # The first arguments are the states of the state-carrying inputs, in
        # input order; then the deps; anything left over is a parameter or
        # constant captured from the module itself.
        state_nodes = []
        for inp in node.inputs:
            if inp.name in state:
                state_nodes.extend(state[inp.name][:2])
        replacements = state_nodes + deps
        if len(nodes_in) < len(replacements):
            raise RuntimeError(
                f'{kind} propagation node for {node} consumed '
                f'{len(nodes_in)} inputs but {len(replacements)} were wired.')

        rename_dict = {}
        for i, replacement in enumerate(replacements):
            assert isinstance(nodes_in[i], BoundInput)
            rename_dict[nodes_in[i].name] = replacement.name
        for i in range(len(replacements), len(nodes_in)):
            rename_dict[nodes_in[i].name] = (
                f'{prefix}{node.name}/params{nodes_in[i].name}')
        for op in nodes_op:
            if op.name not in rename_dict:
                rename_dict[op.name] = f'{prefix}{node.name}/tmp{op.name}'
        jacobian_out, state_out = nodes_out
        rename_dict[jacobian_out.name] = f'{prefix}{node.name}/jacobian'
        rename_dict[state_out.name] = f'{prefix}{node.name}/{kind}'

        self.rename_nodes(nodes_op, nodes_in, rename_dict)
        for i, replacement in enumerate(replacements):
            for op in nodes_op:
                for j in range(len(op.inputs)):
                    if op.inputs[j].name == nodes_in[i].name:
                        op.inputs[j] = replacement
        self.add_nodes(nodes_op + nodes_in[len(replacements):])

        state[name] = (jacobian_out, state_out, jacobian_dummy, state_dummy)

    if output_node.name not in state:
        raise RuntimeError(
            f'Output node {output_node.name} does not depend on the input; '
            f'the {kind} propagation graph could not be built.')
    return state[output_node.name][1]


__all__ = [
    'DirectHessianTraceOP',
    'compute_hessian_trace_bounds',
]
