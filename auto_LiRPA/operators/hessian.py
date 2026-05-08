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
"""Hessian marker operators."""

import torch

from .base import Bound
from ..utils import prod


class HessianOP(torch.autograd.Function):
    @staticmethod
    def symbolic(g, output, input):
        return g.op('grad::hessian', output, input).setType(output.type())

    @staticmethod
    def forward(ctx, output, input):
        output_ = output.flatten(1)
        input_shape = tuple(input.shape[1:])
        return output.new_zeros(
            output.shape[0], output_.shape[-1],
            *input_shape, *input_shape)


class BoundHessianOP(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, output, input):
        return HessianOP.apply(output, input)


class BoundHessianOutputReshape(Bound):
    def forward(self, jacobian_of_jacobian, output_node_value, input_node_value):
        # Incoming jacobian_of_jacobian shape: (Batch, Out * In, In)
        # Outcoming shape:                     (Batch, Out, In, In)
        batch_size = jacobian_of_jacobian.shape[0]
        output_shape = output_node_value.shape[1:]
        input_shape = input_node_value.shape[1:]

        return jacobian_of_jacobian.reshape(batch_size, *output_shape, *input_shape, *input_shape)

    def interval_propagate(self, *node_inputs):
        lower_hessian_bound, upper_hessian_bound = node_inputs[0]
        output_node_bounds = node_inputs[1]
        input_node_bounds = node_inputs[2]
        
        reshaped_lower_bound = None
        if lower_hessian_bound is not None:
            reshaped_lower_bound = self.forward(lower_hessian_bound, output_node_bounds[0], input_node_bounds[0])
            
        reshaped_upper_bound = None
        if upper_hessian_bound is not None:
            reshaped_upper_bound = self.forward(upper_hessian_bound, output_node_bounds[0], input_node_bounds[0])
        
        return reshaped_lower_bound, reshaped_upper_bound

    def bound_backward(self, lower_A_matrix, upper_A_matrix, *node_inputs, **kwargs):
        # Incoming A matrix shape:  (Specification, Batch, Out, In, In)
        # Outcoming A matrix shape: (Specification, Batch, Out * In, In)
        input_node = node_inputs[2]
        input_shape = input_node.forward_value.shape[1:]

        def unreshape_coefficient_matrix(coefficient_matrix):
            if coefficient_matrix is None: 
                return None
            
            # The -1 squishes the Output and first Input dimensions together
            return coefficient_matrix.reshape(coefficient_matrix.shape[0], coefficient_matrix.shape[1], -1, *input_shape)

        return [
            (unreshape_coefficient_matrix(lower_A_matrix), unreshape_coefficient_matrix(upper_A_matrix)), 
            (None, None), 
            (None, None)
        ], 0.0, 0.0
