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
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        attr = {} if attr is None else attr
        super().__init__(attr, inputs, output_index, options)
        self.order = attr.get('order', 2)

    def forward(self, derivative, output, input):
        return _reshape_derivative_output(
            derivative, output.shape[1:], input.shape[1:], self.order)

    def interval_propagate(self, *v):
        return (
            _reshape_derivative_output(
                v[0][0], v[1][0].shape[1:], v[2][0].shape[1:], self.order),
            _reshape_derivative_output(
                v[0][1], v[1][0].shape[1:], v[2][0].shape[1:], self.order))


def _reshape_derivative_output(derivative, output_shape, input_shape, order):
    output_dim = prod(output_shape)
    if order == 1:
        return derivative.reshape(
            derivative.shape[0], output_dim, *input_shape)
    if order == 2:
        return derivative.reshape(
            derivative.shape[0], output_dim, *input_shape, *input_shape)
    raise NotImplementedError(order)
