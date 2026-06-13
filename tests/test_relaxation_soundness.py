"""Direct soundness checks for the nonlinear linear relaxations.

Most of the suite tests bounds end to end on sampled networks, which can hide a
faulty relaxation rule whenever the sampled networks never drive an activation
into the regime where it breaks. These tests instead poke each derivative
operator on its own: we build its linear relaxation on a battery of
pre-activation intervals concentrated around the curvature breakpoints, then
confirm the lower and upper lines really do enclose the true function on a dense
grid inside every interval.

The breakpoints (inflection and extreme points) are where the tangent-line
precompute lookups are most fragile, so an off-by-one or an unfilled-entry lookup
shows up here as a relaxation that crosses the curve, even when it would stay
hidden in an end-to-end test.
"""

from types import SimpleNamespace

import pytest
import torch

from auto_LiRPA.operators.s_shaped import (
    BoundSigmoid, BoundTanh, BoundAtan, BoundSigmoidGrad, BoundTanhGrad,
    BoundSigmoidSecondGrad, BoundTanhSecondGrad, dsigmoid, dtanh, d2sigmoid, d2tanh)

CPU = {'device': torch.device('cpu')}
PIECEWISE = {'sigmoid_second_grad_relaxation': 'piecewise'}
TANGENT = {'sigmoid_second_grad_relaxation': 'tangent'}

# name, operator factory, true function the relaxation must enclose
RELAXATION_CASES = [
    ('sigmoid', lambda: BoundSigmoid(attr=CPU), torch.sigmoid),
    ('tanh', lambda: BoundTanh(attr=CPU), torch.tanh),
    ('atan', lambda: BoundAtan(attr=CPU), torch.atan),
    ('sigmoid_grad', lambda: BoundSigmoidGrad(attr=CPU), dsigmoid),
    ('tanh_grad', lambda: BoundTanhGrad(attr=CPU), dtanh),
    ('sigmoid_second_grad_piecewise',
     lambda: BoundSigmoidSecondGrad(attr=CPU, options=PIECEWISE), d2sigmoid),
    ('sigmoid_second_grad_tangent',
     lambda: BoundSigmoidSecondGrad(attr=CPU, options=TANGENT), d2sigmoid),
    ('tanh_second_grad', lambda: BoundTanhSecondGrad(attr=CPU), d2tanh),
]


def _operator_breakpoints(op):
    """Read the curvature breakpoints off the operator itself, so the test
    follows the constants in s_shaped.py instead of duplicating them."""
    points = list(getattr(op, 'inflections', []))
    points += list(getattr(op, 'extremes', []))
    for attr in ('inflection_point', 'extreme_point', 'outer_inflection_point'):
        if hasattr(op, attr):
            points.append(getattr(op, attr))
    assert points, f'no breakpoints found on {type(op).__name__}'
    return points


def _breakpoint_intervals(breakpoints, max_width=4.0):
    """Pre-activation intervals crowded around each breakpoint.

    Around every breakpoint and its mirror we place endpoints just below and just
    above it, then form every not-too-wide pair. Plenty of the resulting intervals
    straddle a breakpoint with one end sitting just past it, which is exactly the
    case that trips a tangent-line lookup.
    """
    endpoints = {round(v, 4) for v in torch.linspace(-4.0, 4.0, 17).tolist()}
    for point in breakpoints:
        for sign in (1.0, -1.0):
            for offset in (-0.3, -1e-2, -1e-3, 0.0, 1e-3, 1e-2, 0.05, 0.1, 0.3):
                endpoints.add(round(sign * point + offset, 6))
    endpoints = sorted(endpoints)
    lower_ends, upper_ends = [], []
    for i, lower in enumerate(endpoints):
        for upper in endpoints[i + 1:]:
            if upper - lower <= max_width:
                lower_ends.append(lower)
                upper_ends.append(upper)
    return torch.tensor(lower_ends), torch.tensor(upper_ends)


def _relaxation_lines(op, lower, upper):
    # stands in for the input node, which bound_relax only reads .lower/.upper from
    x = SimpleNamespace(lower=lower, upper=upper)
    op.init_linear_relaxation(x)
    op.bound_relax(x, init=False)
    return op.lw, op.lb, op.uw, op.ub


@pytest.mark.parametrize('name, make_op, truth', RELAXATION_CASES,
                         ids=[case[0] for case in RELAXATION_CASES])
def test_relaxation_encloses_function(name, make_op, truth):
    op = make_op()
    breakpoints = _operator_breakpoints(op)
    lower, upper = _breakpoint_intervals(breakpoints)
    lower_slope, lower_bias, upper_slope, upper_bias = (
        _relaxation_lines(op, lower, upper))

    fractions = torch.linspace(0.0, 1.0, 257).unsqueeze(1)
    grid = lower.unsqueeze(0) + fractions * (upper - lower).unsqueeze(0)
    values = truth(grid)
    lower_line = lower_slope.unsqueeze(0) * grid + lower_bias.unsqueeze(0)
    upper_line = upper_slope.unsqueeze(0) * grid + upper_bias.unsqueeze(0)

    crossing_below = (lower_line - values).amax()
    crossing_above = (values - upper_line).amax()
    worst = max(crossing_below.item(), crossing_above.item())
    assert worst <= 1e-5, (
        f'{name} relaxation crosses the function by {worst:.2e}; '
        f'the linear bounds are not sound on some interval near {breakpoints}')
