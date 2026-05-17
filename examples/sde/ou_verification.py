"""
Lyapunov verification benchmark for a 2D Ornstein-Uhlenbeck SDE.

Dynamics:
    dX_t = -X_t dt + sigma dW_t,  sigma = 0.1 by default.

Domain:
    x in [-1, 1]^2.

Candidate:
    A Softplus MLP is trained to fit the quadratic Lyapunov function
    V(x) = x^T P x with P = 0.5 I. This P corresponds to the stable drift
    matrix -I, since (-I)^T P + P (-I) = -I.

Certification target:
    L V(x) + alpha V(x) <= 0.

For the exact quadratic V(x) = 0.5 ||x||^2,
    L V(x) + alpha V(x) = (-1 + 0.5 alpha) ||x||^2 + sigma^2.
Because sigma > 0, the full-domain condition cannot hold at x = 0. This
benchmark reports that analytical gap as a sanity check and certifies the
neural candidate on box partitions of the domain.
"""

import argparse
import copy
import itertools

import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import JacobianOP
from auto_LiRPA.hessian import HessianOP
from auto_LiRPA.perturbations import PerturbationLpNorm


class SoftplusLyapunovMLP(nn.Module):
    def __init__(self, hidden_width=16, beta=5.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_width),
            nn.Softplus(beta=beta),
            nn.Linear(hidden_width, hidden_width),
            nn.Softplus(beta=beta),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, x):
        return self.net(x)


class HessianWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state):
        return HessianOP.apply(self.model(state), state)


class HessianTraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state):
        hessian = HessianOP.apply(self.model(state), state)
        return hessian[:, :, 0, 0] + hessian[:, :, 1, 1]


class JacobianWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state):
        return JacobianOP.apply(self.model(state), state)


class OUGeneratorGapWrapper(nn.Module):
    def __init__(self, model, alpha, sigma):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, state):
        value = self.model(state)
        jacobian = JacobianOP.apply(value, state)
        hessian = HessianOP.apply(value, state)

        drift = -(
            jacobian[:, :, 0] * state[:, 0:1]
            + jacobian[:, :, 1] * state[:, 1:2])
        diffusion = 0.5 * self.sigma ** 2 * (
            hessian[:, :, 0, 0] + hessian[:, :, 1, 1])
        return drift + diffusion + self.alpha * value


def quadratic_target(x):
    return 0.5 * x.square().sum(dim=1, keepdim=True)


def analytic_quadratic_gap(x, alpha, sigma):
    radius_squared = x.square().sum(dim=1, keepdim=True)
    return (-1.0 + 0.5 * alpha) * radius_squared + sigma ** 2


def sample_domain(num_samples, device, exclude_radius=0.0):
    if exclude_radius <= 0:
        return 2.0 * torch.rand(num_samples, 2, device=device) - 1.0
    if exclude_radius >= 2.0 ** 0.5:
        raise ValueError(
            'exclude_radius must be smaller than sqrt(2) for [-1, 1]^2')

    samples = []
    while sum(sample.shape[0] for sample in samples) < num_samples:
        candidate = 2.0 * torch.rand(
            max(num_samples, 1024), 2, device=device) - 1.0
        mask = candidate.square().sum(dim=1) >= exclude_radius ** 2
        samples.append(candidate[mask])
    return torch.cat(samples, dim=0)[:num_samples]


def train_certificate(
        model, epochs, num_samples, batch_size, learning_rate, device):
    model.train()
    samples = sample_domain(num_samples, device)
    targets = quadratic_target(samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        indices = torch.randint(
            num_samples, (batch_size,), device=device)
        batch = samples[indices]
        target = targets[indices]
        prediction = model(batch)
        loss = torch.mean((prediction - target) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        validation = sample_domain(2048, device)
        error = (model(validation) - quadratic_target(validation)).abs()
    return error.max().item(), error.mean().item()


def exact_network_gap(model, x, alpha, sigma):
    x = x.detach().clone().requires_grad_(True)
    value = model(x)
    gradient = torch.autograd.grad(
        value.sum(), x, create_graph=True)[0]

    hessian_diag = []
    for dim in range(x.shape[1]):
        second = torch.autograd.grad(
            gradient[:, dim].sum(), x,
            retain_graph=True, create_graph=False)[0][:, dim]
        hessian_diag.append(second)
    hessian_trace = torch.stack(hessian_diag, dim=1).sum(
        dim=1, keepdim=True)

    drift = -(gradient * x).sum(dim=1, keepdim=True)
    diffusion = 0.5 * sigma ** 2 * hessian_trace
    return drift + diffusion + alpha * value


def box_min_radius_squared(lower, upper):
    crosses_zero = (lower <= 0) & (upper >= 0)
    min_abs = torch.where(
        crosses_zero, torch.zeros_like(lower),
        torch.minimum(lower.abs(), upper.abs()))
    return min_abs.square().sum()


def make_domain_boxes(splits, device, exclude_radius=0.0):
    edges = torch.linspace(-1.0, 1.0, splits + 1, device=device)
    lowers = []
    uppers = []
    for i, j in itertools.product(range(splits), range(splits)):
        lower = torch.stack((edges[i], edges[j]))
        upper = torch.stack((edges[i + 1], edges[j + 1]))
        if (
                exclude_radius > 0
                and box_min_radius_squared(lower, upper) < exclude_radius ** 2):
            continue
        lowers.append(lower)
        uppers.append(upper)
    if not lowers:
        raise ValueError(
            'No certification boxes remain after excluding the origin radius; '
            'reduce --exclude-origin-radius or increase --grid-splits.')
    return torch.stack(lowers), torch.stack(uppers)


def interval_product_bounds(a_lower, a_upper, b_lower, b_upper):
    products = torch.stack((
        a_lower * b_lower,
        a_lower * b_upper,
        a_upper * b_lower,
        a_upper * b_upper,
    ), dim=0)
    return products.min(dim=0).values, products.max(dim=0).values


def make_bound_opts(args):
    return {
        'optimize_bound_args': {
            'iteration': args.alpha_crown_iterations,
            'lr_alpha': args.alpha_crown_lr,
        },
        'sigmoid_second_grad_relaxation': (
            args.sigmoid_second_grad_relaxation),
    }


def make_bounded_module(module, center, bound_opts):
    return BoundedModule(
        module, center, bound_opts=copy.deepcopy(bound_opts))


def certify_gap_direct(
        model, x_lower, x_upper, alpha, sigma, method, batch_size,
        bound_opts):
    """Certify the whole generator expression as one LiRPA graph.

    This is potentially tighter than ``certify_gap_separate``, but it exercises
    nested Jacobian/Hessian graph expansion through broadcasted products. Some
    operator shape paths are not supported by the current Hessian prototype.
    """
    upper_bounds = []
    lower_bounds = []
    for start in range(0, x_lower.shape[0], batch_size):
        end = min(start + batch_size, x_lower.shape[0])
        batch_lower = x_lower[start:end]
        batch_upper = x_upper[start:end]
        center = 0.5 * (batch_lower + batch_upper)
        bounded_input = BoundedTensor(
            center,
            PerturbationLpNorm(
                norm=float('inf'), x_L=batch_lower, x_U=batch_upper))
        bounded_model = make_bounded_module(
            OUGeneratorGapWrapper(model, alpha, sigma), center, bound_opts)
        lower, upper = bounded_model.compute_bounds(
            x=(bounded_input,), method=method)
        lower_bounds.append(lower.detach())
        upper_bounds.append(upper.detach())
    return torch.cat(lower_bounds, dim=0), torch.cat(upper_bounds, dim=0)


def summarize_bound_tensor(lower, upper):
    if lower is None or upper is None:
        return 'bounds unavailable'
    width = upper - lower
    return (
        f'lower=[{lower.min().item():.3e}, {lower.max().item():.3e}], '
        f'upper=[{upper.min().item():.3e}, {upper.max().item():.3e}], '
        f'width_max={width.max().item():.3e}')


def node_shape(node):
    output_shape = getattr(node, 'output_shape', None)
    if output_shape is not None:
        return tuple(output_shape)
    forward_value = getattr(node, 'forward_value', None)
    if hasattr(forward_value, 'shape'):
        return tuple(forward_value.shape)
    lower = getattr(node, 'lower', None)
    if hasattr(lower, 'shape'):
        return tuple(lower.shape)
    return 'unknown'


def summarize_mul_interval(node):
    x, y = node.inputs
    if (
            x.lower is None or x.upper is None
            or y.lower is None or y.upper is None):
        return None
    lower, upper = interval_product_bounds(
        x.lower, x.upper, y.lower, y.upper)
    return summarize_bound_tensor(lower, upper)


def compute_separate_component_bounds(
        model, box_lower, box_upper, alpha, sigma, method, bound_opts):
    center = 0.5 * (box_lower + box_upper)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(
            norm=float('inf'), x_L=box_lower, x_U=box_upper))

    value_model = make_bounded_module(model, center, bound_opts)
    value_lower, value_upper = value_model.compute_bounds(
        x=(bounded_input,), method=method)

    jacobian_model = make_bounded_module(
        JacobianWrapper(model), center, bound_opts)
    jacobian_lower, jacobian_upper = jacobian_model.compute_bounds(
        x=(bounded_input,), method=method)

    hessian_model = make_bounded_module(
        HessianWrapper(model), center, bound_opts)
    hessian_lower, hessian_upper = hessian_model.compute_hessian_bounds(
        bounded_input, method=method)

    trace_model = make_bounded_module(
        HessianTraceWrapper(model), center, bound_opts)
    trace_lower, trace_upper = trace_model.compute_bounds(
        x=(bounded_input,), method=method)

    drift_lower = torch.zeros_like(value_lower)
    drift_upper = torch.zeros_like(value_upper)
    for dim in range(2):
        minus_x_lower = -box_upper[:, dim:dim + 1]
        minus_x_upper = -box_lower[:, dim:dim + 1]
        term_lower, term_upper = interval_product_bounds(
            minus_x_lower, minus_x_upper,
            jacobian_lower[:, 0, dim:dim + 1],
            jacobian_upper[:, 0, dim:dim + 1])
        drift_lower = drift_lower + term_lower
        drift_upper = drift_upper + term_upper

    entry_trace_lower = (
        hessian_lower[:, 0, 0, 0:1] + hessian_lower[:, 0, 1, 1:2])
    entry_trace_upper = (
        hessian_upper[:, 0, 0, 0:1] + hessian_upper[:, 0, 1, 1:2])

    entry_diffusion_lower = 0.5 * sigma ** 2 * entry_trace_lower
    entry_diffusion_upper = 0.5 * sigma ** 2 * entry_trace_upper
    trace_diffusion_lower = 0.5 * sigma ** 2 * trace_lower
    trace_diffusion_upper = 0.5 * sigma ** 2 * trace_upper

    if alpha >= 0:
        value_term_lower = alpha * value_lower
        value_term_upper = alpha * value_upper
    else:
        value_term_lower = alpha * value_upper
        value_term_upper = alpha * value_lower

    separate_lower = drift_lower + entry_diffusion_lower + value_term_lower
    separate_upper = drift_upper + entry_diffusion_upper + value_term_upper
    trace_substituted_lower = (
        drift_lower + trace_diffusion_lower + value_term_lower)
    trace_substituted_upper = (
        drift_upper + trace_diffusion_upper + value_term_upper)

    return {
        'value': (value_lower, value_upper),
        'jacobian': (jacobian_lower, jacobian_upper),
        'drift': (drift_lower, drift_upper),
        'entry_trace': (entry_trace_lower, entry_trace_upper),
        'direct_trace': (trace_lower, trace_upper),
        'entry_diffusion': (entry_diffusion_lower, entry_diffusion_upper),
        'direct_trace_diffusion': (
            trace_diffusion_lower, trace_diffusion_upper),
        'value_term': (value_term_lower, value_term_upper),
        'separate_gap': (separate_lower, separate_upper),
        'trace_substituted_gap': (
            trace_substituted_lower, trace_substituted_upper),
    }


def diagnose_direct_muls(
        model, box_lower, box_upper, alpha, sigma, method, bound_opts,
        samples=2048):
    center = 0.5 * (box_lower + box_upper)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(
            norm=float('inf'), x_L=box_lower, x_U=box_upper))
    bounded_model = make_bounded_module(
        OUGeneratorGapWrapper(model, alpha, sigma), center, bound_opts)
    direct_lower, direct_upper = bounded_model.compute_bounds(
        x=(bounded_input,), method=method)
    components = compute_separate_component_bounds(
        model, box_lower, box_upper, alpha, sigma, method, bound_opts)

    random_samples = (
        box_lower
        + torch.rand(samples, 2, device=box_lower.device)
        * (box_upper - box_lower))
    sampled_gap = exact_network_gap(model, random_samples, alpha, sigma)

    print('\nDirect vs separate diagnostics')
    print(
        'box lower/upper: '
        f'{box_lower.squeeze(0).tolist()} -> {box_upper.squeeze(0).tolist()}')
    print(
        'direct bound on this box: '
        f'[{direct_lower.min().item():.6e}, '
        f'{direct_upper.max().item():.6e}]')
    print(
        'sampled exact gap on this box: '
        f'[{sampled_gap.min().item():.6e}, '
        f'{sampled_gap.max().item():.6e}]')
    print(
        'separate assembled gap: '
        f'{summarize_bound_tensor(*components["separate_gap"])}')
    print(
        'separate gap with direct trace substituted: '
        f'{summarize_bound_tensor(*components["trace_substituted_gap"])}')
    print(
        'drift interval-product contribution: '
        f'{summarize_bound_tensor(*components["drift"])}')
    print(
        'value contribution: '
        f'{summarize_bound_tensor(*components["value_term"])}')
    print(
        'entry-wise summed Hessian trace: '
        f'{summarize_bound_tensor(*components["entry_trace"])}')
    print(
        'direct Hessian trace bound: '
        f'{summarize_bound_tensor(*components["direct_trace"])}')
    print(
        'entry-wise diffusion contribution: '
        f'{summarize_bound_tensor(*components["entry_diffusion"])}')
    print(
        'direct-trace diffusion contribution: '
        f'{summarize_bound_tensor(*components["direct_trace_diffusion"])}')

    found = False
    for node in bounded_model.nodes():
        if type(node).__name__ != 'BoundMul' or len(node.inputs) != 2:
            continue
        if not all(getattr(inp, 'perturbed', False) for inp in node.inputs):
            continue
        found = True
        print(f'\n{node.name}: {type(node).__name__}')
        print(
            '  output '
            f'shape={node_shape(node)} '
            f'{summarize_bound_tensor(node.lower, node.upper)}')
        interval_summary = summarize_mul_interval(node)
        if interval_summary is not None:
            print(f'  interval product estimate {interval_summary}')
        for i, inp in enumerate(node.inputs):
            print(
                f'  input {i} {inp.name}: {type(inp).__name__} '
                f'shape={node_shape(inp)} '
                f'{summarize_bound_tensor(inp.lower, inp.upper)}')
    if not found:
        print('No BoundMul nodes with two perturbed inputs were found.')


def certify_gap_separate(
        model, x_lower, x_upper, alpha, sigma, method, batch_size,
        bound_opts):
    upper_bounds = []
    lower_bounds = []
    for start in range(0, x_lower.shape[0], batch_size):
        end = min(start + batch_size, x_lower.shape[0])
        batch_lower = x_lower[start:end]
        batch_upper = x_upper[start:end]
        center = 0.5 * (batch_lower + batch_upper)
        bounded_input = BoundedTensor(
            center,
            PerturbationLpNorm(
                norm=float('inf'), x_L=batch_lower, x_U=batch_upper))

        value_model = make_bounded_module(model, center, bound_opts)
        value_lower, value_upper = value_model.compute_bounds(
            x=(bounded_input,), method=method)

        jacobian_model = make_bounded_module(
            JacobianWrapper(model), center, bound_opts)
        jacobian_lower, jacobian_upper = jacobian_model.compute_bounds(
            x=(bounded_input,), method=method)

        hessian_model = make_bounded_module(
            HessianWrapper(model), center, bound_opts)
        hessian_lower, hessian_upper = hessian_model.compute_hessian_bounds(
            bounded_input, method=method)

        drift_upper = 0.0
        drift_lower = 0.0
        for dim in range(2):
            minus_x_lower = -batch_upper[:, dim:dim + 1]
            minus_x_upper = -batch_lower[:, dim:dim + 1]
            term_lower, term_upper = interval_product_bounds(
                minus_x_lower, minus_x_upper,
                jacobian_lower[:, 0, dim:dim + 1],
                jacobian_upper[:, 0, dim:dim + 1])
            drift_upper = drift_upper + term_upper
            drift_lower = drift_lower + term_lower

        diffusion_upper = 0.5 * sigma ** 2 * (
            hessian_upper[:, 0, 0, 0:1]
            + hessian_upper[:, 0, 1, 1:2])
        diffusion_lower = 0.5 * sigma ** 2 * (
            hessian_lower[:, 0, 0, 0:1]
            + hessian_lower[:, 0, 1, 1:2])

        if alpha >= 0:
            value_term_lower = alpha * value_lower
            value_term_upper = alpha * value_upper
        else:
            value_term_lower = alpha * value_upper
            value_term_upper = alpha * value_lower

        lower_bounds.append(
            drift_lower + diffusion_lower + value_term_lower)
        upper_bounds.append(
            drift_upper + diffusion_upper + value_term_upper)
    return torch.cat(lower_bounds, dim=0), torch.cat(upper_bounds, dim=0)


def certify_gap(
        model, x_lower, x_upper, alpha, sigma, method, batch_size,
        certifier, bound_opts):
    if certifier == 'direct':
        return certify_gap_direct(
            model, x_lower, x_upper, alpha, sigma, method, batch_size,
            bound_opts)
    if certifier == 'separate':
        return certify_gap_separate(
            model, x_lower, x_upper, alpha, sigma, method, batch_size,
            bound_opts)
    raise ValueError(f'Unknown certifier: {certifier}')


def print_summary(
        fit_max_error, fit_mean_error, samples, network_gap,
        quadratic_gap, certified_lower, certified_upper, alpha, sigma,
        exclude_radius, certified_boxes, total_boxes):
    worst_network_gap = network_gap.max().item()
    worst_quadratic_gap = quadratic_gap.max().item()
    worst_certified_gap = certified_upper.max().item()
    worst_certified_index = certified_upper.argmax().item()
    worst_sample = samples[network_gap.argmax().item()]

    print('\n2D OU Lyapunov benchmark')
    print(f'alpha: {alpha:.4f}, sigma: {sigma:.4f}')
    if exclude_radius > 0:
        print(
            'certification domain: '
            f'[-1, 1]^2 with boxes intersecting '
            f'||x|| < {exclude_radius:.6f} excluded')
    else:
        print('certification domain: [-1, 1]^2')
    print(f'certification boxes: {certified_boxes}/{total_boxes}')
    print(
        'fit error to 0.5 ||x||^2: '
        f'max={fit_max_error:.6e}, mean={fit_mean_error:.6e}')
    print(
        'worst sampled network gap '
        f'L V + alpha V: {worst_network_gap:.6e} '
        f'at x=({worst_sample[0].item():.4f}, '
        f'{worst_sample[1].item():.4f})')
    print(
        'worst sampled analytical quadratic gap: '
        f'{worst_quadratic_gap:.6e}')
    print(
        'worst certified upper gap over boxes: '
        f'{worst_certified_gap:.6e} '
        f'(box index {worst_certified_index})')
    print(
        'certified gap range over boxes: '
        f'[{certified_lower.min().item():.6e}, '
        f'{certified_upper.max().item():.6e}]')
    print(
        'certified exponential supermartingale condition: '
        f'{"YES" if worst_certified_gap <= 0 else "NO"}')
    print(
        'quadratic sanity check at origin: '
        f'L V(0) + alpha V(0) = sigma^2 = {sigma ** 2:.6e}')
    if alpha < 2:
        critical_radius = sigma / (1.0 - 0.5 * alpha) ** 0.5
        print(
            'quadratic gap becomes nonpositive for '
            f'||x|| >= {critical_radius:.6e}')


def run_benchmark(args):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = SoftplusLyapunovMLP(
        hidden_width=args.hidden_width,
        beta=args.softplus_beta).to(device).double()

    fit_max_error, fit_mean_error = train_certificate(
        model, args.epochs, args.train_samples, args.batch_size,
        args.learning_rate, device)

    samples = sample_domain(
        args.eval_samples, device, args.exclude_origin_radius)
    network_gap = exact_network_gap(model, samples, args.alpha, args.sigma)
    quadratic_gap = analytic_quadratic_gap(samples, args.alpha, args.sigma)

    total_boxes = args.grid_splits ** 2
    x_lower, x_upper = make_domain_boxes(
        args.grid_splits, device, args.exclude_origin_radius)
    bound_opts = make_bound_opts(args)
    certified_lower, certified_upper = certify_gap(
        model, x_lower, x_upper, args.alpha, args.sigma,
        args.bound_method, args.cert_batch_size, args.certifier, bound_opts)
    worst_certified_index = certified_upper.argmax().item()

    print_summary(
        fit_max_error, fit_mean_error, samples, network_gap,
        quadratic_gap, certified_lower, certified_upper,
        args.alpha, args.sigma, args.exclude_origin_radius,
        x_lower.shape[0], total_boxes)
    if args.diagnose_direct_muls:
        diagnose_direct_muls(
            model,
            x_lower[worst_certified_index:worst_certified_index + 1],
            x_upper[worst_certified_index:worst_certified_index + 1],
            args.alpha, args.sigma, args.bound_method, bound_opts)


def parse_args():
    parser = argparse.ArgumentParser(
        description='2D Ornstein-Uhlenbeck Lyapunov verification benchmark.')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--hidden-width', type=int, default=16)
    parser.add_argument('--softplus-beta', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--train-samples', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--eval-samples', type=int, default=4096)
    parser.add_argument('--grid-splits', type=int, default=8)
    parser.add_argument(
        '--exclude-origin-radius', type=float, default=0.2,
        help=(
            'Exclude certification boxes that intersect the ball '
            '||x|| < radius, and draw evaluation samples outside it.'))
    parser.add_argument('--cert-batch-size', type=int, default=8)
    parser.add_argument(
        '--bound-method',
        choices=('IBP', 'CROWN', 'alpha-CROWN'),
        default='alpha-CROWN')
    parser.add_argument(
        '--alpha-crown-iterations', type=int, default=20,
        help='Number of optimization iterations for alpha-CROWN bounds.')
    parser.add_argument(
        '--alpha-crown-lr', type=float, default=0.1,
        help='Learning rate for alpha-CROWN relaxation parameters.')
    parser.add_argument(
        '--sigmoid-second-grad-relaxation',
        choices=('tangent', 'piecewise'), default='tangent',
        help='Relaxation method for sigmoid second-gradient bounds.')
    parser.add_argument(
        '--certifier', choices=('direct', 'separate'), default='direct',
        help=(
            'direct bounds the full generator graph and is experimental; '
            'separate bounds V, grad V, and Hessian V independently.'))
    parser.add_argument(
        '--diagnose-direct-muls', action='store_true',
        help=(
            'After certification, rerun the direct graph on the worst box and '
            'print BoundMul nodes whose two inputs are both perturbed.'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


if __name__ == '__main__':
    run_benchmark(parse_args())
