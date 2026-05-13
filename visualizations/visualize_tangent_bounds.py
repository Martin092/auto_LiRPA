import contextlib
import io

from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from auto_LiRPA.operators.s_shaped import BoundSigmoidSecondGrad, d2sigmoid, d3sigmoid


class BoundSigmoidSecondGradVisualizer(BoundSigmoidSecondGrad):

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self._relaxation_log = []

    def add_linear_relaxation(self, mask, type, k, x0, y0=None):
        self._relaxation_log.append({
            'mask': mask,
            'type': type,
            'k': k,
            'x0': x0,
            'y0': y0,
        })
        return super().add_linear_relaxation(mask, type, k, x0, y0)
    
    def compute_bounds_via_relaxation(self, lower, upper):
        class SimpleBoundedInput:
            def __init__(self, l, u):
                self.lower = l
                self.upper = u
        
        x = SimpleBoundedInput(lower, upper)

        self._relaxation_log = []
        
        with contextlib.redirect_stdout(io.StringIO()):
            self.init_linear_relaxation(x)

            self.bound_relax(x, init=False)

        def _last_by_type(line_type):
            for entry in reversed(self._relaxation_log):
                if entry['type'] == line_type:
                    return entry
            raise RuntimeError(f'No {line_type} relaxation was recorded.')

        lower_entry = _last_by_type('lower')
        upper_entry = _last_by_type('upper')

        def _as_float(value):
            if isinstance(value, torch.Tensor):
                return value.reshape(-1)[0].item()
            return float(value)

        return (
            _as_float(lower_entry['k']),
            _as_float(lower_entry['x0']),
            None if lower_entry['y0'] is None else _as_float(lower_entry['y0']),
            _as_float(upper_entry['k']),
            _as_float(upper_entry['x0']),
            None if upper_entry['y0'] is None else _as_float(upper_entry['y0']),
        )


def plot_bounds_single(ax, low, high, title, bound_obj=None):
    if bound_obj is None:
        bound_obj = BoundSigmoidSecondGradVisualizer()
    
    lower = torch.tensor([low])
    upper = torch.tensor([high])
    
    lower_k, lower_x0, lower_y0, upper_k, upper_x0, upper_y0 = bound_obj.compute_bounds_via_relaxation(lower, upper)
    
    grid = torch.linspace(low, high, 1000)
    y_grid = d2sigmoid(grid)
    
    lower_anchor_y = lower_y0 if lower_y0 is not None else d2sigmoid(torch.tensor(lower_x0)).item()
    upper_anchor_y = upper_y0 if upper_y0 is not None else d2sigmoid(torch.tensor(upper_x0)).item()

    y_lower_line = lower_k * (grid - lower_x0) + lower_anchor_y
    y_upper_line = upper_k * (grid - upper_x0) + upper_anchor_y
    envelope_area = torch.trapz((y_upper_line - y_lower_line).clamp(min=0), grid).item()
    
    ax.plot(grid.numpy(), y_grid.numpy(), 'b-', linewidth=2, label='d2sigmoid(x)')
    
    ax.plot(grid.numpy(), y_lower_line.numpy(), 'g--', linewidth=1.5, label='Lower bound')
    ax.plot(grid.numpy(), y_upper_line.numpy(), 'r--', linewidth=1.5, label='Upper bound')
    
    y_l = d2sigmoid(lower)
    y_u = d2sigmoid(upper)
    ax.plot([low, high], [y_l.item(), y_u.item()], 'ko', markersize=8, label='Endpoints')
    
    ax.plot(lower_x0, lower_anchor_y, 'g^', markersize=10, label='Lower tangent point')
    ax.plot(upper_x0, upper_anchor_y, 'r^', markersize=10, label='Upper tangent point')
    
    secant_slope = (y_u - y_l) / (upper - lower)
    y_secant = secant_slope * (grid - lower) + y_l
    ax.plot(grid.numpy(), y_secant.numpy(), 'k:', linewidth=1, alpha=0.5, label='Secant')
    
    ax.axvspan(low, high, alpha=0.1, color='gray')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('d2sigmoid(x)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(low - 0.5, high + 0.5)
    
    secant_slope_val = secant_slope.item()
    d3_lower = d3sigmoid(torch.tensor(lower_x0)).item()
    d3_upper = d3sigmoid(torch.tensor(upper_x0)).item()
    
    info_text = (
        f"Interval: [{low:.2f}, {high:.2f}]\n"
        f"Envelope area: {envelope_area:.6f}\n"
        f"Secant slope: {secant_slope_val:.6f}\n"
        f"d3σ(x_lower): {d3_lower:.6f}\n"
        f"d3σ(x_upper): {d3_upper:.6f}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def create_comparison_plot():
    fig, axes = plt.subplots(2, 3, figsize=(15, 15))
    fig.suptitle('Tangent-Based Bounds for d2sigmoid(x)', fontsize=14, fontweight='bold')
    
    bound_obj = BoundSigmoidSecondGradVisualizer()

    test_cases = [
        (axes[0, 0], -5.0, -2.0, "[-5 -2]"),
        (axes[0, 1], -5.0, -1.0, "[-5 -1]"),
        (axes[0, 2], -5.0, 1.0, "[-5 1]"),
        (axes[1, 0], -5.0, 2.0, "[-5 2]"),
        (axes[1, 1], -1.0, 0.5, "[-1 0.5]"),
        (axes[1, 2], -1.0, 1.5, "[-1 1.5]"),
    ]
    
    for ax, low, high, title in test_cases:
        plot_bounds_single(ax, low, high, title, bound_obj)
    
    handles = [
        mpatches.Patch(color='blue', label='d2sigmoid(x)'),
        mpatches.Patch(color='green', label='Lower bound'),
        mpatches.Patch(color='red', label='Upper bound'),
        mpatches.Patch(color='black', label='Secant line'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


def create_bound_envelope_plot():
    fig, axes = plt.subplots(2, 3, figsize=(15, 15))
    fig.suptitle('Bound Envelope Analysis', fontsize=14, fontweight='bold')
    
    bound_obj = BoundSigmoidSecondGradVisualizer()

    test_cases = [
        (axes[0, 0], -5.0, -2.0, "[-5 -2]"),
        (axes[0, 1], -5.0, -1.0, "[-5 -1]"),
        (axes[0, 2], -5.0, 1.0, "[-5 1]"),
        (axes[1, 0], -5.0, 2.0, "[-5 2]"),
        (axes[1, 1], -1.0, 0.5, "[-1 0.5]"),
        (axes[1, 2], -1.0, 1.5, "[-1 1.5]"),
    ]
    
    for ax, low, high, title in test_cases:
        lower = torch.tensor([low])
        upper = torch.tensor([high])
        
        lower_k, lower_x0, lower_y0, upper_k, upper_x0, upper_y0 = bound_obj.compute_bounds_via_relaxation(lower, upper)
        
        grid = torch.linspace(low, high, 1000)
        y_grid = d2sigmoid(grid)
        
        lower_anchor_y = lower_y0 if lower_y0 is not None else d2sigmoid(torch.tensor(lower_x0)).item()
        upper_anchor_y = upper_y0 if upper_y0 is not None else d2sigmoid(torch.tensor(upper_x0)).item()

        y_lower_line = lower_k * (grid - lower_x0) + lower_anchor_y
        y_upper_line = upper_k * (grid - upper_x0) + upper_anchor_y
        envelope_area = torch.trapz((y_upper_line - y_lower_line).clamp(min=0), grid).item()
        
        lower_margin = y_grid - y_lower_line
        upper_margin = y_upper_line - y_grid
        
        ax.plot(grid.numpy(), y_grid.numpy(), 'b-', linewidth=2.5, label='d2sigmoid(x)', zorder=3)
        ax.fill_between(grid.numpy(), y_lower_line.numpy(), y_upper_line.numpy(),
                        alpha=0.2, color='purple', label='Bound envelope')
        ax.plot(grid.numpy(), y_lower_line.numpy(), 'g--', linewidth=1.5, alpha=0.7)
        ax.plot(grid.numpy(), y_upper_line.numpy(), 'r--', linewidth=1.5, alpha=0.7)
        
        violations_lower = lower_margin < -1e-4
        violations_upper = upper_margin < -1e-4
        if violations_lower.any():
            ax.fill_between(grid[violations_lower].numpy(),
                            y_grid[violations_lower].numpy(),
                            y_lower_line[violations_lower].numpy(),
                            alpha=0.5, color='red', label='Lower violation')
        if violations_upper.any():
            ax.fill_between(grid[violations_upper].numpy(),
                            y_upper_line[violations_upper].numpy(),
                            y_grid[violations_upper].numpy(),
                            alpha=0.5, color='orange', label='Upper violation')
        
        ax.set_xlabel('x')
        ax.set_ylabel('d2sigmoid(x)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        min_lower_margin = lower_margin.min().item()
        min_upper_margin = upper_margin.min().item()
        ax.text(0.98, 0.02, f'Envelope area: {envelope_area:.6f}\nMin lower margin: {min_lower_margin:.2e}\nMin upper margin: {min_upper_margin:.2e}',
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    return fig


def save_all_plots():
    """Generate and save all visualization plots."""
    print("Generating visualization plots...")
    
    import os
    os.makedirs('visualizations_opt', exist_ok=True)
    
    print("  - Creating comparison plot...")
    fig1 = create_comparison_plot()
    fig1.savefig('./visualizations/tangent_bounds_comparison.png', dpi=150, bbox_inches='tight')
    print("    Saved: visualizations/tangent_bounds_comparison.png")
    
    print("  - Creating bound envelope plot...")
    fig2 = create_bound_envelope_plot()
    fig2.savefig('./visualizations/bound_envelope.png', dpi=150, bbox_inches='tight')
    print("    Saved: visualizations/bound_envelope.png")
    
    print("\nAll visualizations saved to ./visualizations/")
    print("\nYou can view the plots interactively by running:")
    print("  python -c \"from visualize_tangent_bounds import *; create_comparison_plot(); import matplotlib.pyplot as plt; plt.show()\"")
    
    return fig1, fig2


def show_interactive():
    print("Generating interactive visualization plots...")
    
    fig1 = create_comparison_plot()
    fig2 = create_bound_envelope_plot()
    
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize tangent-based bounds')
    parser.add_argument('--save', action='store_true', help='Save plots to ./visualizations/')
    parser.add_argument('--show', action='store_true', help='Show interactive plots')
    args = parser.parse_args()

    Path("").mkdir(exist_ok=True)
    
    if args.save or (not args.save and not args.show):
        save_all_plots()
    
    if args.show:
        show_interactive()
