"""
Lyapunov verification for 1D Geometric Brownian Motion using auto_LiRPA.

This demonstrates formally verifying the stability of a stochastic differential
equation (SDE) using certificate networks and auto_LiRPA bounds. An SDE converges
safely if the infinitesimal generator (expected change in energy) is negative.

For a GBM: dX_t = mu * X_t dt + sigma * X_t dW_t
Generator: G[V](x) = (dV/dx) * (mu * X_t) + (1/2) * (sigma^2) * (d2V/dx2) * (X_t^2)
                   = drift_term (Jacobian) + diffusion_term (Hessian)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.hessian import DirectHessianOP
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.bound_ops import JacobianOP

class HessianWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state):
        return DirectHessianOP.apply(self.model(state), state)

class JacobianWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state):
        return JacobianOP.apply(self.model(state), state)

def verify_geometric_brownian_motion(mu=-0.5, sigma=0.8, initial_state=1.0, epsilon=0.2, num_samples=1000):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(0)

    # Softplus is required because ReLU has a zero second derivative, which would 
    # incorrectly eliminate the diffusion penalty from our stochastic verification.
    certificate_model = nn.Sequential(
        nn.Linear(1, 16), nn.Softplus(), 
        nn.Linear(16, 16), nn.Softplus(), 
        nn.Linear(16, 1)
    ).double()

    # We force the outer weights to be positive to ensure the neural network 
    # forms a valid "energy bowl" shape that pulls the system toward the origin.
    with torch.no_grad():
        for parameter in certificate_model.parameters(): 
            parameter.uniform_(-1.0, 1.0)
        certificate_model[0].weight.uniform_(0.5, 1.5)
        certificate_model[4].weight.uniform_(0.5, 1.5)

    initial_state_tensor = torch.tensor([[initial_state]], dtype=torch.double)

    # We sample uniform random points within the perturbation box to find the 
    # empirical worst-case scenarios, which acts as a baseline for our formal bounds.
    sampled_states = initial_state_tensor + (torch.rand(num_samples, 1) * 2 * epsilon - epsilon)
    
    empirical_drifts = []
    empirical_diffusions = []
    empirical_generators = []

    for state in sampled_states:
        state.requires_grad_(True)
        energy_value = certificate_model(state)
        
        # We calculate the exact first and second derivatives for this specific point
        # to find the true physical behavior of the system.
        exact_jacobian = torch.autograd.grad(energy_value, state, create_graph=True)[0]
        exact_hessian = torch.autograd.functional.hessian(lambda x: certificate_model(x).squeeze(), state)

        drift_value = exact_jacobian.item() * (mu * state.item())
        diffusion_value = 0.5 * (sigma**2) * (state.item()**2) * exact_hessian.item()

        empirical_drifts.append(drift_value)
        empirical_diffusions.append(diffusion_value)
        empirical_generators.append(drift_value + diffusion_value)

    # We set up the perturbation box for auto_LiRPA to search for the absolute worst-case bounds.
    bounded_input_tensor = BoundedTensor(initial_state_tensor, PerturbationLpNorm(norm=float('inf'), eps=epsilon))
    
    # We ask auto_LiRPA to find the absolute minimum and maximum Jacobian values in the box.
    # We strictly use 'backward' for the Jacobian to ensure a tight drift bound.
    bounded_jacobian_model = BoundedModule(JacobianWrapper(certificate_model), initial_state_tensor)
    lower_jacobian, upper_jacobian = bounded_jacobian_model.compute_bounds(bounded_input_tensor, method='backward')
    
    # The drift term is Jacobian * mu * state. We must evaluate this product at all
    # extreme endpoints of the interval to guarantee we capture the worst-case drift.
    mu_state_endpoints = [mu * (initial_state + epsilon), mu * (initial_state - epsilon)]
    jacobian_endpoints = [lower_jacobian.item(), upper_jacobian.item()]
    certified_drift = max(jacobian * mu_state for jacobian in jacobian_endpoints for mu_state in mu_state_endpoints)

    # We ask auto_LiRPA to bound the Hessian value in the box using both IBP and backward methods.
    bounded_hessian_model = BoundedModule(HessianWrapper(certificate_model), initial_state_tensor)
    
    _, upper_hessian_ibp = bounded_hessian_model.compute_hessian_bounds(bounded_input_tensor, method='IBP')
    _, upper_hessian_backward = bounded_hessian_model.compute_hessian_bounds(bounded_input_tensor, method='backward')
    
    # The diffusion term scales quadratically with the state. The worst-case noise
    # injection always occurs at the maximum absolute distance from the origin within our box.
    max_state_squared = (initial_state + epsilon)**2
    
    certified_diffusion_ibp = 0.5 * (sigma**2) * max_state_squared * upper_hessian_ibp.item()
    certified_diffusion_backward = 0.5 * (sigma**2) * max_state_squared * upper_hessian_backward.item()
    
    certified_generator_ibp = certified_drift + certified_diffusion_ibp
    certified_generator_backward = certified_drift + certified_diffusion_backward

    print("\nVerification Results:")
    print(f"Max Empirical Drift:     {max(empirical_drifts):.4f}  | Certified Worst-Case Drift:           {certified_drift:.4f}")
    print(f"Max Empirical Diffusion: {max(empirical_diffusions):.4f}  | Certified Worst-Case Diffusion (IBP): {certified_diffusion_ibp:.4f}  | (Backward): {certified_diffusion_backward:.4f}")
    print(f"Max Empirical Generator: {max(empirical_generators):.4f}  | Certified Worst-Case Generator (IBP): {certified_generator_ibp:.4f}  | (Backward): {certified_generator_backward:.4f}\n")
    
    print(f"Is the physical system stable?            {'YES' if max(empirical_generators) < 0 else 'NO'}")
    print(f"Did auto_LiRPA (IBP) formally verify it?  {'YES' if certified_generator_ibp < 0 else 'NO'}")
    print(f"Did auto_LiRPA (Backward) verify it?      {'YES' if certified_generator_backward < 0 else 'NO'}\n")

    plot_verification_results(mu, sigma, initial_state, sampled_states.detach().numpy().flatten(), empirical_generators, certified_generator_ibp, certified_generator_backward)

def plot_verification_results(mu, sigma, initial_state, state_values, empirical_generators, certified_generator_ibp, certified_generator_backward):
    figure, (axis_trajectories, axis_generator) = plt.subplots(1, 2, figsize=(14, 5))
    
    time_steps = np.linspace(0, 5, 200)
    
    # We simulate standard Brownian motion to generate physical system paths
    brownian_motion = np.random.randn(15, 200) * np.sqrt(5/200)
    brownian_motion[:, 0] = 0
    brownian_motion = np.cumsum(brownian_motion, axis=1)
    
    simulated_trajectories = initial_state * np.exp((mu - 0.5 * sigma**2) * time_steps + sigma * brownian_motion)
    axis_trajectories.plot(time_steps, simulated_trajectories.T, color='gray', alpha=0.3)
    
    # Geometric Brownian Motion follows a Log-Normal distribution.
    log_drift = (mu - 0.5 * sigma**2) * time_steps
    log_volatility = sigma * np.sqrt(time_steps)
    
    # Helper to calculate exact trajectory percentiles using Z-scores
    def calculate_percentile(z_score): 
        return initial_state * np.exp(log_drift + z_score * log_volatility)

    # Draw the expected median trajectory
    axis_trajectories.plot(time_steps, calculate_percentile(0), color='darkorange', lw=2, label='Median Drift')
    
    # Draw Outer Band representing 90% of trajectories
    axis_trajectories.fill_between(time_steps, calculate_percentile(-1.645), calculate_percentile(1.645), 
                                   color='blue', alpha=0.08, label='90% Probability Envelope')
    
    # Draw Inner Band representing 50% of trajectories
    axis_trajectories.fill_between(time_steps, calculate_percentile(-0.674), calculate_percentile(0.674), 
                                   color='blue', alpha=0.15, label='50% Probability Core')
    
    axis_trajectories.set_title('Stochastic Physical Trajectories')
    axis_trajectories.set_xlabel('Time')
    axis_trajectories.set_ylabel('State')
    axis_trajectories.legend(loc='upper right')

    # We sort the sampled points to draw a clean, continuous empirical line
    sort_indices = np.argsort(state_values)
    sorted_states = state_values[sort_indices]
    sorted_empirical_generators = np.array(empirical_generators)[sort_indices]

    axis_generator.plot(sorted_states, sorted_empirical_generators, color='green', lw=2, label='Empirical Generator (True Physics)')
    axis_generator.axhline(certified_generator_ibp, color='orange', ls=':', lw=2, label='Certified Ceiling (IBP)')
    axis_generator.axhline(certified_generator_backward, color='red', ls='--', lw=2, label='Certified Ceiling (Backward)')
    axis_generator.axhline(0, color='black', lw=1, alpha=0.5, label='Safety Threshold')
    
    axis_generator.fill_between(sorted_states, sorted_empirical_generators, certified_generator_backward, color='red', alpha=0.1, label='Backward Verification Gap')
    
    axis_generator.set_title('Formal Safety Verification')
    axis_generator.set_xlabel('State (within perturbation box)')
    axis_generator.set_ylabel('Expected Change in Energy')
    axis_generator.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    verify_geometric_brownian_motion()