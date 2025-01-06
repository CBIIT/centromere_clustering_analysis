import torch
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from tqdm import tqdm
import os
import multiprocessing

# Ensure the 'spawn' start method is used
multiprocessing.set_start_method('spawn', force=True)

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initial parameters for MCMC
initial_r0 = 5.0
initial_sigma = 5.0
n_iterations = 1000
T_burnin = 100
num_samples = 250000  # Number of Monte Carlo samples

# Proposal distribution standard deviations for r0 and sigma
eta_r0 = 0.1
eta_sigma = 0.1

# Priors for r0 and sigma
mu_r0 = 5.0  # Prior mean for r0
tau_r0 = 5.0  # Prior std for r0
mu_sigma = 5.0  # Prior mean for sigma
tau_sigma = 5.0  # Prior std for sigma

def compute_pairwise_distances(xy_coords):
    n = xy_coords.shape[0]
    pairwise_distances = torch.zeros((n * (n - 1)) // 2, device=device)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Ensure subtraction and summation are done on tensors
            diff = xy_coords[i] - xy_coords[j]
            dist_squared = torch.sum(diff ** 2)
            pairwise_distances[idx] = torch.sqrt(dist_squared)
            idx += 1
    return pairwise_distances


# Radial Gaussian distribution (using PyTorch)
def P_r(r, r0, sigma):
    return torch.exp(-0.5 * ((r - r0) / sigma) ** 2) / (torch.sqrt(torch.tensor(2 * torch.pi, device=device)) * sigma)

# Monte Carlo-based P(d | r0, sigma) using PyTorch
def P_d_given_r0_sigma_monte_carlo(d, r0, sigma, num_samples=20000):
    # Sample r1 and r2 in a single batch
    r1_samples = torch.normal(mean=r0, std=sigma, size=(num_samples,), device=device)
    r2_samples = torch.normal(mean=r0, std=sigma, size=(num_samples,), device=device)
    
    # Ensure r1 and r2 are positive
    valid = (r1_samples > 0) & (r2_samples > 0)
    r1_samples = r1_samples[valid]
    r2_samples = r2_samples[valid]
    
    # Compute the argument for the Heaviside function
    argument = (r1_samples**2 + r2_samples**2 - d**2) / (2 * r1_samples * r2_samples)
    
    # Apply the Heaviside function
    H_vals = (torch.abs(argument) <= 1).float()
    
    # Compute the integrand
    integrand_vals = (P_r(r1_samples, r0, sigma) * P_r(r2_samples, r0, sigma)) / (2 * r1_samples * r2_samples) * H_vals
    
    # Estimate the integral using the average value multiplied by the sample space
    integral_estimate = torch.mean(integrand_vals)
    
    return (1 / (2 * torch.pi**2)) * integral_estimate

# Log-likelihood function using Monte Carlo-based P(d | r0, sigma)
def log_likelihood_pairwise_monte_carlo(r0, sigma, pairwise_distances, num_samples=20000):
    log_likelihood = torch.tensor(0.0, device=device)
    for d in pairwise_distances:
        p_d = P_d_given_r0_sigma_monte_carlo(d, r0, sigma, num_samples)
        if p_d > 0:
            log_likelihood += torch.log(p_d)
        else:
            # Assign a very low probability to avoid -inf
            log_likelihood += -1e10  # or another sufficiently low value
    return log_likelihood

# Prior for r0 (using PyTorch)
def log_prior_r0(r0):
    return -0.5 * ((r0 - mu_r0) / tau_r0) ** 2

# Prior for sigma (using PyTorch)
def log_prior_sigma(sigma):
    return -0.5 * ((sigma - mu_sigma) / tau_sigma) ** 2 if sigma > 0 else -torch.inf

# MCMC function for a single cell line using pairwise distances and Monte Carlo-based likelihood
def run_mcmc_for_cell_line(cell_type_data, initial_r0, initial_sigma, eta_r0, eta_sigma, n_iterations, T_burnin, num_samples=20000, save_file=None):
    # Initialize current values of r0 and sigma as tensors
    r0_current = torch.tensor(initial_r0, device=device, dtype=torch.float32)
    sigma_current = torch.tensor(initial_sigma, device=device, dtype=torch.float32)
    
    r0_samples = []
    sigma_samples = []
    
    # Precompute all pairwise distances for efficiency
    all_pairwise_distances = []
    for xy_coords in cell_type_data:
        xy_coords = torch.tensor(xy_coords, device=device, dtype=torch.float32)  # Move data to GPU and ensure they are tensors
        pairwise_distances = compute_pairwise_distances(xy_coords)
        all_pairwise_distances.append(pairwise_distances)
    
    # MCMC loop
    for t in tqdm(range(n_iterations), desc="Processing MCMC iterations"):
        # Propose new values for r0 and sigma
        r0_new = r0_current + torch.normal(mean=torch.tensor(0.0, device=device), std=torch.tensor(eta_r0, device=device))
        sigma_new = sigma_current + torch.normal(mean=torch.tensor(0.0, device=device), std=torch.tensor(eta_sigma, device=device))
        
        # Check for positive sigma
        if sigma_new <= 0:
            sigma_new = sigma_current
        
        # Compute log-posterior for current and new parameters
        log_posterior_current = torch.tensor(0.0, device=device)
        log_posterior_new = torch.tensor(0.0, device=device)
        
        # Compute log-likelihoods
        for pairwise_distances in all_pairwise_distances:
            log_likelihood_current = log_likelihood_pairwise_monte_carlo(r0_current, sigma_current, pairwise_distances, num_samples)
            log_likelihood_new = log_likelihood_pairwise_monte_carlo(r0_new, sigma_new, pairwise_distances, num_samples)
            
            log_posterior_current += log_likelihood_current
            log_posterior_new += log_likelihood_new
        
        # Add prior contributions (Gaussian priors for r0 and sigma)
        log_posterior_current += log_prior_r0(r0_current) + log_prior_sigma(sigma_current)
        log_posterior_new += log_prior_r0(r0_new) + log_prior_sigma(sigma_new)
        
        # Calculate acceptance ratio
        alpha = torch.exp(log_posterior_new - log_posterior_current)
        
        # Accept or reject the new values
        if torch.rand(1, device=device) < alpha:
            r0_current = r0_new
            sigma_current = sigma_new
        
        # Store the samples after burn-in
        if t >= T_burnin:
            r0_samples.append(r0_current.item())
            sigma_samples.append(sigma_current.item())
        
        # Save progress after every iteration (after burn-in)
        if t >= T_burnin and save_file:
            with open(save_file, 'wb') as f:
                pickle.dump({'r0': r0_samples, 'sigma': sigma_samples}, f)
    
    # Convert samples to arrays (on CPU for further processing)
    r0_samples = np.array(r0_samples)
    sigma_samples = np.array(sigma_samples)
    
    # Summary statistics
    r0_mean = np.mean(r0_samples)
    sigma_mean = np.mean(sigma_samples)
    
    return r0_mean, sigma_mean

# Function to run MCMC for each cell line in parallel using ProcessPoolExecutor
def run_mcmc_for_all_cell_lines(eight_cell_lines, num_samples=20000):
    cell_line_groups = eight_cell_lines.groupby('cell_type')['nuc_centered_spots']
    results = {}
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_cell_type = {
            executor.submit(
                run_mcmc_for_cell_line,
                cell_type_data,
                initial_r0,
                initial_sigma,
                eta_r0,
                eta_sigma,
                n_iterations,
                T_burnin,
                num_samples,
                save_file=f'/vf/users/HiTIF/data/keikhosravia2/ripley_paper/eight_cell_lines/gpu_bayesian/bayesian_cell_line_results_{cell_type}.pkl'  # Save file for each cell line
            ): cell_type
            for cell_type, cell_type_data in cell_line_groups
        }
        
        for future in tqdm(as_completed(future_to_cell_type), total=len(future_to_cell_type), desc="Processing cell lines"):
            cell_type = future_to_cell_type[future]
            try:
                r0_mean, sigma_mean = future.result()
                results[cell_type] = {'r0': r0_mean, 'sigma': sigma_mean}
            except Exception as exc:
                print(f'{cell_type} generated an exception: {exc}')
    
    return results

if __name__ == '__main__':
    # Example usage:
    eight_cell_lines = pd.read_pickle(r'/vf/users/HiTIF/data/keikhosravia2/ripley_paper/eight_cell_lines/8cell_lines_complete_results.pkl')
    cell_line_results = run_mcmc_for_all_cell_lines(eight_cell_lines, num_samples=20000)

    # Save the final results
    with open('/vf/users/HiTIF/data/keikhosravia2/ripley_paper/eight_cell_lines/gpu_bayesian/bayesian_cell_line_distance_results.pkl', 'wb') as f:
        pickle.dump(cell_line_results, f)
    print(cell_line_results)

