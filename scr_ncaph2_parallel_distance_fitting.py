import numpy as np
import pandas as pd
import torch
from tqdm import tqdm  # for progress monitoring
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set device for PyTorch
device_ids = [f'cuda:{i}' for i in range(4)]  # Four GPUs available, assuming device IDs 0-3

# Initial parameters for MCMC
initial_r0 = 5.0
initial_sigma = 5.0
n_iterations = 3000
T_burnin = 300
number_of_samples = 30000  # Number of Monte Carlo samples

# Proposal distribution standard deviations for r0 and sigma
eta_r0 = 0.1
eta_sigma = 0.1

# Priors for r0 and sigma
mu_r0 = 5.0  # Prior mean for r0
tau_r0 = 5.0  # Prior std for r0
mu_sigma = 5.0  # Prior mean for sigma
tau_sigma = 5.0  # Prior std for sigma

# GPU-optimized function to compute pairwise distances
def compute_pairwise_distances(xy_coords, device):
    n = xy_coords.shape[0]
    pairwise_distances = torch.zeros((n * (n - 1)) // 2, device=device)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = xy_coords[i] - xy_coords[j]
            pairwise_distances[idx] = torch.sqrt(torch.sum(diff ** 2))
            idx += 1
    return pairwise_distances

# Radial Gaussian distribution
def P_r(r, r0, sigma, device):
    return torch.exp(-0.5 * ((r - r0) / sigma) ** 2) / (torch.sqrt(torch.tensor(2 * np.pi, device=device)) * sigma)

# Monte Carlo-based P(d | r0, sigma) using PyTorch
def P_d_given_r0_sigma_monte_carlo(d, r0, sigma, device, num_samples=500):
    r1_samples = torch.normal(mean=r0, std=sigma, size=(num_samples,), device=device)
    r2_samples = torch.normal(mean=r0, std=sigma, size=(num_samples,), device=device)
    
    valid = (r1_samples > 0) & (r2_samples > 0)
    r1_samples = r1_samples[valid]
    r2_samples = r2_samples[valid]
    
    argument = (r1_samples[:, None] ** 2 + r2_samples[:, None] ** 2 - d ** 2) / (2 * r1_samples[:, None] * r2_samples[:, None])
    H_vals = (torch.abs(argument) <= 1).float()
    
    integrand_vals = (P_r(r1_samples[:, None], r0, sigma, device) * P_r(r2_samples[:, None], r0, sigma, device)) / (2 * r1_samples[:, None] * r2_samples[:, None]) * H_vals
    integral_estimate = torch.mean(integrand_vals, axis=0)
    
    return (1 / (2 * np.pi ** 2)) * integral_estimate

# Vectorized log-likelihood function using Monte Carlo-based P(d | r0, sigma)
def log_likelihood_pairwise_monte_carlo(r0, sigma, pairwise_distances, device, num_samples=500):
    p_d = P_d_given_r0_sigma_monte_carlo(pairwise_distances, r0, sigma, device, num_samples)
    valid = p_d > 0
    log_likelihood = torch.sum(torch.log(p_d[valid]))
    log_likelihood += torch.sum(~valid * -1e10)
    return log_likelihood

# Prior functions for r0 and sigma
def log_prior_r0(r0):
    return -0.5 * ((r0 - mu_r0) / tau_r0) ** 2

def log_prior_sigma(sigma):
    return -0.5 * ((sigma - mu_sigma) / tau_sigma) ** 2 if sigma > 0 else -torch.inf

# MCMC function for a single cell line using minibatches
def run_mcmc_for_cell_line(cell_type_data, initial_r0, initial_sigma, eta_r0, eta_sigma, n_iterations, T_burnin, device, num_samples=500):
    torch.cuda.set_device(device)
    r0_current = torch.tensor(initial_r0, device=device)
    sigma_current = torch.tensor(initial_sigma, device=device)
    
    r0_samples = []
    sigma_samples = []
    
    mini_batches = []
    small_batch = torch.zeros(0, device=device)
    
    for i, xy_coords in enumerate(cell_type_data):
        xy_coords = torch.tensor(xy_coords, device=device)
        pairwise_distances = compute_pairwise_distances(xy_coords, device)
        small_batch = torch.cat((small_batch, pairwise_distances))
        
        if len(small_batch) > 100000 or i == len(cell_type_data) - 1:
            mini_batches.append(small_batch)
            small_batch = torch.zeros(0, device=device)
    
    for t in tqdm(range(n_iterations), desc="Processing MCMC iterations"):
        random.shuffle(mini_batches)
        
        r0_new = r0_current + torch.normal(mean=torch.tensor(0.0, device=device), std=torch.tensor(eta_r0, device=device))
        sigma_new = sigma_current + torch.normal(mean=torch.tensor(0.0, device=device), std=torch.tensor(eta_sigma, device=device))
        
        if sigma_new <= 0:
            sigma_new = sigma_current
        
        log_posterior_current = torch.tensor(0.0, device=device)
        log_posterior_new = torch.tensor(0.0, device=device)
        
        for pairwise_distances in mini_batches:
            log_likelihood_current = log_likelihood_pairwise_monte_carlo(r0_current, sigma_current, pairwise_distances, device, num_samples)
            log_likelihood_new = log_likelihood_pairwise_monte_carlo(r0_new, sigma_new, pairwise_distances, device, num_samples)
            
            log_posterior_current += log_likelihood_current
            log_posterior_new += log_likelihood_new
        
        log_posterior_current += log_prior_r0(r0_current) + log_prior_sigma(sigma_current)
        log_posterior_new += log_prior_r0(r0_new) + log_prior_sigma(sigma_new)
        
        alpha = torch.exp(log_posterior_new - log_posterior_current)
        if torch.rand(1, device=device) < alpha:
            r0_current = r0_new
            sigma_current = sigma_new
        
        if t >= T_burnin:
            r0_samples.append(r0_current.item())
            sigma_samples.append(sigma_current.item())
    
    r0_samples = np.array(r0_samples)
    sigma_samples = np.array(sigma_samples)
    
    return np.mean(r0_samples), np.mean(sigma_samples)

# Function to run MCMC for each cell line in parallel
def run_mcmc_for_all_cell_lines(eight_cell_lines, num_samples=500):
    cell_line_groups = eight_cell_lines.groupby('gene_symbol')['nuc_centered_spots']
    results = {}

    cell_types = list(cell_line_groups.groups.keys())
    batches = [cell_types[0], cell_types[1]]  # Split into two batches of 4

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_cell_type = {
            executor.submit(
                run_mcmc_for_cell_line,
                eight_cell_lines[eight_cell_lines['gene_symbol'] == cell_type]['nuc_centered_spots'],
                initial_r0,
                initial_sigma,
                eta_r0,
                eta_sigma,
                n_iterations,
                T_burnin,
                device=device_ids[i % 4],  # Assign each cell line to one of the 4 GPUs
                num_samples=num_samples
            ): cell_type
            for i, cell_type in enumerate(batches)
        }

        for future in tqdm(as_completed(future_to_cell_type), total=len(future_to_cell_type), desc="Cell Lines Batch"):
            cell_type = future_to_cell_type[future]
            try:
                r0_mean, sigma_mean = future.result()
                results[cell_type] = {'r0': r0_mean, 'sigma': sigma_mean}
                
                # Save results after each cell type
                with open(f'/vf/users/HiTIF/data/keikhosravia2/ripley_paper/scr_ncaph2_mcmc_results_gpu_3000/scr_ncaph2_bayesian_cell_line_distance_results_{cell_type}.pkl', 'wb') as f: pickle.dump(results, f)
            except Exception as exc:
                print(f'{cell_type} generated an exception: {exc}')

    return results

scr_df = pd.read_pickle('/vf/users/HiTIF/data/keikhosravia2/ripley_paper/scr_ncaph2_results/scrambled_simulations_complete_results.pkl')
ncaph2_df = pd.read_pickle('/vf/users/HiTIF/data/keikhosravia2/ripley_paper/scr_ncaph2_results/NCAPH2_simulations_complete_results.pkl')
eight_cell_lines = pd.concat([scr_df,ncaph2_df]).dropna().reset_index(drop=True)

cell_line_results = run_mcmc_for_all_cell_lines(eight_cell_lines, num_samples=number_of_samples)

# Save final results in a pickle file
with open('/vf/users/HiTIF/data/keikhosravia2/ripley_paper/scr_ncaph2_mcmc_results_gpu_3000/scr_ncaph2_bayesian_cell_line_distance_results_final.pkl', 'wb') as f:
    pickle.dump(cell_line_results, f)

print(cell_line_results)

