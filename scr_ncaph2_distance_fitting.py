import numpy as np
import pandas as pd
from scipy.stats import norm
from numba import njit
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # for progress monitoring
import pickle
import random

# Initial parameters for MCMC
initial_r0 = 5.0
initial_sigma = 5.0
n_iterations = 1000
T_burnin = 100
number_of_samples = 1500  # Number of Monte Carlo samples

# Proposal distribution standard deviations for r0 and sigma
eta_r0 = 0.1
eta_sigma = 0.1

# Priors for r0 and sigma
mu_r0 = 5.0  # Prior mean for r0
tau_r0 = 5.0  # Prior std for r0
mu_sigma = 5.0  # Prior mean for sigma
tau_sigma = 5.0  # Prior std for sigma

# Numba-optimized function to compute pairwise distances
@njit
def compute_pairwise_distances(xy_coords):
    n = xy_coords.shape[0]
    pairwise_distances = np.zeros((n * (n - 1)) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_distances[idx] = np.sqrt((xy_coords[i, 0] - xy_coords[j, 0])**2 + (xy_coords[i, 1] - xy_coords[j, 1])**2)
            idx += 1
    return pairwise_distances

# Radial Gaussian distribution
def P_r(r, r0, sigma):
    return norm.pdf(r, loc=r0, scale=sigma)

# Monte Carlo-based P(d | r0, sigma) with vectorized distances
def P_d_given_r0_sigma_monte_carlo(d, r0, sigma, num_samples=500):
    r1_samples = np.random.normal(loc=r0, scale=sigma, size=num_samples)
    r2_samples = np.random.normal(loc=r0, scale=sigma, size=num_samples)
    
    # Only keep valid samples where r1 and r2 are positive
    valid = (r1_samples > 0) & (r2_samples > 0)
    r1_samples = r1_samples[valid]
    r2_samples = r2_samples[valid]
    
    # Compute argument matrix for all distances
    argument = (r1_samples[:, None]**2 + r2_samples[:, None]**2 - d**2) / (2 * r1_samples[:, None] * r2_samples[:, None])
    
    # Heaviside function (H_vals), selecting valid samples
    H_vals = np.where(np.abs(argument) <= 1, 1.0, 0.0)
    
    # Compute the integrand for each distance d
    integrand_vals = (P_r(r1_samples[:, None], r0, sigma) * P_r(r2_samples[:, None], r0, sigma)) / (2 * r1_samples[:, None] * r2_samples[:, None]) * H_vals
    
    # Estimate the integral for each distance
    integral_estimate = np.mean(integrand_vals, axis=0)
    
    # Return the probability estimates for each distance
    return (1 / (2 * np.pi**2)) * integral_estimate

# Vectorized Log-likelihood function using Monte Carlo-based P(d | r0, sigma)
def log_likelihood_pairwise_monte_carlo(r0, sigma, pairwise_distances, num_samples=500):
    # Vectorized computation of P(d | r0, sigma) for all pairwise distances
    p_d = P_d_given_r0_sigma_monte_carlo(pairwise_distances, r0, sigma, num_samples)
    
    # Filter out invalid probabilities (p_d <= 0)
    valid = p_d > 0
    
    # Compute the log likelihood only for valid probabilities
    log_likelihood = np.sum(np.log(p_d[valid]))
    
    # Assign a large negative value for invalid distances
    log_likelihood += np.sum(~valid * -1e10)
    
    return log_likelihood

# Prior for r0
def log_prior_r0(r0):
    return -0.5 * ((r0 - mu_r0) / tau_r0)**2

# Prior for sigma
def log_prior_sigma(sigma):
    return -0.5 * ((sigma - mu_sigma) / tau_sigma)**2 if sigma > 0 else -np.inf

# MCMC function for a single cell line using minibatches of pairwise distances and Monte Carlo-based likelihood
def run_mcmc_for_cell_line(cell_type_data, initial_r0, initial_sigma, eta_r0, eta_sigma, n_iterations, T_burnin, num_samples=500):
    r0_current = initial_r0
    sigma_current = initial_sigma
    
    r0_samples = []
    sigma_samples = []
    
    mini_batches = []
    small_batch = np.zeros(0)
    
    # Precompute all pairwise distances and split into minibatches
    for i, xy_coords in enumerate(cell_type_data):
        xy_coords = np.array(xy_coords)
        pairwise_distances = compute_pairwise_distances(xy_coords)
        small_batch = np.concatenate((small_batch, pairwise_distances))
        
        # When the batch reaches a threshold size, add it to minibatches
        if len(small_batch) > 100000 or i == len(cell_type_data) - 1:
            mini_batches.append(small_batch)
            small_batch = np.zeros(0)
    
    for t in tqdm(range(n_iterations), desc="Processing MCMC iterations"):
        # Shuffle minibatches at the start of each iteration
        random.shuffle(mini_batches)
        
        r0_new = r0_current + np.random.normal(0, eta_r0)
        sigma_new = sigma_current + np.random.normal(0, eta_sigma)
        
        if sigma_new <= 0:
            sigma_new = sigma_current
        
        log_posterior_current = 0.0
        log_posterior_new = 0.0
        
        # Loop through shuffled minibatches to compute log likelihoods
        for pairwise_distances in mini_batches:
            log_likelihood_current = log_likelihood_pairwise_monte_carlo(r0_current, sigma_current, pairwise_distances, num_samples)
            log_likelihood_new = log_likelihood_pairwise_monte_carlo(r0_new, sigma_new, pairwise_distances, num_samples)
            
            log_posterior_current += log_likelihood_current
            log_posterior_new += log_likelihood_new
        
        # Add the log priors to the posterior
        log_posterior_current += log_prior_r0(r0_current) + log_prior_sigma(sigma_current)
        log_posterior_new += log_prior_r0(r0_new) + log_prior_sigma(sigma_new)
        
        # Metropolis-Hastings acceptance step
        alpha = np.exp(log_posterior_new - log_posterior_current)
        if np.random.uniform(0, 1) < alpha:
            r0_current = r0_new
            sigma_current = sigma_new
        
        if t >= T_burnin:
            r0_samples.append(r0_current)
            sigma_samples.append(sigma_current)
    
    r0_samples = np.array(r0_samples)
    sigma_samples = np.array(sigma_samples)
    
    r0_mean = np.mean(r0_samples)
    sigma_mean = np.mean(sigma_samples)
    
    return r0_mean, sigma_mean

# Function to run MCMC for each cell line in parallel using ProcessPoolExecutor
def run_mcmc_for_all_cell_lines(eight_cell_lines, num_samples=500):
    cell_line_groups = eight_cell_lines.groupby('gene_symbol')['nuc_centered_spots']
    results = {}
    
    with ProcessPoolExecutor(max_workers=180) as executor:
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
                num_samples
            ): cell_type
            for cell_type, cell_type_data in cell_line_groups
        }
        
        for future in tqdm(as_completed(future_to_cell_type), total=len(future_to_cell_type), desc="Cell Lines"):
            cell_type = future_to_cell_type[future]
            try:
                r0_mean, sigma_mean = future.result()
                results[cell_type] = {'r0': r0_mean, 'sigma': sigma_mean}
                
                # Save results after each iteration
                with open(f'/vf/users/HiTIF/data/keikhosravia2/ripley_paper/mcmc_results_100/scr_ncaph2_bayesian_cell_line_distance_results_{cell_type}.pkl', 'wb') as f:
                    pickle.dump(results, f)
            except Exception as exc:
                print(f'{cell_type} generated an exception: {exc}')
    
    return results

# Example usage:

scr_df = pd.read_pickle('/vf/users/HiTIF/data/keikhosravia2/ripley_paper/scr_ncaph2_results/scrambled_simulations_complete_results.pkl')
ncaph2_df = pd.read_pickle('/vf/users/HiTIF/data/keikhosravia2/ripley_paper/scr_ncaph2_results/NCAPH2_simulations_complete_results.pkl')
eight_cell_lines = pd.concat([scr_df,ncaph2_df]).dropna().reset_index(drop=True)

cell_line_results = run_mcmc_for_all_cell_lines(eight_cell_lines, num_samples=number_of_samples)

# Save final results in a pickle file
with open('/vf/users/HiTIF/data/keikhosravia2/ripley_paper/mcmc_results_100/scr_ncaph2_bayesian_cell_line_distance_results_final.pkl', 'wb') as f:
    pickle.dump(cell_line_results, f)

print(cell_line_results)

