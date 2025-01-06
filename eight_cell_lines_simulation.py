import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pickle
import pandas as pd
from spot_pattern_generator_functions import SpotGenerator
from skimage.io import imread, imsave
from skimage.measure import regionprops, regionprops_table
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from clustering import ClusteringAnalysis
from tqdm import tqdm  # Import tqdm for progress tracking
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import warnings
import sys
from scipy.ndimage import label, distance_transform_edt
analysis = ClusteringAnalysis()

warnings.filterwarnings("ignore", category=RuntimeWarning, module='astropy.stats.spatial')


eight_cell_lines = pd.read_pickle(r'/vf/users/HiTIF/data/keikhosravia2/ripley_paper/eight_cell_lines/new_8cell_lines_simulations_complete_results.pkl')

# Load the results from the pickle file
with open(r'/vf/users/HiTIF/data/keikhosravia2/ripley_paper/mcmc_results_gpu_3000/bayesian_cell_line_distance_results_final.pkl', 'rb') as f:
    loaded_results = pickle.load(f)

r0 = loaded_results[sys.argv[1]]['r0']
sigma = loaded_results[sys.argv[1]]['sigma']

eight_cell_lines = eight_cell_lines.loc[eight_cell_lines['cell_type']==sys.argv[1]]

eight_cell_lines = eight_cell_lines.loc[eight_cell_lines['spots_number'] > 5]
all_ripleys_dict = {}
mean_ripleys_dict = {}
for cell in eight_cell_lines.cell_type.unique():
    ripleys_list = []
    cell_type_specific_df = eight_cell_lines.loc[eight_cell_lines['cell_type'] == cell]
    for ind, row in cell_type_specific_df.iterrows():
        ripley_no_correction_cdf = row['ripley_no_correction'][0][0] / np.max(row['ripley_no_correction'][0][0])
        ripleys_list.append(ripley_no_correction_cdf)
    
    all_ripleys_dict[cell] = ripleys_list
    mean_ripleys_dict[cell] = np.mean(ripleys_list, axis=0)

# Define the analytical K-function for isotropic Gaussian


def RADIAL_DIST_CALC(xyz_round, spot_nuc_labels, radial_dist_df, dist_img):

    """
    Calculates the radial distance of each spot from the center of its respective nucleus.

    Parameters:
    -----------
    xyz_round : ndarray
        Rounded XYZ coordinates of detected spots.
    spot_nuc_labels : ndarray
        Labels indicating to which nucleus each spot belongs.
    radial_dist_df : DataFrame
        DataFrame containing radial distance information for nuclei.
    dist_img : ndarray
        Distance transform image of nuclei.

    Returns:
    --------
    ndarray
        Array containing the radial distance of each spot from the center of its nucleus.
    """
    radial_dist=[]
    eps=0.000001
    for i in range(xyz_round.__len__()):

        sp_dist = dist_img[xyz_round[i,0], xyz_round[i,1]]
        spot_lbl =int(spot_nuc_labels[i])
        if spot_lbl>0:
            cell_max = radial_dist_df.loc[radial_dist_df['label']==spot_lbl]['max_intensity'].iloc[0]
            sp_radial_dist= (cell_max-sp_dist)/(cell_max-1+eps)
        else:
            sp_radial_dist = np.nan
        radial_dist.append(sp_radial_dist)

    return np.array(radial_dist).astype(float)

def gaussian_k_function(r, lambda_, sigma):
    return (2 * np.pi * sigma**2 / lambda_) * (1 - np.exp(-r**2 / (2 * sigma**2)))

# Sample radii values
radii = np.linspace(0, 25, 1000)

fitting_dict = {key: {} for key in all_ripleys_dict.keys()}

# Loop over each cell type and fit the Gaussian K-function
for i, (cell_type, empirical_k_values) in enumerate(mean_ripleys_dict.items()):
    # Fit the analytical K-function to the empirical data
    initial_guess = [1e-3, 5]
    popt, _ = curve_fit(gaussian_k_function, radii, empirical_k_values, p0=initial_guess)
    
    # Extract fitted parameters
    lambda_fitted, sigma_fitted = popt

    # Calculate MSE (mean squared error)
    fitted_k_values = gaussian_k_function(radii, lambda_fitted, sigma_fitted)
    mse = mean_squared_error(empirical_k_values, fitted_k_values)
    mse_percentage = mse / np.mean(empirical_k_values) * 100

    # save the resutls in the dictionary
    
    fitting_dict[cell_type]['sigma_cdf'] = sigma_fitted
    fitting_dict[cell_type]['constant_cdf'] = lambda_fitted
    # Convert to microns
    lambda_fitted_micron = lambda_fitted 
    sigma_fitted_micron = sigma_fitted

# Function to perform the analysis after removing spots
def analyze_after_removal(points, num_remove, area, seed):
    np.random.seed(seed)
    np.random.shuffle(points)
    if num_remove > 0:
        reduced_points = points[:-num_remove]
    else:
        reduced_points = points
    
    metrics = {}

    metrics['ripley_k_score'] = analysis.ripley_k_score(reduced_points, area)
    G = analysis.create_graph_from_points(reduced_points)
    metrics['assortativity'] = analysis.calculate_assortativity(G)
    metrics['modularity'] = analysis.calculate_modularity(G)
    metrics['morans_i'] = analysis.morans_i(reduced_points)
    metrics['mean_nearest_neighbor_distance'] = analysis.mean_nearest_neighbor_distance(reduced_points)
    d_max = 25
    d_step = 25 / 1000
    metrics['pair_correlation_function'] = analysis.pair_correlation_function(reduced_points, d_max, d_step, area)
    metrics['dispersion_index'] = analysis.dispersion_index(reduced_points, area)

    return metrics

def estimate_min_distance(real_data):
    # Compute pairwise distances between all centromere locations
    pairwise_distances = np.linalg.norm(real_data[:, np.newaxis] - real_data, axis=2)
    
    # Exclude zero distances (distance between a point and itself)
    non_zero_distances = pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)]
    
    # Estimate min_distance (you could use median, minimum, or another statistic)
    min_distance = np.percentile(non_zero_distances, 10)  # 10th percentile distance as an example
    return min_distance



# eight_cell_lines['cell_based_gaussian_synth_spot_generator'] = [{} for _ in range(len(eight_cell_lines))]
# eight_cell_lines['poisson_disk_spot_generator'] = [{} for _ in range(len(eight_cell_lines))]
# eight_cell_lines['ripley_based_gaussian_synth_spot_generator'] = [{} for _ in range(len(eight_cell_lines))]
# eight_cell_lines['soft_core_spot_generator'] = [{} for _ in range(len(eight_cell_lines))]
# eight_cell_lines['uniform_spot_generator'] = [{} for _ in range(len(eight_cell_lines))]
# eight_cell_lines['radial_gaussian_spot_generator'] = [{} for _ in range(len(eight_cell_lines))]

eight_cell_lines['diastance_based_radial_gaussian_spot_generator_3000'] = [{} for _ in range(len(eight_cell_lines))]


pixpermic = 0.108

def process_row(row):
    seed = np.random.randint(0, 10000, size=1)
    mask_path = row['masks_path'].replace('/data/krishnendu/ripley_paper/8cell_lines_patches/mask/', '/vf/users/HiTIF/data/keikhosravia2/ripley_paper/eight_cell_lines/new_mask/mask/')

    nuc_mask = imread(mask_path)
    labeled_nuc, number_nuc = label(nuc_mask > 100)
    dist_img = distance_transform_edt(nuc_mask)
    dist_props = regionprops_table(labeled_nuc, dist_img, properties=('label', 'max_intensity'))
    radial_dist_df = pd.DataFrame(dist_props)
    
    
    num_spots = row.spots_number

    # Calculate the region properties
    props = regionprops(labeled_nuc)
    orientation = major_axis_length = minor_axis_length = None
    for region in props:
        orientation = region.orientation
        major_axis_length = region.major_axis_length * pixpermic
        minor_axis_length = region.minor_axis_length * pixpermic
    
    result = {
        'index': row.name,
        'orientation': orientation,
        'major_axis_length': major_axis_length,
        'minor_axis_length': minor_axis_length,
        # 'cell_based_gaussian_synth_spot_generator': {},
        # 'ripley_based_gaussian_synth_spot_generator': {},
        # 'uniform_spot_generator': {},
        # 'poisson_disk_spot_generator': {},
        # 'soft_core_spot_generator': {},
        # 'radial_gaussian_spot_generator': {},
        'diastance_based_radial_gaussian_spot_generator_3000':{}
    }
    
    # # cell_based_gaussian_synth_spot_generator
    # seed = np.random.randint(0, 10000, size=1)
    # all_coordinates, synth_spots = SpotGenerator.cell_based_gaussian_synth_spot_generator(num_spots, nuc_mask, orientation, major_axis_length/pixpermic, 
    #                                                                                       minor_axis_length/pixpermic, gauss_kernel_size=2)
    
    # xyz_round = np.floor(all_coordinates).astype('int')
    # spot_nuc_labels = labeled_nuc[xyz_round[:, 0], xyz_round[:, 1]]
    # result['cell_based_gaussian_synth_spot_generator']['radial_distances'] = RADIAL_DIST_CALC(xyz_round, spot_nuc_labels, radial_dist_df, dist_img)
    
    # all_coordinates = all_coordinates * pixpermic
    # result['cell_based_gaussian_synth_spot_generator']['coords'] = all_coordinates
    # result['cell_based_gaussian_synth_spot_generator']['patch'] = synth_spots
    # result['cell_based_gaussian_synth_spot_generator']['metrics'] = analyze_after_removal(all_coordinates, 0, row.area, seed)
    
    
    # # ripley_based_gaussian_synth_spot_generator
    # seed = np.random.randint(0, 10000, size=1)
    # cov_matrix = np.eye(2) * (fitting_dict[row.cell_type]['sigma_cdf']**2)
    # all_coordinates, synth_spots = SpotGenerator.ripley_based_gaussian_synth_spot_generator(num_spots, nuc_mask, cov_matrix, gauss_kernel_size=2)
    
    # xyz_round = np.floor(all_coordinates).astype('int')
    # spot_nuc_labels = labeled_nuc[xyz_round[:, 0], xyz_round[:, 1]]
    # result['ripley_based_gaussian_synth_spot_generator']['radial_distances'] = RADIAL_DIST_CALC(xyz_round, spot_nuc_labels, radial_dist_df, dist_img)
    
    # all_coordinates = all_coordinates * pixpermic
    # result['ripley_based_gaussian_synth_spot_generator']['coords'] = all_coordinates
    # result['ripley_based_gaussian_synth_spot_generator']['patch'] = synth_spots
    # result['ripley_based_gaussian_synth_spot_generator']['metrics'] = analyze_after_removal(all_coordinates, 0, row.area, seed)
    
    # # uniform_spot_generator
    # seed = np.random.randint(0, 10000, size=1)
    # all_coordinates, synth_spots = SpotGenerator.uniform_spot_generator(num_spots, nuc_mask, gauss_kernel_size=2)
    
    # xyz_round = np.floor(all_coordinates).astype('int')
    # spot_nuc_labels = labeled_nuc[xyz_round[:, 0], xyz_round[:, 1]]
    # result['uniform_spot_generator']['radial_distances'] = RADIAL_DIST_CALC(xyz_round, spot_nuc_labels, radial_dist_df, dist_img)
    
    # all_coordinates = all_coordinates * pixpermic
    # result['uniform_spot_generator']['coords'] = all_coordinates
    # result['uniform_spot_generator']['patch'] = synth_spots
    # result['uniform_spot_generator']['metrics'] = analyze_after_removal(all_coordinates, 0, row.area, seed)
    
    # # poisson_disk_spot_generator
    
    # # Assuming your real centromere locations are stored in 'real_data'
    # min_distance = estimate_min_distance(row.spot_coordinates[0][0])
    
    # seed = np.random.randint(0, 10000, size=1)
    # all_coordinates, synth_spots = SpotGenerator.poisson_disk_spot_generator(num_spots, nuc_mask, gauss_kernel_size=2, min_dist=min_distance)
    
    # xyz_round = np.floor(all_coordinates).astype('int')
    # spot_nuc_labels = labeled_nuc[xyz_round[:, 0], xyz_round[:, 1]]
    # result['poisson_disk_spot_generator']['radial_distances'] = RADIAL_DIST_CALC(xyz_round, spot_nuc_labels, radial_dist_df, dist_img)
    
    # all_coordinates = all_coordinates * pixpermic
    # result['poisson_disk_spot_generator']['coords'] = all_coordinates
    # result['poisson_disk_spot_generator']['patch'] = synth_spots
    # result['poisson_disk_spot_generator']['metrics'] = analyze_after_removal(all_coordinates, 0, row.area, seed)
    
    # # soft_core_spot_generator
    # seed = np.random.randint(0, 10000, size=1)
    # all_coordinates, synth_spots = SpotGenerator.soft_core_spot_generator(num_spots, nuc_mask, gauss_kernel_size=2, min_distance=min_distance, repulsion_strength=1)
    
    # xyz_round = np.floor(all_coordinates).astype('int')
    # spot_nuc_labels = labeled_nuc[xyz_round[:, 0], xyz_round[:, 1]]
    # result['soft_core_spot_generator']['radial_distances'] = RADIAL_DIST_CALC(xyz_round, spot_nuc_labels, radial_dist_df, dist_img)
    
    # all_coordinates = all_coordinates * pixpermic
    # result['soft_core_spot_generator']['coords'] = all_coordinates
    # result['soft_core_spot_generator']['patch'] = synth_spots
    # result['soft_core_spot_generator']['metrics'] = analyze_after_removal(all_coordinates, 0, row.area, seed)

    # # Bayesian Radially shifted
    # all_coordinates, synth_spots = SpotGenerator.bayesian_radial_gaussian_synth_spot_generator(num_spots, nuc_mask, r0, sigma, gauss_kernel_size=2)
    
    # xyz_round = np.floor(all_coordinates).astype('int')
    # spot_nuc_labels = labeled_nuc[xyz_round[:, 0], xyz_round[:, 1]]
    # result['radial_gaussian_spot_generator']['radial_distances'] = RADIAL_DIST_CALC(xyz_round, spot_nuc_labels, radial_dist_df, dist_img)
    
    # all_coordinates = all_coordinates * pixpermic
    # result['radial_gaussian_spot_generator']['coords'] = all_coordinates
    # result['radial_gaussian_spot_generator']['patch'] = synth_spots
    # result['radial_gaussian_spot_generator']['metrics'] = analyze_after_removal(all_coordinates, 0, row.area, seed)

    # Distance Based Bayesian Radially shifted
    all_coordinates, synth_spots = SpotGenerator.bayesian_radial_gaussian_synth_spot_generator(num_spots, nuc_mask, r0, sigma, gauss_kernel_size=2)
    
    xyz_round = np.floor(all_coordinates).astype('int')
    spot_nuc_labels = labeled_nuc[xyz_round[:, 0], xyz_round[:, 1]]
    result['diastance_based_radial_gaussian_spot_generator_3000']['radial_distances'] = RADIAL_DIST_CALC(xyz_round, spot_nuc_labels, radial_dist_df, dist_img)
    
    all_coordinates = all_coordinates * pixpermic
    result['diastance_based_radial_gaussian_spot_generator_3000']['coords'] = all_coordinates
    result['diastance_based_radial_gaussian_spot_generator_3000']['patch'] = synth_spots
    result['diastance_based_radial_gaussian_spot_generator_3000']['metrics'] = analyze_after_removal(all_coordinates, 0, row.area, seed)


    return result

def update_dataframe(result):
    ind = result['index']
    eight_cell_lines.loc[ind, 'orientation'] = result['orientation']
    eight_cell_lines.loc[ind, 'major_axis_length'] = result['major_axis_length']
    eight_cell_lines.loc[ind, 'minor_axis_length'] = result['minor_axis_length']
    for method in ['diastance_based_radial_gaussian_spot_generator_3000']:
    # for method in ['radial_gaussian_spot_generator', 'cell_based_gaussian_synth_spot_generator', 'ripley_based_gaussian_synth_spot_generator', 'uniform_spot_generator', 'poisson_disk_spot_generator', 'soft_core_spot_generator', 'diastance_based_radial_gaussian_spot_generator']:
        eight_cell_lines.at[ind, method] = result[method]
    
    

if __name__ == '__main__':
    num_cores = 48
    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_row, [row for _, row in eight_cell_lines.iterrows()]), total=len(eight_cell_lines), desc="Processing Rows"))
    
    for result in tqdm(results, desc="Updating DataFrame"):
        update_dataframe(result)
    
    file_name = str(sys.argv[1]) +'_simulations_complete_results.pkl'
    full_name = os.path.join('/vf/users/HiTIF/data/keikhosravia2/ripley_paper/eight_cell_lines/latest_complete_simulations/' + file_name)
    eight_cell_lines.to_pickle(full_name)
