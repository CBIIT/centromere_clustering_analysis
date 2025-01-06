import numpy as np
import pickle
import pandas as pd
import os
from spot_pattern_generator_functions import SpotGenerator
from skimage.io import imread, imsave
from skimage.measure import regionprops, label
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from clustering import ClusteringAnalysis
from tqdm import tqdm  # Import tqdm for progress tracking
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module='astropy.stats.spatial')

eight_cell_lines = pd.read_pickle('/vf/users/HiTIF/data/keikhosravia2/ripley_paper/eight_cell_lines/8cell_lines_complete_results.pkl')
eight_cell_lines = eight_cell_lines.loc[eight_cell_lines['cell_type']==sys.argv[1]]

# Function to perform the analysis after removing spots
def analyze_after_removal(points, num_remove, area, seed):
    np.random.seed(seed)
    np.random.shuffle(points)
    if num_remove > 0:
        reduced_points = points[:-num_remove]
    else:
        reduced_points = points
    analysis = ClusteringAnalysis()
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


eight_cell_lines['real_data_spots'] = [{} for _ in range(len(eight_cell_lines))]

def process_row(row):
    seed = np.random.randint(0, 10000, size=1)

    mask_path = row['masks_path'].replace( r'/data/krishnendu/ripley_paper/8cell_lines_patches/mask/', 
                                            r'/vf/users/HiTIF/data/keikhosravia2/ripley_paper/eight_cell_lines/new_mask/mask/')

    nuc_mask = imread(mask_path)
    labeled_nuc_mask = label(nuc_mask > 100)
    num_spots = row.spots_number

    result = {'index': row.name, 'real_data_spots': {} }
    all_coordinates = row['spot_coordinates'][0][0]
    result['real_data_spots']['coords'] = all_coordinates
    # result['cell_based_gaussian_synth_spot_generator']['patch'] = synth_spots
    result['real_data_spots']['metrics'] = analyze_after_removal(all_coordinates, 0, row.area, seed)
    
    return result

def update_dataframe(result):
    ind = result['index']
    for method in ['real_data_spots']:
        eight_cell_lines.at[ind, method] = result[method]
    
if __name__ == '__main__':
    num_cores = 48
    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_row, [row for _, row in eight_cell_lines.iterrows()]), total=len(eight_cell_lines), desc="Processing Rows"))
    
    for result in tqdm(results, desc="Updating DataFrame"):
        update_dataframe(result)
    file_name = str(sys.argv[1]) +'_real_only_complete_results.pkl'
    full_name = os.path.join('/vf/users/HiTIF/data/keikhosravia2/ripley_paper/eight_cell_lines/' + file_name)
    eight_cell_lines.to_pickle(full_name)
