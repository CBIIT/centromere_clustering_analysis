import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
import community as community_louvain
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN, MeanShift, 
                             SpectralClustering, AffinityPropagation, Birch, OPTICS)
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
import warnings
from astropy.stats import RipleysKEstimator
from scipy.spatial import Voronoi
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist

class ClusteringAnalysis:
    """
    A class for analyzing clustering of centromeres in fluorescence imaging.

    Methods
    -------
    ripley_k_score(points, area=None):
        Computes the Ripley's K function for spatial clustering.
    create_graph_from_points(points, k=10):
        Creates a k-nearest neighbor graph from points.
    calculate_assortativity(G):
        Computes the assortativity coefficient of the graph.
    calculate_modularity(G):
        Computes the modularity of the graph using the Louvain method.
    morans_i(points):
        Computes Moran's I statistic for spatial autocorrelation.
    mean_nearest_neighbor_distance(points):
        Computes the mean nearest neighbor distance.
    generate_random_points(num_points, area):
        Generates random points within a given area.
    pair_correlation_function(points, d_max, d_step, area, num_random_samples=100):
        Computes the pair correlation function g(r).
    voronoi_tessellation(points):
        Creates a Voronoi tessellation and analyzes cell areas.
    spot_density(points, area):
        Calculates the density of detected spots.
    local_clustering_coefficient(G):
        Computes the local clustering coefficient for each node.
    hopkins_statistic(points, n=100):
        Assesses the clustering tendency of the points.
    edge_density(G):
        Computes the density of edges in the k-nearest neighbor graph.
    cluster_compactness(points, labels):
        Measures the compactness of clusters.
    normalized_cross_correlation(points1, points2):
        Computes the cross-correlation of spatial patterns.
    dispersion_index(points, area):
        Quantifies the level of dispersion or clustering.
    distribution_of_pairwise_distances(points):
        Analyzes the distribution of pairwise distances between spots.
    compare_clustering_algorithms(points, metric='silhouette', max_clusters=10):
        Compares different clustering algorithms using various metrics.
    """
    
    def __init__(self):
        """Initializes the ClusteringAnalysis class with a set of clustering algorithms."""
        self.clustering_algorithms = {
            'KMeans': KMeans,
            'Agglomerative': AgglomerativeClustering,
            'DBSCAN': DBSCAN,
            'MeanShift': MeanShift,
            'Spectral': SpectralClustering,
            'AffinityPropagation': lambda: AffinityPropagation(random_state=0),
            'Birch': Birch,
            'OPTICS': OPTICS,
            'GMM': GaussianMixture
        }
    
    def ripley_k_score(self, points, area=None):
        """
        Computes the Ripley's K function for spatial clustering.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.
        area : float, optional
            The area of the region containing the points.

        Returns
        -------
        clustering_perc_no_cor : float
            Percentage of clustering without correction.
        data_sig_no_correction : ndarray
            Ripley's K values without correction.
        clustering_perc_r_cor : float
            Percentage of clustering with Ripley correction.
        data_sig_ripley_correction : ndarray
            Ripley's K values with Ripley correction.
        """
        dist_mat = np.zeros((points.shape[0], points.shape[0]), dtype=float)
        for loc1 in range(points.shape[0]):
            for loc2 in range(points.shape[0]):
                dist_mat[loc1, loc2] = np.linalg.norm(points[loc1] - points[loc2])
                
        max_dist = dist_mat.max()+0.01
        max_radius = 25
        # max_radius=40
        Kest = RipleysKEstimator(area=area)
        radii_vals = np.linspace(0, max_radius, 1000)
        
        ### No correction Ripley
        data_sig_no_correction = Kest(data=points, radii=radii_vals, mode='none')
        diff_sig = data_sig_no_correction - Kest.poisson(radii_vals)
        diff_sig = diff_sig[:int(max_dist/max_radius*1000)]
        clustering_perc_no_cor = 100 * np.sum(diff_sig > 0) / len(diff_sig)
        ripley_mse = np.mean(diff_sig**2)
        relative_ripley_mse = ripley_mse/np.mean(data_sig_no_correction ** 2)

        ### Ripley correction Ripley
        Kest = RipleysKEstimator(area=area, x_max=points.max(axis=0)[0], y_max=points.max(axis=0)[1],
                                            x_min=points.min(axis=0)[0], y_min=points.min(axis=0)[1])

        # data_sig_ripley_correction = Kest(data=points, radii=radii_vals, mode='ripley')
        # diff_sig = data_sig_ripley_correction - Kest.poisson(radii_vals)
        # clustering_perc_r_cor = 100 * np.sum(diff_sig > 0) / len(diff_sig)
        
        return clustering_perc_no_cor, data_sig_no_correction, ripley_mse
        
    def create_graph_from_points(self, points, k=10):
        """
        Creates a k-nearest neighbor graph from points.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.
        k : int, optional
            The number of nearest neighbors to consider.

        Returns
        -------
        G : networkx.Graph
            A k-nearest neighbor graph.
        """
        k = max(len(points) - 1, 1)  # Ensure k does not exceed the number of points and is at least 1

        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, indices = nbrs.kneighbors(points)

        G = nx.Graph()
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:
                    distance = distances[i][np.where(indices[i] == j)][0]
                    weight = 1.0 / distance**2 if distance != 0 else 0  # Handle division by zero
                    G.add_edge(i, j, weight=weight)
        return G



    def calculate_assortativity(self, G):
        """
        Computes the assortativity coefficient of the graph.

        Parameters
        ----------
        G : networkx.Graph
            A graph.

        Returns
        -------
        assortativity : float
            Assortativity coefficient of the graph.
        """
        return nx.degree_assortativity_coefficient(G)

    def calculate_modularity(self, G):
        """
        Computes the modularity of the graph using the Louvain method.

        Parameters
        ----------
        G : networkx.Graph
            A graph.

        Returns
        -------
        modularity : float
            Modularity of the graph.
        """
        partition = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(partition, G)
        return modularity

    def morans_i(self, points):
        """
        Computes Moran's I statistic for spatial autocorrelation.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.

        Returns
        -------
        morans_i_value : float
            Moran's I statistic value.
        """
        distances = np.sqrt((points[:, np.newaxis, 0] - points[np.newaxis, :, 0])**2 + 
                            (points[:, np.newaxis, 1] - points[np.newaxis, :, 1])**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = np.where(distances > 0, 1 / distances**2, 0)
        np.fill_diagonal(weights, 0)  # Explicitly set diagonal to zero
        n = len(points)
        mean_x = np.mean(points[:, 0])
        mean_y = np.mean(points[:, 1])
        
        deviations_x = points[:, 0] - mean_x
        deviations_y = points[:, 1] - mean_y
        
        num = 0
        for i in range(n):
            for j in range(n):
                num += weights[i, j] * deviations_x[i] * deviations_x[j] + weights[i, j] * deviations_y[i] * deviations_y[j]
        
        den = np.sum(deviations_x**2 + deviations_y**2)
        sum_weights = np.sum(weights)
        
        morans_i_value = (n / sum_weights) * (num / den)
        return morans_i_value

    def mean_nearest_neighbor_distance(self, points):
        """
        Computes the mean nearest neighbor distance.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.

        Returns
        -------
        mnnd : float
            Mean nearest neighbor distance.
        """
        dist_matrix = distance_matrix(points, points)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_distances = np.min(dist_matrix, axis=1)
        mnnd = np.mean(nearest_distances)
        return mnnd

    def generate_random_points(self, num_points, area):
        """
        Generates random points within a given area.

        Parameters
        ----------
        num_points : int
            The number of random points to generate.
        area : float
            The area within which to generate the points.

        Returns
        -------
        random_points : ndarray
            An array of random points.
        """
        width, height = np.sqrt(area), np.sqrt(area)
        return np.column_stack((np.random.uniform(0, width, num_points), np.random.uniform(0, height, num_points)))

    def pair_correlation_function(self, points, d_max, d_step, area, num_random_samples=100):
        """
        Computes the pair correlation function g(r).

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.
        d_max : float
            The maximum distance for the pair correlation function.
        d_step : float
            The step size for the distance.
        area : float
            The area within which the points are contained.
        num_random_samples : int, optional
            The number of random samples for comparison.

        Returns
        -------
        d : ndarray
            Array of distances.
        g_values : ndarray
            Pair correlation function values for the observed data.
        g_mean : ndarray
            Mean pair correlation function values for the random samples.
        g_std : ndarray
            Standard deviation of the pair correlation function values for the random samples.
        """
        def k_function(points, d):
            n = len(points)
            density = n / area
            k_values = []

            for dist in d:
                count = 0
                for i in range(n):
                    distances = np.sqrt(np.sum((points - points[i])**2, axis=1))
                    count += np.sum(distances <= dist) - 1  # subtracting self-distance
                k_values.append(count / (n * density))

            return np.array(k_values)

        d = np.arange(0, d_max, d_step)
        k_values = k_function(points, d)

        # Avoid division by zero
        valid_indices = np.where((2 * np.pi * d[:-1] * d_step) != 0)

        g_values = np.diff(k_values)[valid_indices] / (2 * np.pi * d[:-1][valid_indices] * d_step)

        random_g_values = []
        for _ in range(num_random_samples):
            random_points = self.generate_random_points(len(points), area)
            random_k_values = k_function(random_points, d)
            random_g = np.diff(random_k_values)[valid_indices] / (2 * np.pi * d[:-1][valid_indices] * d_step)
            random_g_values.append(random_g)

        random_g_values = np.array(random_g_values)
        g_mean = np.mean(random_g_values, axis=0)
        g_std = np.std(random_g_values, axis=0)

        return d[:-1][valid_indices], g_values, g_mean, g_std


        d = np.arange(0, d_max, d_step)
        k_values = k_function(points, d)
        g_values = np.diff(k_values) / (2 * np.pi * d[:-1] * d_step)

        random_g_values = []
        for _ in range(num_random_samples):
            random_points = self.generate_random_points(len(points), area)
            random_k_values = k_function(random_points, d)
            random_g = np.diff(random_k_values) / (2 * np.pi * d[:-1] * d_step)
            random_g_values.append(random_g)

        random_g_values = np.array(random_g_values)
        g_mean = np.mean(random_g_values, axis=0)
        g_std = np.std(random_g_values, axis=0)

        return d[:-1], g_values, g_mean, g_std

    def voronoi_tessellation(self, points):
        """
        Creates a Voronoi tessellation and analyzes cell areas.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.

        Returns
        -------
        areas : list
            List of Voronoi cell areas.
        """
        vor = Voronoi(points)
        regions, vertices = vor.regions, vor.vertices
        areas = []
        for region in regions:
            if not -1 in region:
                polygon = vertices[region]
                area = 0.5 * np.abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
                areas.append(area)
        return areas

    def spot_density(self, points, area):
        """
        Calculates the density of detected spots.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.
        area : float
            The area within which the points are contained.

        Returns
        -------
        density : float
            Density of detected spots.
        """
        num_spots = len(points)
        density = num_spots / area
        return density

    def local_clustering_coefficient(self, G):
        """
        Computes the local clustering coefficient for each node in the graph.

        Parameters
        ----------
        G : networkx.Graph
            A graph.

        Returns
        -------
        clustering_coeffs : dict
            Dictionary of nodes and their local clustering coefficients.
        """
        return nx.clustering(G)

    def hopkins_statistic(self, points, n=46):
        """
        Assesses the clustering tendency of the points.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.
        n : int, optional
            The number of samples to draw.

        Returns
        -------
        hopkins_stat : float
            Hopkins statistic value.
        """
        d = points.shape[1]
        x = np.random.choice(points.shape[0], n)
        dist_matrix = distance_matrix(points, points)
        ujd = np.sum(np.min(dist_matrix[x, :], axis=1))
        y = np.random.rand(n, d) * np.ptp(points, axis=0) + np.min(points, axis=0)
        wjd = np.sum(np.min(distance_matrix(y, points), axis=1))
        return ujd / (ujd + wjd)

    def edge_density(self, G):
        """
        Computes the density of edges in the k-nearest neighbor graph.

        Parameters
        ----------
        G : networkx.Graph
            A graph.

        Returns
        -------
        density : float
            Edge density of the graph.
        """
        return nx.density(G)

    def cluster_compactness(self, points, labels):
        """
        Measures the compactness of clusters.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.
        labels : ndarray
            Cluster labels for each point.

        Returns
        -------
        compactness : float
            Average compactness score of the clusters.
        """
        unique_labels = np.unique(labels)
        compactness = []
        for label in unique_labels:
            cluster_points = points[labels == label]
            if len(cluster_points) > 1:
                compactness.append(np.mean(pairwise_distances(cluster_points)))
        return np.mean(compactness) if compactness else 0

    def normalized_cross_correlation(self, points1, points2):
        """
        Computes the cross-correlation of spatial patterns between two sets of points.

        Parameters
        ----------
        points1 : ndarray
            First set of points.
        points2 : ndarray
            Second set of points.

        Returns
        -------
        cross_correlation : float
            Normalized cross-correlation value.
        """
        mean1, mean2 = np.mean(points1, axis=0), np.mean(points2, axis=0)
        diff1, diff2 = points1 - mean1, points2 - mean2
        cov = np.mean(diff1 * diff2)
        std1, std2 = np.std(points1), np.std(points2)
        return cov / (std1 * std2)

    def dispersion_index(self, points, area):
        """
        Quantifies the level of dispersion or clustering.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.
        area : float
            The area within which the points are contained.

        Returns
        -------
        dispersion_index : float
            Dispersion index value.
        """
        n = len(points)
        density = n / area
        mean_dist = np.mean(pdist(points))
        var_dist = np.var(pdist(points))
        return var_dist / mean_dist

    def distribution_of_pairwise_distances(self, points):
        """
        Analyzes the distribution of pairwise distances between spots.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.

        Returns
        -------
        distances : ndarray
            Condensed distance matrix of pairwise distances.
        """
        return pdist(points)

    def compare_clustering_algorithms(self, points, metric='silhouette', max_clusters=10):
        """
        Compares different clustering algorithms using various metrics.

        Parameters
        ----------
        points : ndarray
            An array of points representing centromeres.
        metric : str, optional
            The metric to use for comparison ('silhouette', 'chi', or 'dbi').
        max_clusters : int, optional
            The maximum number of clusters to consider.

        Returns
        -------
        scores : ndarray
            Matrix of scores for each algorithm and number of clusters.
        best_score : float
            Best score achieved.
        best_method : str
            Name of the best clustering algorithm.
        best_num_clusters : int
            Number of clusters for the best score.
        """
        scores = np.zeros((len(self.clustering_algorithms), max_clusters - 1))
        algorithm_names = list(self.clustering_algorithms.keys())
        best_score = -1 if metric == 'silhouette' else float('inf')
        best_method = None
        best_num_clusters = None

        for i, (name, algorithm) in enumerate(self.clustering_algorithms.items()):
            for n_clusters in range(2, max_clusters + 1):
                if name in ['DBSCAN', 'MeanShift', 'AffinityPropagation', 'OPTICS']:
                    if name == 'DBSCAN':
                        best_metric_score = -1 if metric == 'silhouette' else float('inf')
                        best_eps = None
                        for eps in np.arange(0.1, 1.1, 0.1):
                            model = algorithm(eps=eps)
                            labels = model.fit_predict(points)
                            if len(np.unique(labels)) > 1:
                                if metric == 'silhouette':
                                    score = silhouette_score(points, labels)
                                    if score > best_metric_score:
                                        best_metric_score = score
                                        best_eps = eps
                                elif metric == 'chi':
                                    score = calinski_harabasz_score(points, labels)
                                    if score > best_metric_score:
                                        best_metric_score = score
                                        best_eps = eps
                                elif metric == 'dbi':
                                    score = davies_bouldin_score(points, labels)
                                    if score < best_metric_score:
                                        best_metric_score = score
                                        best_eps = eps
                        score = best_metric_score
                    else:
                        model = algorithm()
                        labels = model.fit_predict(points)
                        if len(np.unique(labels)) > 1:
                            if metric == 'silhouette':
                                score = silhouette_score(points, labels)
                            elif metric == 'chi':
                                score = calinski_harabasz_score(points, labels)
                            elif metric == 'dbi':
                                score = davies_bouldin_score(points, labels)
                        else:
                            score = -1 if metric == 'silhouette' else float('inf')
                elif name == 'Birch':
                    best_metric_score = -1 if metric == 'silhouette' else float('inf')
                    best_threshold = None
                    for threshold in np.arange(0.1, 1.1, 0.1):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=ConvergenceWarning)
                            model = algorithm(threshold=threshold)
                            labels = model.fit_predict(points)
                            if len(np.unique(labels)) > 1:
                                if metric == 'silhouette':
                                    score = silhouette_score(points, labels)
                                    if score > best_metric_score:
                                        best_metric_score = score
                                        best_threshold = threshold
                                elif metric == 'chi':
                                    score = calinski_harabasz_score(points, labels)
                                    if score > best_metric_score:
                                        best_metric_score = score
                                        best_threshold = threshold
                                elif metric == 'dbi':
                                    score = davies_bouldin_score(points, labels)
                                    if score < best_metric_score:
                                        best_metric_score = score
                                        best_threshold = threshold
                    score = best_metric_score
                elif name == 'GMM':
                    model = algorithm(n_components=n_clusters)
                    labels = model.fit(points).predict(points)
                    if metric == 'silhouette':
                        score = silhouette_score(points, labels)
                    elif metric == 'chi':
                        score = calinski_harabasz_score(points, labels)
                    elif metric == 'dbi':
                        score = davies_bouldin_score(points, labels)
                else:
                    model = algorithm(n_clusters=n_clusters)
                    labels = model.fit_predict(points)
                    if len(np.unique(labels)) > 1:
                        if metric == 'silhouette':
                            score = silhouette_score(points, labels)
                        elif metric == 'chi':
                            score = calinski_harabasz_score(points, labels)
                        elif metric == 'dbi':
                            score = davies_bouldin_score(points, labels)
                    else:
                        score = -1 if metric == 'silhouette' else float('inf')
                
                scores[i, n_clusters - 2] = score

                if (metric == 'silhouette' and score > best_score) or (metric == 'chi' and score > best_score) or (metric == 'dbi' and score < best_score):
                    best_score = score
                    best_method = name
                    best_num_clusters = n_clusters
                    cluster_compactness = self.cluster_compactness(points, labels)

        return scores, best_score, best_method, best_num_clusters, cluster_compactness

