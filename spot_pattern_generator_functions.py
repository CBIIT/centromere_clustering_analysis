import numpy as np
from skimage.filters import gaussian
from sklearn.mixture import GaussianMixture
import random
from skimage import exposure
pixpermic = 0.108
class SpotGenerator:
    """
    A class to generate synthetic spot patterns using various spatial point processes.

    Methods
    -------
    cell_based_gaussian_synth_spot_generator(num_spots, cell_mask_img, orientation, major_axis_length, minor_axis_length, gauss_kernel_size=2):
        Generates spots based on a Gaussian distribution with cell-specific parameters.
    
    ripley_based_gaussian_synth_spot_generator(num_spots, cell_mask_img, covariance_matrix, gauss_kernel_size=2):
        Generates spots based on a Gaussian distribution using a precomputed covariance matrix.
    
    uniform_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2):
        Generates spots uniformly distributed within a given cell mask.
    
    poisson_disk_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, min_dist):
        Generates spots based on a Poisson Disk sampling method, ensuring a minimum distance between spots.
    
    neyman_scott_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, num_parents=10, cluster_std=5):
        Generates spots using the Neyman-Scott clustering process.
    
    matern_cluster_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, num_parents=10, cluster_radius=10):
        Generates spots using the Matérn clustering process.
    
    hard_core_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, min_distance=10):
        Generates spots based on the Hard-Core Process with a minimum distance constraint.
    
    soft_core_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, min_distance=10, repulsion_strength=2):
        Generates spots using the Soft-Core Process, with a probabilistic repulsion effect.
    
    strauss_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, interaction_radius=10, inhibition_strength=0.5):
        Generates spots based on the Strauss Process, incorporating both clustering and inhibition.
    
    thomas_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, num_parents=10, cluster_std=5):
        Generates spots using the Thomas Process, a Gaussian clustering process.
    """

    @staticmethod
    def cell_based_gaussian_synth_spot_generator(num_spots, cell_mask_img, orientation, major_axis_length, minor_axis_length, gauss_kernel_size=2):
        """
        Generates synthetic spots based on a Gaussian distribution with cell-specific parameters.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        orientation : float
            Orientation of the Gaussian ellipse.
        major_axis_length : float
            Length of the major axis of the Gaussian ellipse.
        minor_axis_length : float
            Length of the minor axis of the Gaussian ellipse.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        def create_covariance_matrix(orientation, major_axis_length, minor_axis_length):
            cos_angle = np.cos(orientation)
            sin_angle = np.sin(orientation)
            rotation_matrix = np.array([[cos_angle, -sin_angle],
                                        [sin_angle, cos_angle]])
            scaling_matrix = np.array([[major_axis_length**2, 0],
                                       [0, minor_axis_length**2]])
            covariance_matrix = rotation_matrix @ scaling_matrix @ rotation_matrix.T
            return covariance_matrix

        all_coordinates = np.zeros((0, 2))
        nuc_mask = cell_mask_img
        synth_spots = np.zeros((256, 256))
        mean = np.array([128, 128])
        covariance = create_covariance_matrix(orientation, major_axis_length, minor_axis_length)

        while all_coordinates.shape[0] < num_spots:
            xy_coor = np.random.multivariate_normal(mean, covariance).astype(int)
            if 0 <= xy_coor[0] < 256 and 0 <= xy_coor[1] < 256:
                mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
                if mask_value > 0:
                    synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                    all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0).astype(int)

        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("float")
        synth_spots = exposure.rescale_intensity(synth_spots, out_range=(0, 255)).astype("uint8")

        return all_coordinates, synth_spots
    
    
    @staticmethod
    def bayesian_radial_gaussian_synth_spot_generator(num_spots, cell_mask_img, r0, sigma, gauss_kernel_size=2):
        """
        Generates synthetic spots based on a radially shifted Gaussian distribution using fitted r0 and sigma values.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        r0 : float
            Mean radius for the radially shifted Gaussian distribution.
        sigma : float
            Standard deviation for the radially shifted Gaussian distribution.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))  # Initialize an empty array to store coordinates
        nuc_mask = cell_mask_img  # Nucleus mask (binary image)
        synth_spots = np.zeros_like(cell_mask_img)  # Initialize an empty image for synthetic spots
        center = np.array([nuc_mask.shape[0] // 2, nuc_mask.shape[1] // 2])  # Assuming the nucleus is centered

        # Generate num_spots points in polar coordinates, then convert to Cartesian
        while all_coordinates.shape[0] < num_spots:
            # Generate random angle theta in [0, 2pi]
            theta = np.random.uniform(0, 2 * np.pi)

            # Generate radial distance r based on a Gaussian with mean r0 and standard deviation sigma
            r = np.random.normal(r0, np.sqrt(2)*sigma)

            # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
            x = center[0] + r * np.cos(theta)/pixpermic
            y = center[1] + r * np.sin(theta)/pixpermic

            # Round and convert to integer coordinates
            xy_coor = np.array([x, y])

            # Check if the generated coordinate is within the bounds of the image and inside the mask
            if r>0 and 0 <= xy_coor[0] < nuc_mask.shape[0] and 0 <= xy_coor[1] < nuc_mask.shape[1]:
                mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
                if mask_value > 0:  # Make sure the point is within the nucleus mask
                    synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                    all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0).astype(int)

        # Apply Gaussian smoothing to the synthetic spots
        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("float")

        # Rescale the intensity to 8-bit range (0-255)
        synth_spots = exposure.rescale_intensity(synth_spots, out_range=(0, 255)).astype("uint8")

        return all_coordinates, synth_spots

    @staticmethod
    def bayesian_radial_weibull_synth_spot_generator(num_spots, cell_mask_img, lambda_, k, gauss_kernel_size=2):
        """
        Generates synthetic spots based on a radially shifted Weibull distribution using fitted lambda and k values.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        lambda_ : float
            Scale parameter for the Weibull distribution.
        k : float
            Shape parameter for the Weibull distribution.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))  # Initialize an empty array to store coordinates
        nuc_mask = cell_mask_img  # Nucleus mask (binary image)
        synth_spots = np.zeros_like(cell_mask_img)  # Initialize an empty image for synthetic spots
        center = np.array([nuc_mask.shape[0] // 2, nuc_mask.shape[1] // 2])  # Assuming the nucleus is centered

        # Generate num_spots points in polar coordinates, then convert to Cartesian
        while all_coordinates.shape[0] < num_spots:
            # Generate random angle theta in [0, 2pi]
            theta = np.random.uniform(0, 2 * np.pi)

            # Generate radial distance r based on a Weibull distribution
            r = np.random.weibull(k) * lambda_

            # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
            x = center[0] + r * np.cos(theta) / pixpermic
            y = center[1] + r * np.sin(theta) / pixpermic

            # Round and convert to integer coordinates
            xy_coor = np.array([x, y])

            # Check if the generated coordinate is within the bounds of the image and inside the mask
            if 0 <= xy_coor[0] < nuc_mask.shape[0] and 0 <= xy_coor[1] < nuc_mask.shape[1]:
                mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
                if mask_value > 0:  # Make sure the point is within the nucleus mask
                    synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                    all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0).astype(int)

        # Apply Gaussian smoothing to the synthetic spots
        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("float")

        # Rescale the intensity to 8-bit range (0-255)
        synth_spots = exposure.rescale_intensity(synth_spots, out_range=(0, 255)).astype("uint8")

        return all_coordinates, synth_spots

    
    @staticmethod
    def ripley_based_gaussian_synth_spot_generator(num_spots, cell_mask_img, covariance_matrix, gauss_kernel_size=2):
        """
        Generates synthetic spots based on a Gaussian distribution using a precomputed covariance matrix.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        covariance_matrix : ndarray
            Covariance matrix for the Gaussian distribution.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))
        nuc_mask = cell_mask_img
        synth_spots = np.zeros((256, 256))
        cell_center = np.array([128, 128])

        while all_coordinates.shape[0] < num_spots:
            xy_coor = np.random.multivariate_normal([0,0], covariance_matrix)/pixpermic + cell_center
            if 0 <= xy_coor[0] < 256 and 0 <= xy_coor[1] < 256:
                mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
                if mask_value > 0:
                    synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                    all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0).astype(int)

        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("float")
        synth_spots = exposure.rescale_intensity(synth_spots, out_range=(0, 255)).astype("uint8")

        return all_coordinates, synth_spots

    @staticmethod
    def uniform_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2):
        """
        Generates synthetic spots uniformly distributed within a given cell mask.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))
        nuc_mask = cell_mask_img
        synth_spots = np.zeros((256, 256))

        while all_coordinates.shape[0] < num_spots:
            xy_coor = np.array([np.random.uniform(0, 256), np.random.uniform(0, 256)])
            mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
            if mask_value > 0:
                synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0)

        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("uint8")

        return all_coordinates, synth_spots

    @staticmethod
    def poisson_disk_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, min_dist=10):
        """
        Generates synthetic spots based on a Poisson Disk sampling method, ensuring a minimum distance between spots.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).
        min_dist : float
            Minimum distance allowed between generated spots.

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))
        nuc_mask = cell_mask_img
        synth_spots = np.zeros((256, 256))

        while all_coordinates.shape[0] < num_spots:
            xy_coor = np.array([np.random.uniform(0, 256), np.random.uniform(0, 256)])
            mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
            if mask_value > 0:
                if all(np.linalg.norm(xy_coor - np.array(s)) >= min_dist for s in all_coordinates):
                    synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                    all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0)

        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("uint8")

        return all_coordinates, synth_spots

    @staticmethod
    def neyman_scott_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, num_parents=10, cluster_std=5):
        """
        Generates synthetic spots using the Neyman-Scott clustering process.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).
        num_parents : int, optional
            Number of parent points to generate clusters around (default is 10).
        cluster_std : float, optional
            Standard deviation of the Gaussian distribution around parent points (default is 5).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))
        nuc_mask = cell_mask_img
        synth_spots = np.zeros((256, 256))

        # Generate parent points uniformly
        parent_points = np.random.uniform(0, 256, size=(num_parents, 2))

        while all_coordinates.shape[0] < num_spots:
            # Select a random parent point
            parent_point = parent_points[np.random.randint(0, num_parents)]

            # Generate offspring point around the parent point
            xy_coor = np.array([np.random.normal(parent_point[0], cluster_std), np.random.normal(parent_point[1], cluster_std)])

            # Ensure the point is within the bounds of the image
            xy_coor = np.clip(xy_coor, 0, 255)

            # Check if the point is inside the nucleus mask
            mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
            if mask_value > 0:
                synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0)

        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("uint8")

        return all_coordinates, synth_spots

    @staticmethod
    def matern_cluster_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, num_parents=10, cluster_radius=10):
        """
        Generates synthetic spots using the Matérn clustering process.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).
        num_parents : int, optional
            Number of parent points to generate clusters around (default is 10).
        cluster_radius : float, optional
            Radius within which offspring points are generated around parent points (default is 10).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))
        nuc_mask = cell_mask_img
        synth_spots = np.zeros((256, 256))

        # Generate parent points uniformly
        parent_points = np.random.uniform(0, 256, size=(num_parents, 2))

        while all_coordinates.shape[0] < num_spots:
            # Select a random parent point
            parent_point = parent_points[np.random.randint(0, num_parents)]

            # Generate offspring point within a fixed radius around the parent point
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, cluster_radius)
            xy_coor = parent_point + np.array([distance * np.cos(angle), distance * np.sin(angle)])

            # Ensure the point is within the bounds of the image
            xy_coor = np.clip(xy_coor, 0, 255)

            # Check if the point is inside the nucleus mask
            mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
            if mask_value > 0:
                synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0)

        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("uint8")

        return all_coordinates, synth_spots

    @staticmethod
    def hard_core_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, min_distance=10):
        """
        Generates synthetic spots based on the Hard-Core Process with a minimum distance constraint.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).
        min_distance : float, optional
            Minimum allowable distance between spots (default is 10).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))
        nuc_mask = cell_mask_img
        synth_spots = np.zeros((256, 256))

        while all_coordinates.shape[0] < num_spots:
            xy_coor = np.array([np.random.uniform(0, 256), np.random.uniform(0, 256)])

            # Check if the point is inside the nucleus mask
            mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
            if mask_value > 0:
                # Check distance from all previously generated points
                if all(np.linalg.norm(all_coordinates - xy_coor, axis=1) >= min_distance):
                    synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                    all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0)

        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("uint8")

        return all_coordinates, synth_spots

    @staticmethod
    def soft_core_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, min_distance=10, repulsion_strength=2):
        """
        Generates synthetic spots using the Soft-Core Process, with a probabilistic repulsion effect.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).
        min_distance : float, optional
            Minimum allowable distance between spots (default is 10).
        repulsion_strength : float, optional
            Strength of the repulsion effect. Higher values increase the probability of repulsion (default is 2).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))
        nuc_mask = cell_mask_img
        synth_spots = np.zeros((256, 256))

        while all_coordinates.shape[0] < num_spots:
            xy_coor = np.array([np.random.uniform(0, 256), np.random.uniform(0, 256)])

            # Check if the point is inside the nucleus mask
            mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
            if mask_value > 0:
                # Calculate distances from all previously generated points
                distances = np.linalg.norm(all_coordinates - xy_coor, axis=1)

                # Calculate repulsion probability based on the closest point
                if len(distances) == 0 or np.random.rand() < np.exp(-repulsion_strength * (min_distance / np.min(distances))):
                    synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                    all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0)

        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("uint8")

        return all_coordinates, synth_spots

    @staticmethod
    def strauss_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, interaction_radius=10, inhibition_strength=0.5):
        """
        Generates synthetic spots based on the Strauss Process, incorporating both clustering and inhibition.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).
        interaction_radius : float, optional
            Radius within which points inhibit each other (default is 10).
        inhibition_strength : float, optional
            Strength of the inhibition effect. Lower values increase the inhibition (default is 0.5).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))
        nuc_mask = cell_mask_img
        synth_spots = np.zeros((256, 256))

        while all_coordinates.shape[0] < num_spots:
            xy_coor = np.array([np.random.uniform(0, 256), np.random.uniform(0, 256)])

            # Check if the point is inside the nucleus mask
            mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
            if mask_value > 0:
                # Calculate distances from all previously generated points
                distances = np.linalg.norm(all_coordinates - xy_coor, axis=1)

                # Count points within the interaction radius
                num_close_points = np.sum(distances < interaction_radius)

                # Probability of accepting the new point decreases with the number of close points
                acceptance_prob = inhibition_strength ** num_close_points

                if np.random.rand() < acceptance_prob:
                    synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                    all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0)

        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("uint8")

        return all_coordinates, synth_spots

    @staticmethod
    def thomas_spot_generator(num_spots, cell_mask_img, gauss_kernel_size=2, num_parents=10, cluster_std=5):
        """
        Generates synthetic spots using the Thomas Process, a Gaussian clustering process.

        Parameters
        ----------
        num_spots : int
            Number of spots to generate.
        cell_mask_img : ndarray
            Binary mask image of the cell.
        gauss_kernel_size : int, optional
            Size of the Gaussian kernel for smoothing the spots (default is 2).
        num_parents : int, optional
            Number of parent points to generate clusters around (default is 10).
        cluster_std : float, optional
            Standard deviation of the Gaussian distribution around parent points (default is 5).

        Returns
        -------
        all_coordinates : ndarray
            Coordinates of the generated spots.
        synth_spots : ndarray
            Image with synthetic spots.
        """

        all_coordinates = np.zeros((0, 2))
        nuc_mask = cell_mask_img
        synth_spots = np.zeros((256, 256))

        # Generate parent points uniformly
        parent_points = np.random.uniform(0, 256, size=(num_parents, 2))

        while all_coordinates.shape[0] < num_spots:
            # Select a random parent point
            parent_point = parent_points[np.random.randint(0, num_parents)]

            # Generate offspring point around the parent point using a Gaussian distribution
            xy_coor = np.array([np.random.normal(parent_point[0], cluster_std), np.random.normal(parent_point[1], cluster_std)])

            # Ensure the point is within the bounds of the image
            xy_coor = np.clip(xy_coor, 0, 255)

            # Check if the point is inside the nucleus mask
            mask_value = nuc_mask[int(xy_coor[0]), int(xy_coor[1])]
            if mask_value > 0:
                synth_spots[int(xy_coor[0]), int(xy_coor[1])] = mask_value
                all_coordinates = np.concatenate((all_coordinates, xy_coor.reshape(1, 2)), axis=0)

        synth_spots = gaussian(synth_spots, gauss_kernel_size).astype("uint8")

        return all_coordinates, synth_spots

