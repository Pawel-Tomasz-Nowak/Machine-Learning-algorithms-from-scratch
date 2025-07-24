import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.distance_measures as measures


class KMeans:
    """
    Implementation of the k-means clustering algorithm.

    K-means partitions data into k clusters by iteratively updating cluster centroids
    and reassigning points to the nearest centroid until convergence.

    Parameters:
        k (int): Number of centroids (clusters).
        dist_meas (Callable): Dissimilarity measure function (e.g., Euclidean distance).
        eps (float): Tolerance for centroid movement to determine convergence.
        matrix_norm (Callable): Function to compute the norm between centroid matrices.
    """

    def __init__(
        self,
        k: int,
        dist_meas: Callable[[np.ndarray, np.ndarray], float] = measures.Euclidean_distance,
        eps: float = 0.01,
        matrix_norm: Callable = np.linalg.norm,
    ) -> None:
        """
        Initialize the KMeans clustering algorithm.

        Args:
            k (int): Number of centroids (clusters).
            dist_meas (Callable): Distance measure function for computing point-centroid distances.
            eps (float): Convergence tolerance for centroid movement.
            matrix_norm (Callable): Function to compute matrix norm for convergence checking.
        """
        self.k: int = k
        self.dist_meas: Callable[[np.ndarray, np.ndarray], float] = dist_meas
        self.eps: float = eps
        self.matrix_norm: Callable = matrix_norm

    def _generate_centroids(self, X: np.ndarray, m: int) -> np.ndarray:
        """
        Generate k initial centroids by computing the mean of m randomly drawn data points.

        This initialization method helps avoid poor local minima by using actual
        data distribution characteristics rather than purely random placement.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            m (int): Number of data points to average for each centroid.

        Returns:
            np.ndarray: Array of shape (k, n_features) containing the initial centroids.
        """
        n, p = X.shape
        centroids: np.ndarray = np.zeros((self.k, p), dtype=np.float64)

        for centr_id in range(self.k):
            # Randomly select a starting index for a slice of m consecutive samples
            slice_start: int = np.random.randint(0, n - m)
            
            # Draw m consecutive vectors from X
            vectors: np.ndarray = X[slice_start: slice_start + m]
            
            # Compute the mean vector to serve as initial centroid
            centroid: np.ndarray = np.mean(vectors, axis=0)
            centroids[centr_id] = centroid

        return centroids

    def _stopping_criterion(
        self,
        centroids: np.ndarray,
        new_centroids: np.ndarray
    ) -> bool:
        """
        Check convergence based on centroid movement magnitude.

        The algorithm converges when the movement of centroids between iterations
        falls below the specified tolerance threshold (eps).

        Args:
            centroids (np.ndarray): Previous iteration centroids.
            new_centroids (np.ndarray): Current iteration centroids.

        Returns:
            bool: True if convergence criterion is met, False otherwise.
        """
        centroid_movement: float = self.matrix_norm(centroids - new_centroids)
        return centroid_movement < self.eps

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize the input data matrix using z-score normalization.

        Each feature (column) is transformed to have zero mean and unit variance:
            standardized_value = (value - mean) / standard_deviation

        This preprocessing step ensures all features contribute equally to distance
        calculations, preventing features with larger scales from dominating.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Standardized data of the same shape as input.
        """
        # Compute feature-wise statistics
        means: np.ndarray = np.mean(X, axis=0)
        stds: np.ndarray = np.std(X, axis=0)
        
        # Apply z-score normalization
        return (X - means) / stds

    def fit(self, X: np.ndarray, m: float = 0.05, standardize: bool = False) -> np.ndarray:
        """
        Perform k-means clustering on the input data.

        The algorithm alternates between two steps:
        1. Assignment: assign each point to the nearest centroid
        2. Update: recompute centroids as the mean of assigned points

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            m (float): Fraction of data points to average for centroid initialization (0 < m < 1).
            standardize (bool): Whether to standardize features before clustering.

        Returns:
            np.ndarray: Cluster labels for each data point (1-indexed).

        Raises:
            ValueError: If k > n_samples, X is not 2D, or m not in (0,1).
        """
        n, p = X.shape

        # Input validation
        if self.k > n:
            raise ValueError("Number of clusters k must be less than or equal to number of samples n.")
        if X.ndim != 2:
            raise ValueError("Input data X must be 2-dimensional.")
        if not (0 < m < 1):
            raise ValueError("Parameter m must be in the interval (0, 1).")

        # Preprocessing: standardize features if requested
        if standardize:
            X = self._standardize(X)

        # Calculate number of vectors for centroid initialization
        num_vectors: int = max(1, int(n * m))

        # Initialize algorithm state
        cluster_labels: np.ndarray = np.zeros(n, dtype=np.int32)
        centroids: np.ndarray = self._generate_centroids(X, num_vectors)
        new_centroids: np.ndarray = np.copy(centroids)

        def compute_centroid_distances(x: np.ndarray) -> np.ndarray:
            """Compute distances from point x to all current centroids."""
            return np.array([self.dist_meas(x, centroids[i]) for i in range(self.k)])

        first_iteration: bool = True

        # Main k-means iteration loop
        while not self._stopping_criterion(centroids, new_centroids) or first_iteration:
            centroids = new_centroids
            first_iteration = False

            # Assignment step: assign each point to nearest centroid
            centroid_distances = np.apply_along_axis(
                func1d=compute_centroid_distances,
                axis=1,
                arr=X
            )
            cluster_labels = centroid_distances.argmin(axis=1)

            # Update step: recompute centroids as cluster means
            unique_cluster_labels = np.unique(cluster_labels)
            new_centroids = np.copy(centroids)

            for cluster_id in unique_cluster_labels:
                # Get all points assigned to current cluster
                cluster_points: np.ndarray = X[cluster_labels == cluster_id]
                
                # Update centroid as arithmetic mean of cluster points
                new_centroids[cluster_id] = np.mean(cluster_points, axis=0)

        # Store final centroids for potential future predictions
        self.centroids: np.ndarray = centroids

        # Return 1-indexed cluster labels
        return cluster_labels + 1