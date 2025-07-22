import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the differentiation module from the core package
import core.distance_measures as measures


class KMeans:
    """
    Implementation of the k-means clustering algorithm.

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
            k (int): Number of centroids.
            dist_meas (Callable): Dissimilarity measure function.
            eps (float): Tolerance for centroid movement.
            matrix_norm (Callable): Function to compute the norm between centroid matrices.
        """
        self.k: int = k
        self.dist_meas: Callable[[np.ndarray, np.ndarray], float] = dist_meas
        self.eps: float = eps
        self.matrix_norm: Callable = matrix_norm

    def _generate_centroids(
        self,
        X: np.ndarray,
        m: int
    ) -> np.ndarray:
        """
        Generate k centroids by computing the mean of m randomly drawn data points.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            m (int): Number of data points to average for each centroid.

        Returns:
            np.ndarray: Array of shape (k, n_features) containing the centroids.
        """
        n, p = X.shape
        centroids: np.ndarray = np.zeros((self.k, p), dtype=np.float64)
        for centr_id in range(self.k):
            # Randomly select a starting index for a slice of m samples
            slice_start: int = np.random.randint(0, n - m)
            # Draw m vectors from X
            vectors: np.ndarray = X[slice_start: slice_start + m]
            # Compute the mean vector to serve as a centroid
            centroid: np.ndarray = np.mean(vectors, axis=0)
            centroids[centr_id] = centroid
        return centroids

    def _stopping_criterion(
        self,
        centroids: np.ndarray,
        new_centroids: np.ndarray
    ) -> bool:
        """
        Check if the algorithm should stop based on centroid movement.

        Args:
            centroids (np.ndarray): Previous centroids.
            new_centroids (np.ndarray): Updated centroids.

        Returns:
            bool: True if the change is less than eps, False otherwise.
        """
        matrix_norm_val: float = self.matrix_norm(centroids - new_centroids)
        return matrix_norm_val < self.eps
    
    def _standarize(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize the input data matrix X.

        Each feature (column) is transformed to have zero mean and unit variance:
            standardized_value = (value - mean) / std

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Standardized data of the same shape as X.
        """
        # Compute the mean for each feature (column)
        means: np.ndarray = np.mean(X, axis=0)
        # Compute the standard deviation for each feature (column)
        stds: np.ndarray = np.std(X, axis=0)
        # Standardize the data
        return (X - means) / stds

    def fit(self, X: np.ndarray, m: float = 0.05, standarize: bool = False) -> np.ndarray:
        """
        Perform k-means clustering on the data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            m (float): Fraction of data points to average for each centroid (0 < m < 1).

        Returns:
            np.ndarray: Cluster labels for each data point.
        """
        n, p = X.shape
        if self.k > n:
            raise ValueError("Number of clusters k must be less than or equal to number of samples n.")
        if X.ndim != 2:
            raise ValueError("Input data X must be 2-dimensional.")
        if not (0 < m < 1):
            raise ValueError("Parameter m must be in the interval (0, 1).")
        
        # Standarize the variables if necessary
        if standarize:
            X: np.ndarray = self._standarize(X)
        
        num_vectors: int = max(1, int(n * m))

        # Initialize cluster labels
        cluster_labels: np.ndarray = np.zeros(n, np.int32)
        # Initialize centroids
        centroids: np.ndarray = self._generate_centroids(X, num_vectors)
        new_centroids: np.ndarray = np.copy(centroids)

        def centr_distances_func(x: np.ndarray) -> np.ndarray:
            """Compute distances from a point x to all centroids."""
            return np.array([self.dist_meas(x, centroids[i]) for i in range(self.k)])

        first_iteration: bool = True

        while not self._stopping_criterion(centroids, new_centroids) or first_iteration:
            centroids = new_centroids
            first_iteration = False

            # Compute distances to centroids for all points
            centr_distances = np.apply_along_axis(
                func1d=centr_distances_func,
                axis=1,
                arr=X
            )

            # Assign each point to the nearest centroid
            cluster_labels = centr_distances.argmin(axis=1)
            unique_cluster_labels = np.unique(cluster_labels)
            new_centroids = np.copy(centroids)

            for cluster_id in unique_cluster_labels:
                # Data points belonging to the current cluster
                X_clustered: np.ndarray = X[cluster_labels == cluster_id]
                
                # Update centroid as the mean of assigned points
                new_centroids[cluster_id] = np.mean(X_clustered, axis=0)

        # Save the converged centroids for potential predicting
        self.centroids: np.ndarray = centroids

        return cluster_labels + 1