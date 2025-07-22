import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the differentiation module from the core package
import core.distance_measures as measures

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters:
        min_pts (int): Minimum number of points to form a core point.
        dist_meas (Callable): Distance measure function.
        eps (float): Maximum distance for two samples to be considered as neighbors (eps-vicinity).
    """

    def __init__(
        self,
        min_pts: int = 3,
        dist_meas: Callable[[np.ndarray, np.ndarray], float] = measures.Euclidean_distance,
        eps: float = 0.5
    ) -> None:
        """
        Initialize DBSCAN clustering.

        Args:
            min_pts (int): Minimum number of points to form a core point.
            dist_meas (Callable): Distance measure function.
            eps (float): Maximum distance for two samples to be considered as neighbors (eps-vicinity).
        """
        self.min_pts: int = min_pts
        self.dist_meas: Callable[[np.ndarray, np.ndarray], float] = dist_meas
        self.eps: float = eps

    def _compute_dissimilarity_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the pairwise dissimilarity (distance) matrix for dataset X.

        Args:
            X (np.ndarray): Dataset of shape (n_samples, n_features).

        Returns:
            np.ndarray: Symmetric matrix of pairwise distances.
        """
        n_samples: int = X.shape[0]
        dissim_matrix: np.ndarray = np.zeros((n_samples, n_samples), dtype=np.float64)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance: float = self.dist_meas(X[i], X[j])
                dissim_matrix[i, j] = distance
                dissim_matrix[j, i] = distance
        return dissim_matrix

    def _find_eps_vicinity(self, dissim_matrix: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Find indices of all points within eps-vicinity of the given point.

        Args:
            dissim_matrix (np.ndarray): Precomputed pairwise distance matrix.
            point_idx (int): Index of the query point.

        Returns:
            np.ndarray: Indices of neighbors within eps-vicinity.
        """
        distances: np.ndarray = dissim_matrix[point_idx, :]
        return np.where(distances < self.eps)[0]

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Perform DBSCAN clustering using an iterative approach.

        Args:
            X (np.ndarray): Dataset of shape (n_samples, n_features).

        Returns:
            np.ndarray: Cluster labels for each sample (0 means noise).
        """
        n_samples: int = X.shape[0]
        visited: np.ndarray = np.zeros(n_samples, dtype=bool)  # Track visited points
        cluster_labels: np.ndarray = np.zeros(n_samples, dtype=np.int32)  # 0 means noise
        cluster_id: int = 1  # Cluster label counter

        dissim_matrix: np.ndarray = self._compute_dissimilarity_matrix(X)

        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            eps_vicinity: np.ndarray = self._find_eps_vicinity(dissim_matrix, point_idx)

            # Check if point is a core point
            if len(eps_vicinity) < self.min_pts:
                # Not a core point, remains noise unless later density-reachable
                continue

            # Start a new cluster from this core point
            seeds: list[int] = list(eps_vicinity)
            cluster_labels[point_idx] = cluster_id
            seeds.remove(point_idx)  # Remove the core point itself

            # Expand the cluster iteratively
            while seeds:
                current_point: int = seeds.pop()

                if not visited[current_point]:
                    visited[current_point] = True

                    current_vicinity: np.ndarray = self._find_eps_vicinity(dissim_matrix, current_point)

                    # If current_point is a core point, add its neighbors
                    if len(current_vicinity) >= self.min_pts:
                        for neighbor in current_vicinity:
                            if neighbor not in seeds:
                                seeds.append(neighbor)

                # Assign cluster label if not yet assigned (border point or newly discovered core)
                if cluster_labels[current_point] == 0:
                    cluster_labels[current_point] = cluster_id

            cluster_id += 1

        return cluster_labels