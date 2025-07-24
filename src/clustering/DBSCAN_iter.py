import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.distance_measures as measures


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    DBSCAN groups together points that are closely packed while marking points
    in low-density regions as outliers (noise). It uses two parameters:
    - eps: defines the epsilon-neighborhood radius
    - min_pts: minimum number of points required to form a dense region (core point)

    Parameters:
        min_pts (int): Minimum number of points to form a core point.
        dist_meas (Callable): Distance measure function.
        eps (float): Maximum distance for two samples to be considered as neighbors.
    """

    def __init__(
        self,
        min_pts: int = 3,
        dist_meas: Callable[[np.ndarray, np.ndarray], float] = measures.Euclidean_distance,
        eps: float = 0.5
    ) -> None:
        """
        Initialize DBSCAN clustering algorithm.

        Args:
            min_pts (int): Minimum number of points to form a core point.
            dist_meas (Callable): Distance measure function for computing point distances.
            eps (float): Maximum distance for two samples to be considered neighbors.
        """
        self.min_pts: int = min_pts
        self.dist_meas: Callable[[np.ndarray, np.ndarray], float] = dist_meas
        self.eps: float = eps

    def _compute_dissimilarity_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the pairwise dissimilarity (distance) matrix for dataset X.

        Creates a symmetric matrix where element (i,j) represents the distance
        between point i and point j in the dataset.

        Args:
            X (np.ndarray): Dataset of shape (n_samples, n_features).

        Returns:
            np.ndarray: Symmetric distance matrix of shape (n_samples, n_samples).
        """
        n_samples: int = X.shape[0]
        dissim_matrix: np.ndarray = np.zeros((n_samples, n_samples), dtype=np.float64)

        # Compute upper triangle and mirror to lower triangle for efficiency
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance: float = self.dist_meas(X[i], X[j])
                dissim_matrix[i, j] = distance
                dissim_matrix[j, i] = distance  # Symmetric matrix

        return dissim_matrix

    def _find_eps_vicinity(self, dissim_matrix: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Find indices of all points within eps-vicinity of the given point.

        The eps-vicinity (or epsilon-neighborhood) contains all points whose
        distance to the query point is less than the eps parameter.

        Args:
            dissim_matrix (np.ndarray): Precomputed pairwise distance matrix.
            point_idx (int): Index of the query point.

        Returns:
            np.ndarray: Array of indices representing neighbors within eps-vicinity.
        """
        distances: np.ndarray = dissim_matrix[point_idx, :]
        return np.where(distances < self.eps)[0]

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Perform DBSCAN clustering using an iterative approach.

        The algorithm identifies three types of points:
        - Core points: have at least min_pts neighbors within eps distance
        - Border points: within eps distance of a core point but not core themselves
        - Noise points: neither core nor border points

        Args:
            X (np.ndarray): Dataset of shape (n_samples, n_features).

        Returns:
            np.ndarray: Cluster labels for each sample (0 means noise/outlier).
        """
        n_samples: int = X.shape[0]
        visited: np.ndarray = np.zeros(n_samples, dtype=bool)
        cluster_labels: np.ndarray = np.zeros(n_samples, dtype=np.int32)  # 0 = noise
        cluster_id: int = 1  # Start cluster numbering from 1

        # Precompute distance matrix for efficiency
        dissim_matrix: np.ndarray = self._compute_dissimilarity_matrix(X)

        # Process each point in the dataset
        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            eps_vicinity: np.ndarray = self._find_eps_vicinity(dissim_matrix, point_idx)

            # Check if current point qualifies as a core point
            if len(eps_vicinity) < self.min_pts:
                # Not a core point - remains noise unless later assigned to cluster
                continue

            # Start new cluster expansion from this core point
            seeds: list[int] = list(eps_vicinity)
            cluster_labels[point_idx] = cluster_id
            seeds.remove(point_idx)  # Remove core point from expansion queue

            # Iteratively expand cluster by processing density-reachable points
            while seeds:
                current_point: int = seeds.pop()

                # Process unvisited points
                if not visited[current_point]:
                    visited[current_point] = True
                    current_vicinity: np.ndarray = self._find_eps_vicinity(
                        dissim_matrix, current_point
                    )

                    # If current point is also a core point, add its neighbors to expansion
                    if len(current_vicinity) >= self.min_pts:
                        for neighbor in current_vicinity:
                            if neighbor not in seeds:
                                seeds.append(neighbor)

                # Assign cluster label to border points and newly discovered core points
                if cluster_labels[current_point] == 0:
                    cluster_labels[current_point] = cluster_id

            cluster_id += 1  # Move to next cluster ID

        return cluster_labels