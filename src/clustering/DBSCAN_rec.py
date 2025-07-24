import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.distance_measures as measures


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN) - Recursive Implementation.

    DBSCAN groups together points that are closely packed while marking points
    in low-density regions as outliers (noise). This implementation uses a recursive
    approach for cluster formation through density-reachability.

    Uses two key parameters:
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

    def form_cluster(self, X: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
        Recursively forms a cluster around a given point using density reachability.

        The method implements the core concept of density-reachability in DBSCAN:
        a point is density-reachable from another if there exists a chain of points
        where each consecutive pair is within eps distance and each intermediate
        point has at least min_pts neighbors.

        Args:
            X (np.ndarray): Dataset of shape (n_samples, n_features).
            point (np.ndarray): The seed point around which to form the cluster.

        Returns:
            np.ndarray: Boolean mask indicating cluster membership (True if in cluster).
        """
        # Calculate distances from the current point to all points in dataset
        distances: np.ndarray = np.apply_along_axis(
            lambda x: self.dist_meas(x, point), axis=1, arr=X
        )

        # Find eps-neighbors: unassigned points within eps distance
        neighbors: np.ndarray = (self.cluster_labels == 0) & (distances < self.eps)

        # Check if current point qualifies as a core point
        if np.sum(neighbors) >= self.min_pts:
            # Core point found - expand cluster through density-reachable points
            unvisited_neighbors: np.ndarray = neighbors & (~self.current_cluster_mask)
            cluster_masks: list[np.ndarray] = [self.current_cluster_mask.copy()]

            # Recursively explore each unvisited neighbor
            for neighbor_idx in np.where(unvisited_neighbors)[0]:
                self.current_cluster_mask[neighbor_idx] = True
                
                # Recursive call to expand cluster from this neighbor
                neighbor_cluster_mask: np.ndarray = self.form_cluster(X, X[neighbor_idx, :])
                cluster_masks.append(neighbor_cluster_mask)

            # Merge all discovered cluster regions using logical OR
            merged_cluster: np.ndarray = np.logical_or.reduce(cluster_masks)
            return merged_cluster
        else:
            # Not enough neighbors - cannot expand cluster from this point
            return np.zeros(self.n_samples, dtype=bool)

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Perform DBSCAN clustering using recursive density-reachability expansion.

        The algorithm iterates through unassigned points, attempting to form clusters
        by recursive expansion. Points that cannot form valid clusters remain as noise.

        Args:
            X (np.ndarray): Dataset of shape (n_samples, n_features).

        Returns:
            np.ndarray: Cluster labels for each sample (0 means noise/outlier).
        """
        self.n_samples: int = X.shape[0]
        self.cluster_labels: np.ndarray = np.zeros(self.n_samples, dtype=np.int8)  # 0 = noise
        cluster_id: int = 1  # Start cluster numbering from 1

        # Process each unassigned point as potential cluster seed
        for i in range(self.n_samples):
            if self.cluster_labels[i] == 0:  # Process only unassigned points
                seed_point: np.ndarray = X[i, :]
                
                # Initialize cluster mask for recursive expansion
                self.current_cluster_mask: np.ndarray = np.zeros(self.n_samples, dtype=bool)
                self.current_cluster_mask[i] = True

                # Attempt to form cluster through recursive density-reachability
                cluster_mask: np.ndarray = self.form_cluster(X, seed_point)

                # Assign cluster ID if sufficient points found
                if np.sum(cluster_mask) >= self.min_pts:
                    self.cluster_labels[cluster_mask] = cluster_id
                    cluster_id += 1

        return self.cluster_labels