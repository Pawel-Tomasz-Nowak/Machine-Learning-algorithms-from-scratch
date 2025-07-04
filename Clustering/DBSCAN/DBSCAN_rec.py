import os
import sys
import numpy as np
from typing import Callable

# Add a path to a directory with distance_measures module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

import distance_measures as measures

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters:
        min_pts (int): Minimum number of points to form a dense region (cluster).
        dist_meas (Callable): Distance measure function.
        eps (float): Maximum distance between two samples for them to be considered as in the same neighborhood.
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
            min_pts (int): Minimum number of points to form a dense region (cluster).
            dist_meas (Callable): Distance measure function.
            eps (float): Maximum distance between two samples for them to be considered as in the same neighborhood.
        """
        self.min_pts: int = min_pts
        self.dist_meas: Callable[[np.ndarray, np.ndarray], float] = dist_meas
        self.eps: float = eps

    def form_cluster(
        self,
        X: np.ndarray,
        point: np.ndarray
    ) -> np.ndarray:
        """
        Recursively forms a cluster around a given point using density reachability.

        Args:
            X (np.ndarray): Dataset of shape (n_samples, n_features).
            point (np.ndarray): The point around which to form the cluster.

        Returns:
            np.ndarray: Boolean array indicating cluster membership (True if in cluster, False otherwise).
        """
        # Calculate distances from the current point to all points in X
        distances: np.ndarray = np.apply_along_axis(
            lambda x: self.dist_meas(x, point), axis=1, arr=X
        )

        # Find neighbors: points not yet assigned to a cluster and within eps distance
        neighbors: np.ndarray = (self.cluster_labels == 0) & (distances < self.eps)

        if np.sum(neighbors) >= self.min_pts:
            # The cluster is big enough
            unvisited_neighbors: np.ndarray = neighbors & (~self.current_cluster_mask)
            cluster_masks: list[np.ndarray] = [self.current_cluster_mask.copy()]

            for neighbor_idx in np.where(unvisited_neighbors)[0]:
                self.current_cluster_mask[neighbor_idx] = True
                neighbor_cluster_mask: np.ndarray = self.form_cluster(X, X[neighbor_idx, :])
                cluster_masks.append(neighbor_cluster_mask)

            # Merge all cluster masks using logical OR
            merged_cluster: np.ndarray = np.logical_or.reduce(cluster_masks)
            return merged_cluster
        else:
            # Not enough neighbors to form a cluster
            return np.zeros(self.n_samples, dtype=bool)

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the DBSCAN clustering algorithm to the data.

        Args:
            X (np.ndarray): Dataset of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of cluster assignments for each sample (0 means noise).
        """
        self.n_samples: int = X.shape[0]
        self.cluster_labels: np.ndarray = np.zeros(self.n_samples, np.int8)  # 0 means noise/unassigned
        cluster_id: int = 1  # The ID of the next cluster

        for i in range(self.n_samples):
            # Iterate over unassigned points to form clusters
            if self.cluster_labels[i] == 0:
                point: np.ndarray = X[i, :]
                self.current_cluster_mask: np.ndarray = np.zeros(self.n_samples, dtype=bool)
                self.current_cluster_mask[i] = True

                # Boolean array: True if point is part of the cluster
                cluster_mask: np.ndarray = self.form_cluster(X, point)

                # Assign cluster if big enough
                if np.sum(cluster_mask) >= self.min_pts:
                    self.cluster_labels[cluster_mask] = cluster_id
                    cluster_id += 1

        return self.cluster_labels