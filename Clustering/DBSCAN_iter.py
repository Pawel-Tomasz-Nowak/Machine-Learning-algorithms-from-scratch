import numpy as np
import distance_measures as measures
from typing import Callable

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
        self.min_pts = min_pts
        self.dist_meas = dist_meas
        self.eps = eps

    def _region_query(self, X: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Find indices of all points in X within eps of point X[point_idx].

        Args:
            X (np.ndarray): Dataset of shape (n_samples, n_features).
            point_idx (int): Index of the point to query.

        Returns:
            np.ndarray: Indices of neighbors within eps distance.
        """
        distances: np.ndarray = np.apply_along_axis(lambda x: self.dist_meas(x, X[point_idx]), 1, X)
        return np.where(distances < self.eps)[0]

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Iteratively fits the DBSCAN clustering algorithm to the data.

        Args:
            X (np.ndarray): Dataset of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of cluster assignments for each sample (0 means noise).
        """
        n_samples: int = X.shape[0]
        cluster_labels: np.ndarray = np.zeros(n_samples, np.int32)  # 0 means noise/unassigned
        visited: np.ndarray = np.zeros(n_samples, dtype=bool)
        cluster_id: int = 1

        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue
            visited[point_idx] = True

            neighbors: np.ndarray = self._region_query(X, point_idx)
            if len(neighbors) < self.min_pts:
                # Mark as noise (remains 0)
                continue

            # Start a new cluster
            cluster_labels[point_idx] = cluster_id
            seeds: list[int] = list(neighbors)
            seeds.remove(point_idx)  # Remove the seed point itself

            while seeds:
                current_point: int = seeds.pop()
                if not visited[current_point]:
                    visited[current_point] = True
                    current_neighbors: np.ndarray = self._region_query(X, current_point)
                    if len(current_neighbors) >= self.min_pts:
                        # Add new neighbors to seeds if not already in seeds
                        for n in current_neighbors:
                            if n not in seeds:
                                seeds.append(n)
                if cluster_labels[current_point] == 0:
                    cluster_labels[current_point] = cluster_id
            cluster_id += 1

        return cluster_labels