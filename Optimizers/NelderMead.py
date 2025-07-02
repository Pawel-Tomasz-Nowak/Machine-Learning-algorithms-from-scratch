import numpy as np
import os
import sys
from typing import Callable

# Add parent directory to sys.path to allow importing modules from parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import distance_measures as dist_measures


class NelderMeadOptimizer:
    """
    Nelder-Mead simplex optimization algorithm encapsulated in a class.

    Args:
        f (Callable): Function to minimize.
        x0 (np.ndarray): Initial guess, shape (n,).
        eps (float): Tolerance for stopping.
        norm (Callable): Norm function to compute distance.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        eps: float,
        norm: Callable[[np.ndarray], float] = dist_measures.Euclidean_distance
    ) -> None:
        self.f = f
        self.x0 = x0
        self.eps = eps
        self.norm = norm

    def stopping_condition(self, X: np.ndarray) -> bool:
        """
        Check if the simplex vertices are close enough to stop the algorithm.

        Args:
            X (np.ndarray): Simplex vertices, shape (n+1, n).

        Returns:
            bool: True if the maximum distance from the best vertex is less than eps.
        """
        distances = np.apply_along_axis(lambda x: self.norm(x - X[0, :]), 1, X[1:])
        return np.max(distances) < self.eps

    def update_vertices(self, X: np.ndarray) -> np.ndarray:
        """
        Shrink all vertices towards the best vertex (first row).

        Args:
            X (np.ndarray): Simplex vertices, shape (n+1, n).

        Returns:
            np.ndarray: Updated simplex vertices.
        """
        x0 = X[0, :]
        X[1:] = np.apply_along_axis(lambda x: x0 + 0.5 * (x - x0), 1, X[1:])
        return X

    def optimize(self) -> np.ndarray:
        """
        Run the Nelder-Mead simplex optimization algorithm.

        Returns:
            np.ndarray: Estimated minimum point.
        """
        delta = 0.05 * np.linalg.norm(self.x0)
        n = self.x0.shape[0]

        # Initialize simplex: first vertex is x0, others are x0 + delta * unit vectors
        X = np.vstack([self.x0, self.x0 + delta * np.eye(n, dtype=np.float64)])

        while not self.stopping_condition(X):
            # Evaluate function at all simplex vertices
            f_val = np.apply_along_axis(self.f, 1, X)
            # Sort vertices by function value (ascending)
            X = X[np.argsort(f_val), :]

            # Compute centroid of all but the worst vertex
            c = np.mean(X[:-1, :], axis=0)

            # Reflection
            xr = c + (c - X[-1, :])

            if self.f(xr) < self.f(X[0, :]):
                # Expansion
                xs = c + 2 * (c - X[-1, :])
                if self.f(xs) < self.f(xr):
                    X[-1, :] = xs
                else:
                    X[-1, :] = xr
            elif self.f(xr) < self.f(X[-2, :]):
                # Accept reflection
                X[-1, :] = xr
            elif self.f(xr) < self.f(X[-1, :]):
                # Outside contraction
                xz = c + 0.5 * (c - X[-1, :])
                if self.f(xz) < self.f(xr):
                    X[-1, :] = xz
                else:
                    X = self.update_vertices(X)
            else:
                # Inside contraction
                xw = c - 0.5 * (c - X[-1, :])
                if self.f(xw) < self.f(X[-1, :]):
                    X[-1, :] = xw
                else:
                    X = self.update_vertices(X)

        return X[0, :]