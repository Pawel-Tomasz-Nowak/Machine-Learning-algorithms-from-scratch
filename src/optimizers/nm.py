import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.distance_measures as measures


class NelderMeadOptimizer:
    """
    Nelder-Mead simplex optimization algorithm.

    Parameters:
        eps (float): Convergence tolerance for simplex size.
        norm (Callable): Distance function for computing vertex distances.
    """

    def __init__(
        self,
        eps: float,
        norm: Callable[[np.ndarray, np.ndarray], float] = measures.Euclidean_distance
    ) -> None:
        """
        Initialize the Nelder-Mead optimizer.

        Args:
            eps (float): Tolerance threshold for stopping criterion.
            norm (Callable): Function for computing distances between vertices.
        """
        self.eps: float = eps
        self.norm: Callable[[np.ndarray, np.ndarray], float] = norm

    def _stopping_condition(self, X: np.ndarray) -> bool:
        """
        Check convergence based on simplex vertex distances.

        Args:
            X (np.ndarray): Simplex vertices of shape (n+1, n).

        Returns:
            bool: True if maximum distance from best vertex is below tolerance.
        """
        distances: np.ndarray = np.apply_along_axis(
            lambda x: self.norm(x, X[0]), axis=1, arr=X[1:]
        )
        return np.max(distances) < self.eps

    def _update_vertices(self, X: np.ndarray) -> np.ndarray:
        """
        Shrink all vertices towards the best vertex (contraction step).

        Args:
            X (np.ndarray): Simplex vertices of shape (n+1, n).

        Returns:
            np.ndarray: Updated simplex vertices after shrinkage.
        """
        x0: np.ndarray = X[0, :]
        X[1:] = np.apply_along_axis(
            lambda x: x0 + 0.5 * (x - x0), axis=1, arr=X[1:]
        )
        return X

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray
    ) -> np.ndarray:
        """
        Perform Nelder-Mead simplex optimization.

        Args:
            f (Callable): Objective function to minimize.
            x0 (np.ndarray): Initial guess vector of shape (n,).

        Returns:
            np.ndarray: Optimized parameter vector (approximate minimizer).
        """
        delta: float = float(0.05 * np.linalg.norm(x0))
        n: int = x0.shape[0]

        # Initialize simplex: first vertex is x0, others are x0 + delta * unit vectors
        X: np.ndarray = np.vstack([x0, x0 + delta * np.eye(n, dtype=np.float64)])

        # Main optimization loop
        while not self._stopping_condition(X):
            # Evaluate function at all simplex vertices and sort by function value
            f_val: np.ndarray = np.apply_along_axis(f, axis=1, arr=X)
            X = X[np.argsort(f_val), :]

            # Compute centroid of all vertices except the worst
            c: np.ndarray = np.mean(X[:-1, :], axis=0)

            # Reflection step
            xr: np.ndarray = c + (c - X[-1, :])

            if f(xr) < f(X[0, :]):
                # Expansion step
                xs: np.ndarray = c + 2 * (c - X[-1, :])
                X[-1, :] = xs if f(xs) < f(xr) else xr
            elif f(xr) < f(X[-2, :]):
                # Accept reflection
                X[-1, :] = xr
            elif f(xr) < f(X[-1, :]):
                # Outside contraction step
                xz: np.ndarray = c + 0.5 * (c - X[-1, :])
                if f(xz) < f(xr):
                    X[-1, :] = xz
                else:
                    X = self._update_vertices(X)
            else:
                # Inside contraction step
                xw: np.ndarray = c - 0.5 * (c - X[-1, :])
                if f(xw) < f(X[-1, :]):
                    X[-1, :] = xw
                else:
                    X = self._update_vertices(X)

        return X[0]