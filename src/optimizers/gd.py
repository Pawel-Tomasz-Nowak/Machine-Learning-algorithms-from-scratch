import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the differentiation module from the core package
import utils.differentiation as diff


class GradientDescentOptimizer:
    """
    Gradient Descent optimizer with configurable parameters.

    Args:
        lr (float): Learning rate.
        g_tol (float): Stopping criterion for the gradient norm.
        h (float): Step size for numerical differentiation.
        num_der (Callable): Numerical differentiation method.
    """

    def __init__(
        self,
        lr: float,
        g_tol: float = 1e-4,
        h: float = 0.01,
        num_der: Callable = diff.central_difference
    ) -> None:
        self.lr = lr
        self.g_tol = g_tol
        self.h = h
        self.num_der = num_der

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        max_iter: int = 25_000
    ) -> np.ndarray:
        """
        Perform gradient descent optimization.

        Args:
            f (Callable): Function to minimize.
            x0 (np.ndarray): Initial point.

        Returns:
            np.ndarray: The point that (approximately) minimizes the function.
        """
        # Rename the initial point for convenience
        xt: np.ndarray = x0.copy()

        # Iteration counter
        t: int = 0

        # Compute the gradient at the initial point
        grad_xt: np.ndarray = self.num_der(f, xt, self.h)

        # Iterate until the gradient norm is less than the tolerance
        while t < max_iter and np.linalg.norm(grad_xt) > self.g_tol:
            # Update the point in the direction of the negative gradient
            xt = xt - self.lr * grad_xt

            # Recompute the gradient at the new point
            grad_xt = self.num_der(f, xt, self.h)

            # Increment the counter
            t += 1


        return xt