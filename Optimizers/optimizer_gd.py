import numpy as np
import os
import sys
from typing import Callable

# Add parent directory to sys.path to allow importing modules from parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import differentiating as diff


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
        g_tol: float,
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
        # Rename the initial point
        xk: np.ndarray = x0

        # Iteration counter
        i: int = 0

        # Compute the gradient at the initial point
        grad_xk: np.ndarray = self.num_der(f, xk, self.h)

        # Iterate until the gradient norm is less than the tolerance
        while i < max_iter and np.linalg.norm(grad_xk) > self.g_tol:
            # Update the point in the direction of the negative gradient
            xk = xk - self.lr * grad_xk

            # Recompute the gradient at the new point
            grad_xk = self.num_der(f, xk, self.h)

            # Increment the counter
            i += 1


        return xk