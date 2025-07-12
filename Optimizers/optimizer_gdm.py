import numpy as np
import os
import sys
from typing import Callable

# Add parent directory to sys.path to allow importing modules from parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import differentiating as diff


class GradientDescentMomentumOptimizer:
    """
    Gradient Descent optimizer with momentum and configurable parameters.

    Args:
        lr (float): Learning rate.
        g_tol (float): Stopping criterion for the gradient norm.
        beta (float): Momentum hyperparameter.
        h (float): Step size for numerical differentiation.
        num_der (Callable): Numerical differentiation method.
    """

    def __init__(
        self,
        lr: float,
        g_tol: float,
        beta: float = 0.9,
        h: float = 0.01,
        num_der: Callable = diff.central_difference
    ) -> None:
        self.lr = lr
        self.g_tol = g_tol
        self.beta = beta
        self.h = h
        self.num_der = num_der

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        Vt: np.ndarray | None = None,
        max_iter: int = 25_000
    ) -> np.ndarray:
        """
        Perform gradient descent optimization with momentum.

        Args:
            f (Callable): Function to minimize.
            x0 (np.ndarray): Initial point.
            Vt (np.ndarray or None, optional): Initial momentum vector. Default is zeros.
            max_iter (int, optional): Maximum number of iteration. Can be infinity


        Returns:
            np.ndarray: The point that (approximately) minimizes the function
        """
        # Rename the initial point
        xk: np.ndarray = x0

        # Initialize momentum vector if not provided
        if Vt is None:
            Vt = np.zeros_like(xk)

        # Iteration counter
        i: int = 0

        # Compute the gradient at the initial point
        grad_xk: np.ndarray = self.num_der(f, xk, self.h)

        # Iterate until the gradient norm is less than the tolerance
        while i < max_iter and np.linalg.norm(grad_xk) > self.g_tol:

            # Update momentum
            Vt = self.beta * Vt + (1 - self.beta) * grad_xk

            # Update the point in the direction of the negative momentum
            xk = xk - self.lr * Vt

            # Recompute the gradient at the new point
            grad_xk = self.num_der(f, xk, self.h)

            # Increment the counter
            i += 1

        return xk