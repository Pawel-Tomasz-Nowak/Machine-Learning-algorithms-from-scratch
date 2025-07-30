import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.differentiation as diff


class GradientDescentOptimizer:
    """
    Standard Gradient Descent optimizer.

    Parameters:
        lr (float): Learning rate (step size).
        g_tol (float): Convergence tolerance for gradient norm.
        h (float): Step size for numerical gradient computation.
        num_der (Callable): Numerical differentiation method for gradient calculation.
    """

    def __init__(
        self,
        lr: float = 0.5,
        g_tol: float = 1e-4,
        h: float = 0.01,
        num_der: Callable = diff.central_difference
    ) -> None:
        """
        Initialize the Gradient Descent optimizer.

        Args:
            lr (float): Learning rate for parameter updates.
            g_tol (float): Stopping criterion threshold for gradient norm.
            h (float): Step size for numerical differentiation.
            num_der (Callable): Function for computing numerical gradients.
        """
        self.lr: float = lr
        self.g_tol: float = g_tol
        self.h: float = h
        self.num_der: Callable = num_der

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        max_iter: int = 1_000
    ) -> np.ndarray:
        """
        Perform gradient descent optimization.

        Args:
            f (Callable): Objective function to minimize.
            x0 (np.ndarray): Initial parameter vector.
            max_iter (int): Maximum number of optimization iterations.

        Returns:
            np.ndarray: Optimized parameter vector (approximate minimizer).
        """
        # Initialize current parameter vector
        xt: np.ndarray = x0.copy()
        
        # Initialize iteration counter
        t: int = 0

        # Compute initial gradient
        grad_xt: np.ndarray = self.num_der(f, xt, self.h)

        # Main optimization loop
        while t < max_iter and np.linalg.norm(grad_xt) > self.g_tol:
            # Update parameters in negative gradient direction
            xt = xt - self.lr * grad_xt

            # Recompute gradient at new parameter values
            grad_xt = self.num_der(f, xt, self.h)

            # Increment iteration counter
            t += 1

        return xt