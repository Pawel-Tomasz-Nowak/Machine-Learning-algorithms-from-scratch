import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the differentiation module from the core package
import utils.differentiation as diff


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
        g_tol: float = 1e-4,
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
        M0: np.ndarray | None = None,
        max_iter: int = 25_000
    ) -> np.ndarray:
        """
        Perform gradient descent optimization with momentum.

        Args:
            f (Callable): Function to minimize.
            x0 (np.ndarray): Initial point.
            M0 (np.ndarray or None, optional): Initial momentum vector. Default is zeros.
            max_iter (int, optional): Maximum number of iteration. Can be infinity

        Returns:
            np.ndarray: The point that (approximately) minimizes the function
        """
        # Rename the initial point
        xt: np.ndarray = x0.copy()

        # Initialize momentum vector if not provided
        Mt: np.ndarray = np.zeros_like(xt) if M0 is None else M0.copy()

        # Iteration counter
        t: int = 0

        # Compute the gradient at the initial point
        grad_xt: np.ndarray = self.num_der(f, xt, self.h)

        # Iterate until the gradient norm is less than the tolerance
        while t < max_iter and np.linalg.norm(grad_xt) > self.g_tol:
            # Update momentum
            Mt = self.beta * Mt + (1 - self.beta) * grad_xt

            # Update the point in the direction of the negative momentum
            xt = xt - self.lr * Mt

            # Recompute the gradient at the new point
            grad_xt = self.num_der(f, xt, self.h)

            # Increment the counter
            t += 1


        return xt