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
        eps (float): Stopping criterion for the gradient norm.
        beta (float): Momentum hyperparameter.
        h (float): Step size for numerical differentiation.
        num_der (Callable): Numerical differentiation method.
    """

    def __init__(
        self,
        lr: float,
        eps: float,
        beta: float = 0.9,
        h: float = 0.01,
        num_der: Callable = diff.central_difference
    ) -> None:
        self.lr = lr
        self.eps = eps
        self.beta = beta
        self.h = h
        self.num_der = num_der

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        Vt: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Perform gradient descent optimization with momentum.

        Args:
            f (Callable): Function to minimize.
            x0 (np.ndarray): Initial point.
            Vt (np.ndarray or None, optional): Initial momentum vector. Default is zeros.

        Returns:
            np.ndarray: The point that (approximately) minimizes the function
        """
        # Rename the initial point
        xk: np.ndarray = x0

        # Initialize momentum vector if not provided
        if Vt is None:
            Vt = np.zeros_like(xk)

        # Compute the gradient at the initial point
        grad_xk: np.ndarray = self.num_der(f, xk, self.h)

        # Iterate until the gradient norm is less than the tolerance
        while np.linalg.norm(grad_xk) > self.eps:

            # Update momentum
            Vt = self.beta * Vt + (1 - self.beta) * grad_xk

            # Update the point in the direction of the negative momentum
            xk = xk - self.lr * Vt

            # Recompute the gradient at the new point
            grad_xk = self.num_der(f, xk, self.h)


        return xk