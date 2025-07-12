import numpy as np
import os
import sys
from typing import Callable

# Add parent directory to sys.path to allow importing modules from parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import differentiating as diff

class NesterovAcceleratedGradientOptimizer:
    """
    Nesterov Accelerated Gradient Descent optimizer with configurable parameters.

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
    ) -> np.ndarray:
        """
        Perform Nesterov accerelated gradient descent optimization with momentum.

        Args:
            f (Callable): Function to minimize.
            x0 (np.ndarray): Initial point.

        Returns:
            np.ndarray: The point that (approximately) minimizes the function
        """

        # Initialize points x(k-1) and x(k-2)
        xk_1: np.ndarray = x0
        xk_2: np.ndarray = x0

        # Find point y(k)
        yk: np.ndarray = xk_1 + self.beta*(xk_1 - xk_2)

        # Compute the gradient at point y(k)
        grad_xk: np.ndarray = self.num_der(f, yk, h)

        # Iterate until the gradient norm is less than the tolerance
        while np.linalg.norm(grad_xk) > self.eps:
            # Update the point
            xk = yk  - self.lr * grad_xk

            # Update x(k-1) and x(k-2)
            xk_2 = xk_1
            xk_1 = xk

            # Compute the new y(k)
            yk = xk_1 + self.beta * (xk_1 - xk_2)

            # Recompute the gradient at the new point
            grad_xk: np.ndarray = self.num_der(f, yk, h)

        return xk_1