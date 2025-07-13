import numpy as np
import os
import sys
from typing import Callable

# Add parent directory to sys.path to allow importing modules from parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import differentiating as diff

class RootMeanSquaredPropagadionOptimizer:
    """
    Root Mean Squared Propagation optimizer with configurable parameters.

    Args:
        lr (float): Learning rate.
        g_tol (float): Stopping criterion for the gradient norm.
        eps (float): hyperparameter for preventing dividing by zero.
        beta (float): Momentum hyperparameter.
        h (float): Step size for numerical differentiation.
        num_der (Callable): Numerical differentiation method.

    """


    def __init__(
        self,
        lr: float,
        g_tol: float,
        eps: float = 1e-8,
        beta: float = 0.9,
        h: float = 0.01,
        num_der: Callable = diff.central_difference
    ) -> None:
        self.lr = lr
        self.g_tol = g_tol
        self.eps = eps
        self.beta = beta
        self.h = h
        self.num_der = num_der


    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        V0: np.ndarray | None = None,
        max_iter: int = 25_000
    ) -> np.ndarray:
        """
        Perform RMSProp optimization (without momentum).

        Args:
            f (Callable): Function to minimize.
            x0 (np.ndarray): Initial point.
            V0 (np.ndarray or None): Initial accumulator. Default zeros.
            max_iter (int, optional): Maximum number of iteration. Can be infinity

        Returns:
            np.ndarray: Approximate minimizer.
        """

        # Rename the initial point and V0 for convenience
        xt: np.ndarray = x0

        if V0 is None:
            Vt = np.zeros_like(xt)

        # Compute the gradient at initial point
        grad_xt: np.ndarray = self.num_der(f, xt, self.h)

        # Iteration counter
        i: int = 0

        # Keep iterating until gradient norm is less than tolerance
        while i < max_iter and np.linalg.norm(grad_xt) > self.g_tol:
            # Update the accumulator
            Vt: np.ndarray = self.beta * Vt + (1 - self.beta) * grad_xt**2

            # Update the parameter
            xt: np.ndarray = xt - self.lr * grad_xt / np.sqrt(Vt + self.eps)

            # Recompute the gradient at xk
            grad_xt: np.ndarray = self.num_der(f, xt, self.h)

            # Increment the counter
            i += 1


        return xt
    