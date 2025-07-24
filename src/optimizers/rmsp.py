import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.differentiation as diff


class RootMeanSquaredPropagationOptimizer:
    """
    Root Mean Squared Propagation (RMSprop) optimizer.

    Parameters:
        lr (float): Learning rate (step size).
        g_tol (float): Convergence tolerance for gradient norm.
        eps (float): Small constant to prevent division by zero.
        beta (float): Exponential decay rate for squared gradient accumulation.
        h (float): Step size for numerical gradient computation.
        num_der (Callable): Numerical differentiation method for gradient calculation.
    """

    def __init__(
        self,
        lr: float,
        g_tol: float = 1e-4,
        eps: float = 1e-8,
        beta: float = 0.9,
        h: float = 0.01,
        num_der: Callable = diff.central_difference
    ) -> None:
        """
        Initialize the RMSprop optimizer.

        Args:
            lr (float): Learning rate for parameter updates.
            g_tol (float): Stopping criterion threshold for gradient norm.
            eps (float): Regularization constant to avoid division by zero.
            beta (float): Decay rate for squared gradient accumulation.
            h (float): Step size for numerical differentiation.
            num_der (Callable): Function for computing numerical gradients.
        """
        self.lr: float = lr
        self.g_tol: float = g_tol
        self.eps: float = eps
        self.beta: float = beta
        self.h: float = h
        self.num_der: Callable = num_der

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        V0: np.ndarray | None = None,
        max_iter: int = 25_000
    ) -> np.ndarray:
        """
        Perform RMSprop optimization.

        Args:
            f (Callable): Objective function to minimize.
            x0 (np.ndarray): Initial parameter vector.
            V0 (np.ndarray, optional): Initial squared gradient accumulator. Defaults to zeros.
            max_iter (int): Maximum number of optimization iterations.

        Returns:
            np.ndarray: Optimized parameter vector (approximate minimizer).
        """
        # Initialize current parameter vector
        xt: np.ndarray = x0.copy()

        # Initialize squared gradient accumulator
        Vt: np.ndarray = np.zeros_like(xt) if V0 is None else V0.copy()

        # Initialize iteration counter
        t: int = 0

        # Compute initial gradient
        grad_xt: np.ndarray = self.num_der(f, xt, self.h)

        # Main optimization loop
        while t < max_iter and np.linalg.norm(grad_xt) > self.g_tol:
            # Update squared gradient accumulator with exponential decay
            Vt = self.beta * Vt + (1 - self.beta) * grad_xt ** 2

            # Update parameters with adaptive learning rate
            xt = xt - self.lr * grad_xt / (np.sqrt(Vt) + self.eps)

            # Recompute gradient at new parameter values
            grad_xt = self.num_der(f, xt, self.h)

            # Increment iteration counter
            t += 1

        return xt