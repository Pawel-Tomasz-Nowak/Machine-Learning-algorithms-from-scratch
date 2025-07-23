import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the differentiation module from the core package
import utils.differentiation as diff

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
        g_tol: float = 1e-4,
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
        # Rename the initial point for convenience
        xt: np.ndarray = x0

        Vt: np.ndarray = np.zeros_like(xt) if V0 is None else V0.copy()

        # Iteration counter
        t: int = 0

        # Compute the gradient at initial point
        grad_xt: np.ndarray = self.num_der(f, xt, self.h)

     
        # Keep iterating until gradient norm is less than tolerance
        while t < max_iter and np.linalg.norm(grad_xt) > self.g_tol:
            # Update the accumulator
            Vt: np.ndarray = self.beta * Vt + (1 - self.beta) * grad_xt**2

            # Update the parameter
            xt: np.ndarray = xt - self.lr * grad_xt / (np.sqrt(Vt) + self.eps)

            # Recompute the gradient at xk
            grad_xt: np.ndarray = self.num_der(f, xt, self.h)

            # Increment the counter
            t += 1


        return xt