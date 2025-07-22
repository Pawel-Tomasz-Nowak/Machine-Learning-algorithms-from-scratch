import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the differentiation module from the core package
import core.differentiation as diff


class AdaGradOptimizer:
    """
    AdaGrad optimizer with configurable parameters.

    Args:
        lr (float): Learning rate.
        g_tol (float): Stopping criterion for the gradient norm.
        eps (float): hyperparameter for preventing dividing by zero.
        h (float): Step size for numerical differentiation.
        num_der (Callable): Numerical differentiation method.

    """


    def __init__(
        self,
        lr: float,
        g_tol: float = 1e-4,
        eps: float = 1e-8,
        h: float = 0.01,
        num_der: Callable = diff.central_difference
    ) -> None:
        self.lr = lr
        self.g_tol = g_tol
        self.eps = eps
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
        Perform AdaGrad optimization

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

        # Initialize Vt if not given
        Vt = np.zeros_like(xt) if V0 is None else V0
        
        # Initialize iteration counter
        t: int = 0

        # Compute the gradient at x0
        curr_grad: np.ndarray = self.num_der(f, xt, self.h)

        while t < max_iter and np.linalg.norm(curr_grad) > self.g_tol:
            # Recompute the accumulator
            Vt: np.ndarray = Vt + curr_grad**2

            # Upgrade the parameter
            xt: np.ndarray = xt - self.lr/(self.eps + np.sqrt(Vt))*curr_grad

            # Update the gradient
            curr_grad: np.ndarray = self.num_der(f, xt, self.h)

            # Increment the counter
            t += 1


        return xt