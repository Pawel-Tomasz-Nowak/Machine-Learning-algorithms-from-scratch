import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.differentiation as diff


class AdaGradOptimizer:
    """
    Adaptive Gradient Algorithm (AdaGrad) optimizer.

    
    Parameters:
        lr (float): Base learning rate (step size).
        g_tol (float): Convergence tolerance for gradient norm.
        eps (float): Small constant to prevent division by zero in denominator.
        h (float): Step size for numerical gradient computation.
        num_der (Callable): Numerical differentiation method for gradient calculation.
    """

    def __init__(
        self,
        lr: float,
        g_tol: float = 1e-4,
        eps: float = 1e-8,
        h: float = 0.01,
        num_der: Callable = diff.central_difference
    ) -> None:
        """
        Initialize the AdaGrad optimizer.

        Args:
            lr (float): Base learning rate for parameter updates.
            g_tol (float): Stopping criterion threshold for gradient norm.
            eps (float): Regularization constant to avoid division by zero.
            h (float): Step size for numerical differentiation.
            num_der (Callable): Function for computing numerical gradients.
        """
        self.lr: float = lr
        self.g_tol: float = g_tol
        self.eps: float = eps
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
        Perform AdaGrad optimization to minimize the objective function.

        The algorithm maintains a running sum of squared gradients for each parameter
        and uses this to adaptively scale the learning rate:
            x_{t+1} = x_t - (lr / (eps + sqrt(V_t))) * g_t

        Args:
            f (Callable): Objective function to minimize.
            x0 (np.ndarray): Initial parameter vector.
            V0 (np.ndarray, optional): Initial gradient accumulator. Defaults to zeros.
            max_iter (int): Maximum number of optimization iterations.

        Returns:
            np.ndarray: Optimized parameter vector (approximate minimizer).
        """
        # Initialize current parameter vector
        xt: np.ndarray = x0.copy()

        # Initialize gradient accumulator (sum of squared gradients)
        Vt: np.ndarray = np.zeros_like(xt) if V0 is None else V0.copy()
        
        # Initialize iteration counter
        t: int = 0

        # Compute initial gradient
        curr_grad: np.ndarray = self.num_der(f, xt, self.h)

        # Main optimization loop
        while t < max_iter and np.linalg.norm(curr_grad) > self.g_tol:
            # Update gradient accumulator with squared current gradient
            Vt = Vt + curr_grad ** 2

            # Adaptive parameter update with scaled learning rate
            xt = xt - (self.lr / (self.eps + np.sqrt(Vt))) * curr_grad

            # Recompute gradient at new parameter values
            curr_grad = self.num_der(f, xt, self.h)

            # Increment iteration counter
            t += 1

        return xt