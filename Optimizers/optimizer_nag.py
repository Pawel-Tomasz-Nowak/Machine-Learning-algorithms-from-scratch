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
        M0: np.ndarray | None = None,
        max_iter: int = 25_000
    ) -> np.ndarray:
        """
        Perform Nesterov accerelated gradient descent optimization with momentum.

        Args:
            f (Callable): Function to minimize.
            x0 (np.ndarray): Initial point.
            M0 (np.ndarray or None, optional): Initial momentum vector. Default is zeros.
            max_iter (int, optional): Maximum number of iteration. Can be infinity

        Returns:
            np.ndarray: The point that (approximately) minimizes the function
        """
        # Initialize iteration counter
        i: int = 0

        # Initialize point x(t)
        xt: np.ndarray = x0

        # Initialize momentum vector if not provided
        if M0 is None:
            Mt = np.zeros_like(xt)

        # Compute the gradient at point x(k) + beta*M(k)
        curr_grad: np.ndarray = self.num_der(f, xt + self.beta*Mt, self.h)

        # Iterate until the gradient norm is less than the tolerance
        while i < max_iter and np.linalg.norm(curr_grad) > self.g_tol:
            # Update the parameter
            xt: np.ndarray = xt + Mt
        

            # Recompute the gradient 
            curr_grad: np.ndarray = self.num_der(f, xt + self.beta*Mt)
            
            # Recompute the moment 
            Mt: np.ndarray = self.beta * Mt - self.lr * curr_grad


            # Increment the counter
            i += 1

        return xt