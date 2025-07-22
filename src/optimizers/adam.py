import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the differentiation module from the core package
import core.differentiation as diff


class AdamOptimizer:
    """
    Adam optimizer with configurable parameters.

    Args:
        lr (float): Learning rate.
        beta1, beta2 (float): Hyperparameters for momentum and accumulator respectively
        g_tol (float): Stopping criterion for the gradient norm.
        eps (float): hyperparameter for preventing dividing by zero.
        h (float): Step size for numerical differentiation.
        num_der (Callable): Numerical differentiation method.

    """


    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        g_tol: float = 1e-4,
        eps: float = 1e-8,
        h: float = 0.01,
        num_der: Callable = diff.central_difference
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.g_tol = g_tol
        self.eps = eps
        self.h = h
        self.num_der = num_der


    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        M0: np.ndarray | None = None,
        V0: np.ndarray | None = None,
        max_iter: int = 25_000
    ) -> np.ndarray:
        """
        Perform Adam optimization

        Args:
            f (Callable): Function to minimize.
            x0 (np.ndarray): Initial point.
            M0 (np.ndarray or None): Initial momentum. Default zeros.
            V0 (np.ndarray or None): Initial accumulator. Default zeros.
            max_iter (int, optional): Maximum number of iteration. Can be infinity

        Returns:
            np.ndarray: Approximate minimizer.
        """
        # Rename the initial point for convenience
        xt: np.ndarray = x0.copy()

        # Initialize Mt and Vt
        Mt: np.ndarray = np.zeros_like(xt) if M0 is None else M0.copy()
        Vt: np.ndarray = np.zeros_like(xt) if V0 is None else V0.copy()

        # Initialize the iteration counter
        t: int = 0

        # Compute the gradient at xt (x0)
        curr_grad: np.ndarray = self.num_der(f, xt, self.h)

        # Keep iterating until exceeding max_iter or the gradient norm is small enough
        while t < max_iter and np.linalg.norm(curr_grad) > self.g_tol:
            # Compute m(t+1) and v(t+1)
            Mt: np.ndarray = self.beta1*Mt + (1-self.beta1)*curr_grad
            Vt: np.ndarray = self.beta2*Vt + (1-self.beta2)*curr_grad**2

            # Rescale m(t+1) and v(t+1)
            Mt_hat: np.ndarray = Mt/(1-self.beta1**(t+1))
            Vt_hat: np.ndarray = Vt/(1-self.beta2**(t+1))

            # Update the parameter
            xt: np.ndarray = xt - self.lr/(self.eps + np.sqrt(Vt_hat))*Mt_hat

            # Recompute gradient
            curr_grad: np.ndarray = self.num_der(f, xt, self.h)

            # Increment the counter
            t += 1

        return xt