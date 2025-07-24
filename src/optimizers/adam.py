import sys
import os
import numpy as np
from typing import Callable

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.differentiation as diff


class AdamOptimizer:
    """
    Adaptive Moment Estimation (Adam) optimizer.

    
    Parameters:
        lr (float): Base learning rate (step size).
        beta1 (float): Exponential decay rate for first moment estimates (momentum).
        beta2 (float): Exponential decay rate for second moment estimates (variance).
        g_tol (float): Convergence tolerance for gradient norm.
        eps (float): Small constant to prevent division by zero.
        h (float): Step size for numerical gradient computation.
        num_der (Callable): Numerical differentiation method for gradient calculation.
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
        """
        Initialize the Adam optimizer.

        Args:
            lr (float): Base learning rate for parameter updates.
            beta1 (float): Decay rate for first moment (momentum) estimates.
            beta2 (float): Decay rate for second moment (variance) estimates.
            g_tol (float): Stopping criterion threshold for gradient norm.
            eps (float): Regularization constant to avoid division by zero.
            h (float): Step size for numerical differentiation.
            num_der (Callable): Function for computing numerical gradients.
        """
        self.lr: float = lr
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.g_tol: float = g_tol
        self.eps: float = eps
        self.h: float = h
        self.num_der: Callable = num_der

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        M0: np.ndarray | None = None,
        V0: np.ndarray | None = None,
        max_iter: int = 25_000
    ) -> np.ndarray:
        """
        Perform Adam optimization to minimize the objective function.

        The algorithm maintains exponentially weighted moving averages of gradients
        and squared gradients, with bias correction:
            m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            m_hat = m_t / (1 - beta1^t)
            v_hat = v_t / (1 - beta2^t)
            x_{t+1} = x_t - lr * m_hat / (sqrt(v_hat) + eps)

        Args:
            f (Callable): Objective function to minimize.
            x0 (np.ndarray): Initial parameter vector.
            M0 (np.ndarray, optional): Initial first moment estimate. Defaults to zeros.
            V0 (np.ndarray, optional): Initial second moment estimate. Defaults to zeros.
            max_iter (int): Maximum number of optimization iterations.

        Returns:
            np.ndarray: Optimized parameter vector (approximate minimizer).
        """
        # Initialize current parameter vector
        xt: np.ndarray = x0.copy()

        # Initialize moment estimates
        Mt: np.ndarray = np.zeros_like(xt) if M0 is None else M0.copy()  # First moment
        Vt: np.ndarray = np.zeros_like(xt) if V0 is None else V0.copy()  # Second moment

        # Initialize iteration counter
        t: int = 0

        # Compute initial gradient
        curr_grad: np.ndarray = self.num_der(f, xt, self.h)

        # Main optimization loop
        while t < max_iter and np.linalg.norm(curr_grad) > self.g_tol:
            # Update biased first and second moment estimates
            Mt = self.beta1 * Mt + (1 - self.beta1) * curr_grad
            Vt = self.beta2 * Vt + (1 - self.beta2) * curr_grad ** 2

            # Compute bias-corrected moment estimates
            Mt_hat: np.ndarray = Mt / (1 - self.beta1 ** (t + 1))
            Vt_hat: np.ndarray = Vt / (1 - self.beta2 ** (t + 1))

            # Update parameters using bias-corrected estimates
            xt = xt - (self.lr / (self.eps + np.sqrt(Vt_hat))) * Mt_hat

            # Recompute gradient at new parameter values
            curr_grad = self.num_der(f, xt, self.h)

            # Increment iteration counter
            t += 1

        return xt