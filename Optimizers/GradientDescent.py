import numpy as np
import os
import sys
from typing import Callable

# Add parent directory to sys.path to allow importing modules from parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import differentiating as diff


def gradient_descent(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lr: float,
    eps: float,
    h: float = 0.01,
    num_der: Callable[[Callable[[np.ndarray], float], np.ndarray, float], np.ndarray] = diff.central_difference
) -> np.ndarray:
    """
    Perform gradient descent optimization.

    Args:
        f (Callable): Function to minimize.
        x0 (np.ndarray): Initial point.
        lr (float): Learning rate.
        eps (float): Stopping criterion for the gradient norm.
        h (float, optional): Step size for numerical differentiation. Default is 0.01.
        num_der (Callable, optional): Numerical differentiation method. Default is central_difference.

    Returns:
        np.ndarray: The point that (approximately) minimizes the function.
    """

    # Compute the gradient at the initial point
    grad_x0: np.ndarray = num_der(f, x0, h)

    # Iterate until the gradient norm is less than the tolerance
    while np.linalg.norm(grad_x0) > eps:

        # Update the point in the direction of the negative gradient
        x0 = x0 - lr * grad_x0

        # Recompute the gradient at the new point
        grad_x0 = num_der(f, x0, h)

    return x0