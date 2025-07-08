import numpy as np
from typing import Callable

import differentiating as diff

def gradient_descent_momentum(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lr: float,
    eps: float,
    beta: float = 0.9,
    h: float = 0.01,
    Vt: np.ndarray | None = None,
    num_der: Callable = diff.central_difference
) -> np.ndarray:
    """
    Perform gradient descent optimization with momentum.

    Args:
        f (Callable): Function to minimize.
        x0 (np.ndarray): Initial point.
        lr (float): Learning rate.
        eps (float): Stopping criterion for the gradient norm.
        beta (float, optional): Momentum hyperparameter. Default is 0.9.
        h (float, optional): Step size for numerical differentiation. Default is 0.01.
        Vt (np.ndarray or None, optional): Initial momentum vector. Default is zeros.
        num_der (Callable, optional): Numerical differentiation method. Default is central_difference.

    Returns:
        np.ndarray: The point that (approximately) minimizes the function.
    """

    # Initialize momentum vector if not provided
    if Vt is None:
        Vt = np.zeros_like(x0)

    # Compute the gradient at the initial point
    grad_x0: np.ndarray = num_der(f, x0, h)

    # Iterate until the gradient norm is less than the tolerance
    while np.linalg.norm(grad_x0) > eps:

        # Update momentum
        Vt = beta * Vt + (1 - beta) * grad_x0

        # Update the point in the direction of the negative momentum
        x0 = x0 - lr * Vt

        # Recompute the gradient at the new point
        grad_x0 = num_der(f, x0, h)

    return x0