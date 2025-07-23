import numpy as np
from typing import Callable


def forward_difference(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Compute the gradient of f at x0 using the forward difference method.

    Args:
        f (Callable[[np.ndarray], float]): Function to differentiate.
        x0 (np.ndarray): Point at which to compute the gradient.
        h (float): Step size.

    Returns:
        np.ndarray: Approximated gradient vector.
    """
    gradient = np.zeros_like(x0)

    for idx in np.ndindex(x0.shape):
        offset = np.zeros_like(x0)
        offset[idx] = h
        gradient[idx] = (f(x0 + offset) - f(x0)) / h

    return gradient


def backward_difference(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Compute the gradient of f at x0 using the backward difference method.

    Args:
        f (Callable[[np.ndarray], float]): Function to differentiate.
        x0 (np.ndarray): Point at which to compute the gradient.
        h (float): Step size.

    Returns:
        np.ndarray: Approximated gradient vector.
    """
    n = x0.shape[0]
    gradient = np.zeros(n, np.float64)

    for i in range(n):
        offset = np.zeros(n, np.float64)
        offset[i] = h
        gradient[i] = (f(x0) - f(x0 - offset)) / h

    return gradient


def central_difference(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Compute the gradient of f at x0 using the central difference method.

    Args:
        f (Callable[[np.ndarray], float]): Function to differentiate.
        x0 (np.ndarray): Point at which to compute the gradient.
        h (float): Step size.

    Returns:
        np.ndarray: Approximated gradient vector.
    """
    gradient = np.zeros_like(x0, np.float64)

    for idx in np.ndindex(x0.shape):
        offset = np.zeros_like(x0, np.float64)
        offset[idx] = h
        gradient[idx] = (f(x0 + offset) - f(x0 - offset)) / (2 * h)

    return gradient


def five_point_central_difference(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Compute the gradient of f at x0 using the five-point central difference method.

    Args:
        f (Callable[[np.ndarray], float]): Function to differentiate.
        x0 (np.ndarray): Point at which to compute the gradient.
        h (float): Step size.

    Returns:
        np.ndarray: Approximated gradient vector.
    """
    gradient = np.zeros_like(x0, np.float64)

    for idx in np.ndindex(x0.shape):
        offset = np.zeros_like(x0, np.float64)

        offset[idx] = 2 * h
        val1 = f(x0 + offset)

        offset[idx] = h
        val2 = f(x0 + offset)

        offset[idx] = -h
        val3 = f(x0 + offset)

        offset[idx] = -2 * h
        val4 = f(x0 + offset)

        gradient[idx] = (-val1 + 8 * val2 - 8 * val3 + val4) / (12 * h)

    return gradient