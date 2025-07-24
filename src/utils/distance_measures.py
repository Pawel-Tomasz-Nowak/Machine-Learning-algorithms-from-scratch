import numpy as np
import sys
import os

# Ensure the parent directory is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tests.unit_tests import assert_1d_same_length


def Euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two 1D numpy arrays.

    Args:
        x1 (np.ndarray): First vector.
        x2 (np.ndarray): Second vector.

    Returns:
        float: Euclidean distance between x1 and x2.
    """
    assert_1d_same_length(x1, x2)
    return float(np.sqrt(np.sum((x1 - x2) ** 2)))


def Minkowski_distance(x1: np.ndarray, x2: np.ndarray, p: int = 1) -> float:
    """
    Calculate the Minkowski distance of order p between two 1D numpy arrays.

    Args:
        x1 (np.ndarray): First vector.
        x2 (np.ndarray): Second vector.
        p (int): Order of the Minkowski distance.

    Returns:
        float: Minkowski distance of order p between x1 and x2.
    """
    assert_1d_same_length(x1, x2)
    return float((np.sum(np.abs(x1 - x2) ** p)) ** (1 / p))