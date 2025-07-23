import numpy as np
import sys
import os

# Ensure the parent directory is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Import unit tests for validating x1 and x2 arrays
from tests.unit_tests import assert_1d_same_length

def Euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two 1D numpy arrays.
    """
    assert_1d_same_length(x1, x2)
    return np.sqrt(sum((x1 - x2) ** 2))

def Minkowski_distance(x1: np.ndarray, x2: np.ndarray, p: int = 1) -> float:
    """
    Calculates the Minkowski distance of order p between two 1D numpy arrays.
    """
    assert_1d_same_length(x1, x2)
    return (sum((abs(x1 - x2)) ** p)) ** (1 / p)