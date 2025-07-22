from numpy import sum, abs, sqrt, ndarray

def validate_1d_same_length_arrays(x1: ndarray, x2: ndarray) -> None:
    """
    Validates that both x1 and x2 are 1D numpy arrays of the same length.
    Raises AssertionError if the conditions are not met.
    """
    assert isinstance(x1, ndarray), "x1 must be a numpy.ndarray"
    assert isinstance(x2, ndarray), "x2 must be a numpy.ndarray"
    assert x1.ndim == 1, "x1 must be a 1-dimensional array"
    assert x2.ndim == 1, "x2 must be a 1-dimensional array"
    assert x1.shape[0] == x2.shape[0], "x1 and x2 must have the same number of elements"

def Euclidean_distance(x1: ndarray, x2: ndarray) -> float:
    """
    Calculates the Euclidean distance between two 1D numpy arrays.
    """
    validate_1d_same_length_arrays(x1, x2)
    return sqrt(sum((x1 - x2) ** 2))

def Minkowski_distance(x1: ndarray, x2: ndarray, p: int = 1) -> float:
    """
    Calculates the Minkowski distance of order p between two 1D numpy arrays.
    """
    validate_1d_same_length_arrays(x1, x2)
    return (sum((abs(x1 - x2)) ** p)) ** (1 / p)