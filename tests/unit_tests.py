import numpy as np


def assert_is_ndarray(x: np.ndarray) -> None:
    """
    Assert that x is a numpy ndarray.
    
    Args:
        x (np.ndarray): Input to check.
        
    Raises:
        AssertionError: If x is not a numpy ndarray.
    """
    assert isinstance(x, np.ndarray), "Input must be a numpy.ndarray"


def assert_ndim(x: np.ndarray, ndim: int = 2) -> None:
    """
    Assert that x has the specified number of dimensions.
    
    Args:
        x (np.ndarray): Array to check.
        ndim (int): Expected number of dimensions.
        
    Raises:
        AssertionError: If x doesn't have the expected dimensions.
    """
    assert_is_ndarray(x)
    assert x.ndim == ndim, f"Array must be {ndim}-dimensional"


def assert_1d_same_length(x1: np.ndarray, x2: np.ndarray) -> None:
    """
    Assert that x1 and x2 are 1D arrays of the same length.
    
    Args:
        x1 (np.ndarray): First array to compare.
        x2 (np.ndarray): Second array to compare.
        
    Raises:
        AssertionError: If arrays are not 1D or have different lengths.
    """
    assert_is_ndarray(x1)
    assert_is_ndarray(x2)
    assert_ndim(x1, 1)
    assert_ndim(x2, 1)
    assert x1.shape[0] == x2.shape[0], "Arrays must have the same number of elements"


def assert_2d_same_rows(x1: np.ndarray, x2: np.ndarray) -> None:
    """
    Assert that x1 and x2 are 2D arrays with the same number of rows.
    
    Args:
        x1 (np.ndarray): First 2D array to compare.
        x2 (np.ndarray): Second 2D array to compare.
        
    Raises:
        AssertionError: If arrays are not 2D or have different row counts.
    """
    assert_is_ndarray(x1)
    assert_is_ndarray(x2)
    assert_ndim(x1, 2)
    assert_ndim(x2, 2)
    assert x1.shape[0] == x2.shape[0], "Arrays must be 2-dimensional and have the same number of rows"


def assert_feature_count(x: np.ndarray, p: int) -> None:
    """
    Assert that x is a 2D array with exactly p features (columns).
    
    Args:
        x (np.ndarray): Array to check.
        p (int): Expected number of features.
        
    Raises:
        AssertionError: If array doesn't have exactly p features.
    """
    assert_is_ndarray(x)
    assert_ndim(x, 2)
    assert x.shape[1] == p, f"Array must have exactly {p} features"


def assert_fitted(is_fit: bool) -> None:
    """
    Assert that a model has been fitted.
    
    Args:
        is_fit (bool): Flag indicating if model is fitted.
        
    Raises:
        AssertionError: If the model is not fitted.
    """
    assert is_fit, "The object isn't fitted yet"