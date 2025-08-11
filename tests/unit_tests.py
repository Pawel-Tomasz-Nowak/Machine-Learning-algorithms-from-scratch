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


def assert_2d_same_rows(X: np.ndarray, y: np.ndarray) -> None:
    """
    Assert that X is 2D and y is 1D or 2D, both with the same number of rows.
    
    Args:
        X (np.ndarray): Feature matrix, expected shape (n_samples, n_features).
        y (np.ndarray): Target array, expected shape (n_samples,) or (n_samples, n_targets).
        
    Raises:
        AssertionError: If X is not 2D, y is not 1D or 2D, or they have different row counts.
    """
    assert_is_ndarray(X)
    assert_is_ndarray(y)
    assert_ndim(X, 2)
    
    # y can be 1D or 2D
    assert y.ndim in [1, 2], f"y must be 1D or 2D, got {y.ndim}D"
    
    # Check that both arrays have the same number of rows
    assert X.shape[0] == y.shape[0], f"X and y must have the same number of rows: X has {X.shape[0]}, y has {y.shape[0]}"


def assert_matrix_vector_match(X: np.ndarray, y: np.ndarray) -> None:
    """
    Assert that feature matrix (2D) and label vector (1D) have compatible shapes.
    
    Validates that X is 2D, y is 1D, and they have the same number of samples.
    Commonly used for validating X (features) and y (labels) in ML algorithms.
    
    Args:
        X (np.ndarray): Feature matrix, expected shape (n_samples, n_features).
        y (np.ndarray): Label vector, expected shape (n_samples,).
        
    Raises:
        AssertionError: If X is not 2D, y is not 1D, or they have different sample counts.
    """
    assert_is_ndarray(X)
    assert_is_ndarray(y)
    assert_ndim(X, 2)
    assert_ndim(y, 1)
    assert X.shape[0] == y.shape[0], "Feature matrix and label vector must have the same number of samples"


def assert_feature_count(X: np.ndarray, expected_features: int) -> None:
    """
    Assert that X is a 2D array with exactly the expected number of features (columns).
    
    Args:
        X (np.ndarray): Feature matrix to check, expected shape (n_samples, n_features).
        expected_features (int): Expected number of features.
        
    Raises:
        AssertionError: If array doesn't have exactly the expected number of features.
    """
    assert_is_ndarray(X)
    assert_ndim(X, 2)
    assert X.shape[1] == expected_features, f"Array must have exactly {expected_features} features, got {X.shape[1]}"


def assert_fitted(is_fit: bool) -> None:
    """
    Assert that a model has been fitted.
    
    Args:
        is_fit (bool): Flag indicating if model is fitted.
        
    Raises:
        AssertionError: If the model is not fitted.
    """
    assert is_fit, "Model must be fitted before making predictions. Call fit() first."