import numpy as np
import sys
import os
from typing import Callable, Optional

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "..")))

from utils.bootstrap import Bootstrap


def mse_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **bootstrap_kwargs
) -> float:
    """
    Compute the Mean Squared Error (MSE) between true and predicted values using bootstrap.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.
        **bootstrap_kwargs: Optional bootstrap parameters:
            - boot_n (int): Number of bootstrap samples (required)
            - frac (float): Fraction of data in each bootstrap sample (default: 1.0)

    Returns:
        float: Bootstrap estimate of mean squared error.
    """
    # Concatenate y_true and y_pred as columns for bootstrap processing
    combined_data: np.ndarray = np.concatenate([y_true.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)

    # Define the MSE function for bootstrap estimation
    def mse_function(X: np.ndarray, y: np.ndarray) -> float:
        """Calculate MSE for a bootstrap sample."""
        true_values: np.ndarray = X[:, 0]
        predicted_values: np.ndarray = X[:, 1]
        return float(np.sum((true_values - predicted_values) ** 2) / X.shape[0])

    # Perform bootstrap estimation
    bootstrap_estimator = Bootstrap(mse_function, **bootstrap_kwargs)
    bootstrap_mse: float = float(bootstrap_estimator.estimate(combined_data))

    return bootstrap_mse


def rmse_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **bootstrap_kwargs
) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between true and predicted values using bootstrap.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.
        **bootstrap_kwargs: Optional bootstrap parameters:
            - boot_n (int): Number of bootstrap samples (required)
            - frac (float): Fraction of data in each bootstrap sample (default: 1.0)

    Returns:
        float: Bootstrap estimate of root mean squared error.
    """
    # Calculate RMSE as square root of MSE
    mse_value: float = mse_metric(y_true, y_pred, **bootstrap_kwargs)
    return float(np.sqrt(mse_value))


def r2_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **bootstrap_kwargs
) -> float:
    """
    Compute the R^2 (coefficient of determination) regression score using bootstrap.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.
        **bootstrap_kwargs: Optional bootstrap parameters:
            - boot_n (int): Number of bootstrap samples (required)
            - frac (float): Fraction of data in each bootstrap sample (default: 1.0)

    Returns:
        float: Bootstrap estimate of R^2 score.
    """
    # Concatenate y_true and y_pred as columns for bootstrap processing
    combined_data: np.ndarray = np.concatenate([y_true.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)

    def r2_function(X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R^2 for a bootstrap sample."""
        true_values: np.ndarray = X[:, 0]
        predicted_values: np.ndarray = X[:, 1]
        
        # Calculate sum of squares of residuals
        sum_squared_residuals: float = float(np.sum((true_values - predicted_values) ** 2))
        
        # Calculate total sum of squares
        mean_true: float = float(np.mean(true_values))
        sum_squared_total: float = float(np.sum((true_values - mean_true) ** 2))
        
        # Return R^2 score, handle division by zero
        if sum_squared_total == 0:
            return 0.0
        
        return 1.0 - (sum_squared_residuals / sum_squared_total)

    # Perform bootstrap estimation
    bootstrap_estimator = Bootstrap(r2_function, **bootstrap_kwargs)
    bootstrap_r2: float = float(bootstrap_estimator.estimate(combined_data))

    return bootstrap_r2


def mae_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **bootstrap_kwargs
) -> float:
    """
    Compute the Mean Absolute Error (MAE) between true and predicted values using bootstrap.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.
        **bootstrap_kwargs: Optional bootstrap parameters:
            - boot_n (int): Number of bootstrap samples (required)
            - frac (float): Fraction of data in each bootstrap sample (default: 1.0)

    Returns:
        float: Bootstrap estimate of mean absolute error.
    """
    # Concatenate y_true and y_pred as columns for bootstrap processing
    combined_data: np.ndarray = np.concatenate([y_true.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)

    def mae_function(X: np.ndarray, y: np.ndarray) -> float:
        """Calculate MAE for a bootstrap sample."""
        true_values: np.ndarray = X[:, 0]
        predicted_values: np.ndarray = X[:, 1]
        return float(np.sum(np.abs(true_values - predicted_values)) / X.shape[0])

    # Perform bootstrap estimation
    bootstrap_estimator = Bootstrap(mae_function, **bootstrap_kwargs)
    bootstrap_mae: float = float(bootstrap_estimator.estimate(combined_data))

    return bootstrap_mae


def accuracy_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **bootstrap_kwargs
) -> float:
    """
    Compute classification accuracy between true and predicted labels using bootstrap.

    Args:
        y_true (np.ndarray): Ground truth class labels.
        y_pred (np.ndarray): Predicted class labels.
        **bootstrap_kwargs: Optional bootstrap parameters:
            - boot_n (int): Number of bootstrap samples (required)
            - frac (float): Fraction of data in each bootstrap sample (default: 1.0)

    Returns:
        float: Bootstrap estimate of classification accuracy.
    """
    # Concatenate y_true and y_pred as columns for bootstrap processing
    combined_data: np.ndarray = np.concatenate([y_true.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)

    def accuracy_function(X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy for a bootstrap sample."""
        true_labels: np.ndarray = X[:, 0]
        predicted_labels: np.ndarray = X[:, 1]
        correct_predictions: int = int(np.sum(true_labels == predicted_labels))
        return float(correct_predictions / X.shape[0])

    # Perform bootstrap estimation
    bootstrap_estimator = Bootstrap(accuracy_function, **bootstrap_kwargs)
    bootstrap_accuracy: float = float(bootstrap_estimator.estimate(combined_data))

    return bootstrap_accuracy