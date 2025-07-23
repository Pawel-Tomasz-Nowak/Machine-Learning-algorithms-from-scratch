import numpy as np
from bootstrap import bootstrap_mean as boot_mean
from typing import Callable

def mse_metric(y_true: np.ndarray, y_pred: np.ndarray, boot_n: int = 25, frac: float = 1) -> float:
    """
    Compute the Mean Squared Error (MSE) between true and predicted values using bootstrap

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Mean squared error.
    """
    # Concatenate y_true and y_pred
    Y: np.ndarray = np.concat([y_true, y_pred],
                              axis = 1)

    # A function, f, to be estimated with bootstrap
    f: Callable = lambda X: np.sum((X[:, 0] - X[:, 1])**2)/X.shape[0]


    # Perform bootstrap
    boot_mse: float = float(boot_mean(f, Y, boot_n, frac))

    return boot_mse

def rmse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Root mean squared error.
    """
    return np.sqrt(mse_metric(y_true, y_pred))

def r2_metric(y_true: np.ndarray, y_pred: np.ndarray,boot_n: int = 25, frac: float = 1) -> float:
    """
    Compute the R^2 (coefficient of determination) regression score using bootstrap

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: R^2 score.
    """
    Y: np.ndarray = np.concat([y_true, y_pred],
                              axis = 1)

    
    # Define a function for estimating R^2
    def r2_fun(X: np.ndarray) -> float:
        yt: np.ndarray = X[:, 0]
        yp: np.ndarray = X[:, 1]

        ss_res: float = np.sum( (yt-yp)**2)
        ss_tot: float = float(np.sum(   (yt - np.mean(yt))**2  ))


        return 1 - ss_res/ss_tot if ss_tot !=0 else 0.0
    
    # Estimate R^2
    return float(boot_mean(r2_fun, Y, boot_n, frac))
    