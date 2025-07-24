import numpy as np


def L1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the L1 loss (sum of absolute errors) between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: L1 loss (sum of absolute differences).
    """
    r: np.ndarray = y_true - y_pred
    return float(np.sum(np.abs(r)))


def L2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the L2 loss (sum of squared errors) between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: L2 loss (sum of squared differences).
    """
    r: np.ndarray = y_true - y_pred
    return float(np.sum(r ** 2))


def Huber(y_true: np.ndarray, y_pred: np.ndarray, c_h: float = 0.9818) -> float:
    """
    Compute the Huber loss between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.
        c_h (float): Huber threshold parameter.

    Returns:
        float: Huber loss.
    """
    r: np.ndarray = y_true - y_pred
    condition: np.ndarray = np.abs(r) <= c_h
    huber_loss: np.ndarray = np.where(
        condition,
        0.5 * r ** 2,
        c_h * (np.abs(r) - 0.5 * c_h)
    )
    
    return float(np.sum(huber_loss))