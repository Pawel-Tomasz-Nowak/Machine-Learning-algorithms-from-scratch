import numpy as np
import sys
import os

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "..")))

from typing import Callable, Optional
from functools import partial

from core.base_model import BaseModel
from utils.loss import L2
from optimizers.adam import AdamOptimizer
import tests.unit_tests as unitests


class LinearRegression(BaseModel):
    """
    Multivariate linear regression with customizable loss and regularization.

    Parameters:
        loss (Callable): Loss function to use (default: L2).
        loss_params (Optional[dict]): Parameters for the loss function.
        reg (Callable): Regularization function (default: no penalty).
        reg_params (Optional[dict]): Parameters for the regularization function.
    """

    def __init__(
        self,
        loss: Callable = L2,
        loss_params: Optional[dict] = None,
        reg: Callable = lambda x: 0,
        reg_params: Optional[dict] = None,
    ) -> None:
        """
        Initialize the linear regression model.

        Args:
            loss (Callable): Loss function for training.
            loss_params (Optional[dict]): Additional parameters for loss function.
            reg (Callable): Regularization function applied to coefficients.
            reg_params (Optional[dict]): Additional parameters for regularization.
        """
        super().__init__()
        self.loss: Callable = loss
        self.loss_params: dict = loss_params or {}
        self.reg: Callable = reg
        self.reg_params: dict = reg_params or {}

    def _loss_function(
        self,
        B: np.ndarray,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute the total loss (data loss + regularization) for given coefficients.

        Args:
            B (np.ndarray): Coefficient matrix (including intercept), shape (p+1, m).
            X (np.ndarray): Feature matrix with intercept column, shape (n, p+1).
            y_true (np.ndarray): True target values, shape (n, m).

        Returns:
            float: Total loss value.
        """
        y_pred: np.ndarray = X @ B  # Predicted values
        
        # Compute the loss for each output independently
        losses: np.ndarray = np.array([
            self.loss(y_true[:, i], y_pred[:, i], **self.loss_params)
            for i in range(self.m)
        ])
        
        return float(np.sum(losses) + self.reg(B, **self.reg_params))

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        optimizer: Callable = AdamOptimizer,
        optimizer_params: Optional[dict] = None,
        lr: float = 0.5
    ) -> np.ndarray:
        """
        Fit the linear regression model to the data.

        Args:
            X (np.ndarray): Feature matrix, shape (n, p).
            y_true (np.ndarray): Target matrix, shape (n, m).
            optimizer (Callable): Optimizer class to use (default: AdamOptimizer).
            optimizer_params (Optional[dict]): Parameters for the optimizer.
            lr (float): Learning rate for the optimizer.

        Returns:
            np.ndarray: Fitted coefficient matrix, shape (p+1, m).
        """
        # Input validation
        unitests.assert_is_ndarray(X)
        unitests.assert_is_ndarray(y_true)
        unitests.assert_ndim(X, 2)
        unitests.assert_2d_same_rows(X, y_true)

        n: int = X.shape[0]
        self.p: int = X.shape[1]  # Number of predictors
        self.m: int = y_true.shape[1]  # Number of outputs

        # Add intercept column to X
        one_col: np.ndarray = np.ones((n, 1), dtype=np.float32)
        X_exp: np.ndarray = np.concatenate([one_col, X], axis=1)  # expanded X

        # Prepare optimizer
        optimizer_params = optimizer_params or {}
        optimizer_inst = optimizer(lr, **optimizer_params)

        # Optimize the loss function to find coefficients
        initial_B = np.ones((self.p + 1, self.m))
        coefficient = optimizer_inst.optimize(
            partial(self._loss_function, X=X_exp, y_true=y_true),
            initial_B
        )

        self.coefficient_: np.ndarray = coefficient
        self.is_fit = True

        return coefficient

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input data.

        Args:
            X (np.ndarray): Feature matrix, shape (n, p).

        Returns:
            np.ndarray: Predicted target values, shape (n, m).

        Raises:
            AssertionError: If the model is not fitted or X has wrong shape/type.
        """
        # Input validation
        unitests.assert_fitted(self.is_fit)
        unitests.assert_is_ndarray(X)
        unitests.assert_ndim(X, 2)
        unitests.assert_feature_count(X, self.p)

        n: int = X.shape[0]
        one_col: np.ndarray = np.ones((n, 1), dtype=np.float32)
        X_exp: np.ndarray = np.concatenate([one_col, X], axis=1)
        y_pred: np.ndarray = X_exp @ self.coefficient_

        return y_pred