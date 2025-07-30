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
from utils.reg_terms import L2Regularization
from utils.bootstrap import Bootstrap


class LinearRegression(BaseModel):
    """
    Multivariate linear regression with customizable loss and regularization.

    This implementation supports various loss functions, regularization techniques,
    and optimization algorithms. It uses bootstrap estimation for robust parameter
    estimation and confidence intervals.
    """

    def __init__(
        self,
        loss: Callable[[np.ndarray, np.ndarray], float] = L2,
        loss_params: Optional[dict] = None,
        reg: Callable = L2Regularization,
        reg_params: Optional[dict] = None,
        optimizer: Callable = AdamOptimizer,
        optimizer_kwargs: Optional[dict] = None,
        bootstrap_kwargs: Optional[dict] = None
    ) -> None:
        """
        Initialize the linear regression model.

        Args:
            loss (Callable[[np.ndarray, np.ndarray], float]): Loss function for training.
                Must take y_true and y_pred arrays and return a float.
            loss_params (Optional[dict]): Additional parameters for loss function.
                Depends on the specific loss function used.
            reg (Callable): Regularization class constructor.
                The class must have a compute_penalty method.
            reg_params (Optional[dict]): Parameters for regularization constructor.
                Most regularizers require lambda_reg (float) parameter.
            optimizer (Callable): Optimizer class for minimizing loss function.
                Must have an optimize method that takes (objective_func, initial_params).
                Gradient-based optimizers require lr (float) parameter.
            optimizer_kwargs (Optional[dict]): Parameters for the optimizer constructor.
                Required: lr (float) for gradient-based optimizers.
                Optional: g_tol (float), h (float), and optimizer-specific parameters.
            bootstrap_kwargs (Optional[dict]): Parameters for Bootstrap class.
                Required: boot_n (int). Optional: frac (float, default: 1.0).
        """
        super().__init__()
        
        # Loss function setup
        self.loss: Callable[[np.ndarray, np.ndarray], float] = loss
        self.loss_params: dict = loss_params or {}

        # Regularization setup
        self.reg_params: dict = reg_params or {}
        self.reg = reg(**self.reg_params)
        
        # Optimizer setup
        self.__optimizer_kwargs: dict = optimizer_kwargs or {}
        self.optimizer = optimizer(**self.__optimizer_kwargs)

        # Bootstrap setup
        self.bootstrap_kwargs: dict = bootstrap_kwargs or {}

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
            float: Total loss value (data loss + regularization penalty).
        """
        # Compute predicted values
        y_predicted: np.ndarray = X @ B
        
        # Compute the loss for each output independently and sum
        data_loss: float = sum(
            self.loss(y_true[:, i], y_predicted[:, i], **self.loss_params)
            for i in range(self.m)
        )
            
        # Add regularization penalty
        regularization_penalty: float = self.reg.compute_penalty(B)
        
        return float(data_loss + regularization_penalty)

    def fit(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit the linear regression model to the data using bootstrap estimation.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            y_true (np.ndarray): Target matrix, shape (n_samples, n_outputs).

        Raises:
            AssertionError: If input arrays are invalid or incompatible.
        """
        # Input validation
        unitests.assert_is_ndarray(X)
        unitests.assert_is_ndarray(y_true)
        unitests.assert_ndim(X, 2)
        unitests.assert_2d_same_rows(X, y_true)

        # Store dimensions
        n: int = X.shape[0]  # Number of samples
        self.p: int = X.shape[1]  # Number of predictors
        self.m: int = y_true.shape[1]  # Number of outputs

        # Add intercept column to feature matrix
        intercept_column: np.ndarray = np.ones((n, 1), dtype=np.float64)
        X_expanded: np.ndarray = np.concatenate([intercept_column, X], axis=1)

        # Initial parameter matrix for optimization
        initial_coefficients: np.ndarray = np.ones((self.p + 1, self.m), dtype=np.float64)
        
        def coefficient_estimation(X_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
            """
            Estimate coefficients for a bootstrap sample using optimization.

            Args:
                X_sample (np.ndarray): Bootstrap sample of features with intercept.
                y_sample (np.ndarray): Bootstrap sample of targets.

            Returns:
                np.ndarray: Estimated coefficient matrix.
            """
            estimated_coefficients: np.ndarray = self.optimizer.optimize(
                partial(self._loss_function, X=X_sample, y_true=y_sample),
                initial_coefficients
            )
            return estimated_coefficients
      
        # Bootstrap estimation for robust parameter estimation
        bootstrap_estimator = Bootstrap(coefficient_estimation, **self.bootstrap_kwargs)
        self.coefficients_: np.ndarray = bootstrap_estimator.estimate(X_expanded, y_true)
        
        # Mark model as fitted
        self.is_fit = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input data.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values, shape (n_samples, n_outputs).

        Raises:
            AssertionError: If the model is not fitted or X has wrong shape/type.
        """
        # Input validation
        unitests.assert_fitted(self.is_fit)
        unitests.assert_is_ndarray(X)
        unitests.assert_ndim(X, 2)
        unitests.assert_feature_count(X, self.p)

        # Add intercept column and make predictions
        X_with_intercept: np.ndarray = np.column_stack([np.ones(X.shape[0]), X])
        y_predicted: np.ndarray = X_with_intercept @ self.coefficients_

        return y_predicted