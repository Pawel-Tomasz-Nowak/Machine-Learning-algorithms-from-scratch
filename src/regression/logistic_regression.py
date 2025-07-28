import numpy as np
import sys
import os

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Callable, Optional

from core.base_model import BaseModel
from optimizers.adam import AdamOptimizer
import tests.unit_tests as unitests
from utils.bootstrap import Bootstrap


class BinaryLogisticRegression(BaseModel):
    """
    Binary logistic regression classifier using maximum likelihood estimation.

    This implementation uses bootstrap estimation for parameter stability and
    supports various gradient-based optimizers for likelihood maximization.
    """

    def __init__(
        self,
        cut_off: float = 0.5,
        bootstrap_kwargs: Optional[dict] = None,
        optimizer: Callable = AdamOptimizer,
        optimizer_kwargs: Optional[dict] = None
    ) -> None:
        """
        Initialize binary logistic regression classifier.

        Args:
            cut_off (float): Decision threshold for probability classification 
                           (objects with probability > cut_off are assigned to class 1).
            bootstrap_kwargs (Optional[dict]): Parameters for Bootstrap class:
                - boot_n (int): Number of bootstrap replications
                - frac (float): Fraction of rows for each bootstrap sample (0 < frac <= 1)
            optimizer (Callable): Optimizer class for maximizing likelihood function.
                                 Depends on the chosen optimizer, but gradient-based optimizers
                                 typically require:
                                 - lr (float): Learning rate (required for all gradient-based)
                                 - g_tol (float): Convergence tolerance for gradient norm
                                 - h (float): Step size for numerical differentiation
                                 Additional optimizer-specific parameters:
                                 * Adam: beta1, beta2, eps
                                 * AdaGrad: eps  
                                 * GDM/NAG: beta
                                 * RMSprop: beta, eps
            optimizer_kwargs (Optional[dict]): Parameters passed to the optimizer constructor.
        """
        super().__init__()
        self.cut_off: float = cut_off

        self.__optimizer_kwargs: dict = optimizer_kwargs or {}
        self.__bootstrap_kwargs: dict = bootstrap_kwargs or {}

        self.optimizer = optimizer(**self.__optimizer_kwargs)

    def _sigmoid_fun(self, b: np.ndarray, x: np.ndarray, y: int) -> float:
        """
        Compute sigmoid probability for given parameters and data point.

        Args:
            b (np.ndarray): Coefficient vector including intercept.
            x (np.ndarray): Feature vector with intercept (expanded).
            y (int): True class label (0 or 1).

        Returns:
            float: Sigmoid probability for the given class.
        """
        lin_comb: float = (x @ b)[0]
        p: float = 1 / (1 + np.exp(-lin_comb))

        return p if y == 1 else 1 - p

    def fit(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit the binary logistic regression model using maximum likelihood estimation.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y_true (np.ndarray): Binary target vector of shape (n_samples, 1).
        """
        # Input validation
        unitests.assert_is_ndarray(X)
        unitests.assert_is_ndarray(y_true)
        unitests.assert_2d_same_rows(X, y_true)
        unitests.assert_feature_count(y_true, 1)

        # Save feature dimensionality
        self.p: int = X.shape[1]

        # Add intercept column to feature matrix
        one_col: np.ndarray = np.ones((X.shape[0], 1), dtype=np.float32)
        X_exp: np.ndarray = np.concatenate([one_col, X], axis=1)

        # Initial parameter vector for optimization
        b0: np.ndarray = np.ones(shape=(X_exp.shape[1], 1), dtype=np.float64)

        def likelihood_estimation_function(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            """
            Estimate logistic regression parameters via likelihood maximization.

            Args:
                X (np.ndarray): Expanded feature matrix with intercept.
                y (np.ndarray): Binary target vector.

            Returns:
                np.ndarray: Optimized coefficient vector.
            """
            # Define negative log-likelihood objective function
            negative_log_likelihood: Callable = lambda b: (
                (-1) * np.sum([
                    np.log(self._sigmoid_fun(b, X[i], y[i, 0]))
                    for i in range(X.shape[0])
                ])
            )

            return self.optimizer.optimize(negative_log_likelihood, b0)

        # Bootstrap parameter estimation for stability
        bootstrap_estimator = Bootstrap(likelihood_estimation_function, **self.__bootstrap_kwargs)
        self.coefficients_: np.ndarray = bootstrap_estimator.estimate(X_exp, y_true)
        self.is_fit = True

    def _classify(self, x: np.ndarray) -> int:
        """
        Classify a single data point based on computed probability.

        Args:
            x (np.ndarray): Feature vector for classification.

        Returns:
            int: Predicted class label (0 or 1).
        """
        lin_comb: float = self.coefficients_[0] + (x @ self.coefficients_[1:])[0]
        
        # Compute classification probability
        p: float = 1 / (1 + np.exp(-lin_comb))

        return 1 if p > self.cut_off else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted binary class labels of shape (n_samples, 1).
        """
        # Input validation
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)

        y_pred: np.ndarray = np.array([
            self._classify(X[i]) for i in range(X.shape[0])
        ]).reshape(-1, 1)

        return y_pred