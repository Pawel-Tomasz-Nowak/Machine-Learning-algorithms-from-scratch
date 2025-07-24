import numpy as np
import sys
import os

# Ensure the parent directory is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Callable, Optional

from linear_regression import LinearRegression as LinReg
from optimizers.adam import AdamOptimizer
from transformers.poly_features import PolynomialFeatures

class PolynomialRegression(LinReg):
    """
    Polynomial regression model extending linear regression with polynomial features.
    
    Args:
        d (int): Degree of polynomial features.
        **linreg_kwargs: Additional arguments passed to LinearRegression.
    """

    def __init__(self, d: int, **linreg_kwargs) -> None:
        super().__init__(**linreg_kwargs)
        self.d: int = d
        self.poly_transformer: Optional[PolynomialFeatures] = None

    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input features to polynomial features.
        
        Args:
            X (np.ndarray): Input feature matrix, shape (n, p).
            
        Returns:
            np.ndarray: Polynomial feature matrix, shape (n, n_poly_features).
        """
        if self.poly_transformer is None:
            self.poly_transformer = PolynomialFeatures(self.d)
            self.poly_transformer.fit(X)
        
        X_pol: np.ndarray = self.poly_transformer.transform(X)
        return X_pol

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        optimizer: Callable = AdamOptimizer,
        optimizer_params: Optional[dict] = None,
        lr: float = 0.5
    ) -> np.ndarray:
        """
        Fit the polynomial regression model to the data.
        
        Args:
            X (np.ndarray): Feature matrix, shape (n, p).
            y_true (np.ndarray): Target matrix, shape (n, m).
            optimizer (Callable): Optimizer class to use.
            optimizer_params (Optional[dict]): Parameters for the optimizer.
            lr (float): Learning rate for the optimizer.
            
        Returns:
            np.ndarray: Fitted coefficient matrix.
        """
        X_pol: np.ndarray = self._preprocess_features(X)
        return super().fit(X_pol, y_true, optimizer, optimizer_params, lr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input data using polynomial features.
        
        Args:
            X (np.ndarray): Feature matrix, shape (n, p).
            
        Returns:
            np.ndarray: Predicted target values, shape (n, m).
        """
        X_pol: np.ndarray = self._preprocess_features(X)
        return super().predict(X_pol)