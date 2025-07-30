import numpy as np
import sys
import os

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional

from linear_regression import LinearRegression as LinReg
from transformers.poly_features import PolynomialFeatures


class PolynomialRegression(LinReg):
    """
    Polynomial regression model extending linear regression with polynomial features.
    
    This implementation automatically transforms input features to polynomial features
    up to a specified degree and applies linear regression on the expanded feature space.
    Supports all loss functions, regularization techniques, and optimizers from LinearRegression.
    """

    def __init__(self, d: int, **linreg_kwargs) -> None:
        """
        Initialize the polynomial regression model.

        Args:
            d (int): Maximum degree of polynomial features to generate.
            **linreg_kwargs: Additional keyword arguments passed to LinearRegression.
                           See LinearRegression documentation for complete parameter details.
                           Key required parameters for some components:
                           - optimizer_kwargs must include lr (float) for gradient-based optimizers
                           - bootstrap_kwargs must include boot_n (int)  
                           - reg_params must include lambda_reg (float) for most regularizers
        """
        super().__init__(**linreg_kwargs)
        self.d: int = d
        self.poly_transformer: Optional[PolynomialFeatures] = None

    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input features to polynomial features.
        
        Creates polynomial features up to degree d, including interaction terms
        between different features. The transformer is fitted on first call.
        
        Args:
            X (np.ndarray): Input feature matrix, shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Polynomial feature matrix, shape (n_samples, n_poly_features).
            
        Note:
            The number of polynomial features depends on the degree d and the number
            of input features p. For degree d and p features, the number of polynomial
            features is C(p+d, d) where C is the binomial coefficient.
        """
        if self.poly_transformer is None:
            self.poly_transformer = PolynomialFeatures(self.d)
            self.poly_transformer.fit(X)
        
        X_polynomial: np.ndarray = self.poly_transformer.transform(X)
        return X_polynomial

    def fit(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit the polynomial regression model to the data.
        
        Transforms the input features to polynomial features, then applies
        the linear regression fitting procedure with bootstrap estimation.
        
        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            y_true (np.ndarray): Target matrix, shape (n_samples, n_outputs).
            
        Raises:
            AssertionError: If input arrays are invalid or incompatible.
        """
        # Transform to polynomial features
        X_polynomial: np.ndarray = self._preprocess_features(X)
        
        # Fit using parent class method on polynomial features
        super().fit(X_polynomial, y_true)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input data using polynomial features.
        
        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted target values, shape (n_samples, n_outputs).
            
        Raises:
            AssertionError: If the model is not fitted or X has wrong shape/type.
        """
        # Check if model is fitted
        if not self.is_fit:
            raise AssertionError("The model is not fitted yet")
        
        # Validate against original feature count
        if X.shape[1] != self.p:
            raise AssertionError(f"X must have exactly {self.p} features")
        
        # Transform to polynomial features and predict
        X_polynomial: np.ndarray = self._preprocess_features(X)
        
        # Add intercept column and compute predictions
        X_with_intercept: np.ndarray = np.column_stack([np.ones(X_polynomial.shape[0]), X_polynomial])
        y_predicted: np.ndarray = X_with_intercept @ self.coefficients_

        return y_predicted
    
    def get_polynomial_feature_coefficients(self, feature_names: list[str] = []) -> dict[str, np.ndarray]:
        """
        Get coefficients for polynomial features with interpretable names.
        
        Returns a dictionary mapping polynomial feature names to their corresponding
        coefficients. Feature names are constructed from power combinations of
        input features (e.g., 'X_1^2 * X_2', 'X_1 * X_3').
        
        Args:
            feature_names (list[str]): Names for input features. If empty,
                                     uses default names 'X_1', 'X_2', etc.
                                     Length must match number of input features.
        
        Returns:
            dict[str, np.ndarray]: Dictionary mapping feature names to coefficients.
                                 Includes 'intercept' for the bias term.
        
        Raises:
            AssertionError: If feature_names has wrong type or length,
                          or polynomial transformer is not fitted.
        """
        # Validate input type
        assert isinstance(feature_names, list), "feature_names has to be a list"
        
        # Generate default feature names if list is empty
        if not feature_names:
            for i in range(1, self.p + 1):
                feature_names.append(f"X_{i}")

        # Validate feature names length
        assert len(feature_names) == self.p, f"The length of list has to be exactly {self.p}"

        # Check if polynomial transformer is fitted
        assert self.poly_transformer is not None, "The polynomial transformer has to be fitted!"
        
        # Initialize coefficient mapping with intercept
        coefficient_mapping: dict[str, np.ndarray] = {}
        coefficient_mapping["intercept"] = self.coefficients_[0]

        # Iterate over all power combinations and create feature names
        for combination_idx, power_combination in enumerate(self.poly_transformer.combs):
            polynomial_terms: list[str] = []

            # Build polynomial term name from power combination
            for feature_idx, power in enumerate(power_combination):
                if power == 0:
                    continue  # Skip features with zero power

                # Format feature term based on power
                if power == 1:
                    polynomial_terms.append(feature_names[feature_idx])
                else:
                    polynomial_terms.append(f"{feature_names[feature_idx]}^{power}")

            # Create feature name and store coefficient
            feature_name: str = " * ".join(polynomial_terms)
            coefficient_mapping[feature_name] = self.coefficients_[combination_idx + 1]

        return coefficient_mapping