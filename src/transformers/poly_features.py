import numpy as np
import sys
import os

# --- Import order: stdlib, third-party, project-local ---
# Ensure the parent directory is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tests.unit_tests as unitests

class PolynomialFeatures:
    """
    Generate polynomial features for input data up to a given degree.
    """

    def __init__(self, d: int) -> None:
        """
        Args:
            d (int): Maximum sum of powers of features (polynomial degree).
        """
        self.d = d
        self.is_fit: bool = False

    def _find_polynomial_combinations(self, comb: np.ndarray, i: int = 0) -> None:
        """
        Recursively find all combinations of feature powers that sum to degree d.
        """
        if i < self.p - 1:
            for jk in range(self.d - np.sum(comb) + 1):
                new_comb: np.ndarray = np.copy(comb)
                new_comb[i] = jk
                self._find_polynomial_combinations(new_comb, i + 1)
        elif i == self.p - 1:
            new_comb: np.ndarray = np.copy(comb)
            new_comb[i] = self.d - np.sum(comb)
            self.combs.append(new_comb)

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the polynomial feature generator to the data.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
        """
        unitests.assert_is_ndarray(X)
        unitests.assert_ndim(X)
        self.p = X.shape[1]
        self.combs: list[np.ndarray] = []
        comb: np.ndarray = np.zeros(self.p, dtype=np.uint32)
        self._find_polynomial_combinations(comb)
        self.is_fit = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data to polynomial features.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Matrix of polynomial features of shape (n_samples, n_polynomial_features).
        """
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)
        n_combs, n= len(self.combs), X.shape[0]
        pol_features: np.ndarray = np.zeros(shape=(n, n_combs), dtype=np.float64)
        for i, comb in enumerate(self.combs):
            pol_features[:, i] = np.prod(X ** comb, axis=1)
        return pol_features