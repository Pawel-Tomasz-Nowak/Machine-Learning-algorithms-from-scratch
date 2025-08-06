import numpy as np
import sys
import os

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tests.unit_tests as unitests


class PolynomialFeatures:
    """
    Generate polynomial features for input data up to a given degree.

    Parameters:
        d (int): Maximum sum of powers of features (polynomial degree).
    """

    def __init__(self, d: int) -> None:
        """
        Initialize the polynomial feature generator.

        Args:
            d (int): Maximum sum of powers of features (polynomial degree).
        """
        assert d > 0, "The polynomial degree has to be a positive integer number"

        self.d: int = d
        self.is_fit: bool = False

    def _find_polynomial_combinations(self, d:int, comb: np.ndarray, i: int = 0) -> None:
        """
        Recursively find all combinations of feature powers that sum to degree d.

        Args:
            comb (np.ndarray): Current combination of powers being built.
            i (int): Current feature index being processed.
        """
        if i < self.p - 1:
            for jk in range(d - np.sum(comb) + 1):
                new_comb: np.ndarray = np.copy(comb)
                new_comb[i] = jk
                self._find_polynomial_combinations(d, new_comb, i + 1)
        elif i == self.p - 1:
            new_comb: np.ndarray = np.copy(comb)
            new_comb[i] = d - np.sum(comb)
            self.combs.append(new_comb)

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the polynomial feature generator to the data.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
        """
        unitests.assert_is_ndarray(X)
        unitests.assert_ndim(X, 2)

        self.p: int = X.shape[1]
        self.combs: list[np.ndarray] = []
        comb: np.ndarray = np.zeros(self.p, dtype=np.uint32)
        
        # Generate polynomial features of degree 1, 2, ..., self.d
        for deg in range(1, self.d+1):
            self._find_polynomial_combinations(deg, comb)

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
        unitests.assert_is_ndarray(X)
        unitests.assert_ndim(X, 2)
        unitests.assert_feature_count(X, self.p)

        n_combs: int = len(self.combs)
        n: int = X.shape[0]
        pol_features: np.ndarray = np.zeros(shape=(n, n_combs), dtype=np.float64)

        for i, comb in enumerate(self.combs):
            pol_features[:, i] = np.prod(X ** comb, axis=1)

        return pol_features

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the transformer and transform the data in one step.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Matrix of polynomial features of shape (n_samples, n_polynomial_features).
        """
        if not self.is_fit:
            self.fit(X)

        return self.transform(X)