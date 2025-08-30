import numpy as np
import sys
import os
from typing import Callable, Optional, Union, Dict

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.base_model import BaseModel
from optimizers.adam import AdamOptimizer
import tests.unit_tests as unitests
from utils.bootstrap import Bootstrap
from src.transformers.OneHotEncoding import OneHotEncoder as OHE

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
            cut_off (float): Decision threshold for probability classification.
            bootstrap_kwargs (Optional[dict]): Parameters for Bootstrap class.
            optimizer (Callable): Optimizer class for maximizing likelihood function.
            optimizer_kwargs (Optional[dict]): Parameters passed to the optimizer constructor.
        """
        super().__init__()
        self.cut_off: float = cut_off
        self.__optimizer_kwargs: dict = optimizer_kwargs or {}
        self.__bootstrap_kwargs: dict = bootstrap_kwargs or {}
        self.optimizer = optimizer(**self.__optimizer_kwargs)

    def _sigmoid(self, b: np.ndarray, x: np.ndarray, y: int) -> float:
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
        unitests.assert_2d_same_rows(X, y_true)
        unitests.assert_feature_count(y_true, 1)
        self.p: int = X.shape[1]

        # Add intercept column to feature matrix
        X_exp: np.ndarray = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)
        b0: np.ndarray = np.ones(shape=(X_exp.shape[1], 1), dtype=np.float64)

        def neg_log_likelihood(b: np.ndarray) -> float:
            return (-1) * np.sum([
                np.log(self._sigmoid(b, X_exp[i], y_true[i, 0]))
                for i in range(X_exp.shape[0])
            ])

        def likelihood_estimation_function(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            return self.optimizer.optimize(neg_log_likelihood, b0)

        # Bootstrap parameter estimation for stability
        bootstrap_estimator = Bootstrap(likelihood_estimation_function, **self.__bootstrap_kwargs)
        self.coefficients_: np.ndarray = bootstrap_estimator.estimate(X_exp, y_true)
        self.is_fit = True

    def _compute_likelihood(self, x: np.ndarray) -> float:
        """
        Compute the probability of class 1 for a single observation.

        Args:
            x (np.ndarray): Feature vector for classification.

        Returns:
            float: Probability of class 1.
        """
        lin_comb: float = self.coefficients_[0] + (x @ self.coefficients_[1:])[0]
        p: float = 1 / (1 + np.exp(-lin_comb))
        return p

    def _classify(self, x: np.ndarray) -> int:
        """
        Classify a single data point based on computed probability.

        Args:
            x (np.ndarray): Feature vector for classification.

        Returns:
            int: Predicted class label (0 or 1).
        """
        p: float = self._compute_likelihood(x)
        return 1 if p > self.cut_off else 0

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        For a given input data, return the probabilities of each record being classified as 1.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities for each sample.
        """
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)
        probs: np.ndarray = np.array([self._compute_likelihood(x) for x in X])
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted binary class labels of shape (n_samples, 1).
        """
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)
        y_pred: np.ndarray = np.array([
            self._classify(X[i]) for i in range(X.shape[0])
        ]).reshape(-1, 1)
        return y_pred


class MultinomialLogisticRegression(BaseModel):
    """
    Multinomial logistic regression classifier using one-vs-rest binary logistic regressions.
    """

    def __init__(
        self,
        bootstrap_kwargs: Optional[dict] = None,
        optimizer: Callable = AdamOptimizer,
        optimizer_kwargs: Optional[dict] = None
    ) -> None:
        """
        Initialize multinomial logistic regression classifier.

        Args:
            bootstrap_kwargs (Optional[dict]): Parameters for Bootstrap class.
            optimizer (Callable): Optimizer class for maximizing likelihood function.
            optimizer_kwargs (Optional[dict]): Parameters passed to the optimizer constructor.
        """
        super().__init__()
        self.__optimizer_kwargs: dict = optimizer_kwargs or {}
        self.__bootstrap_kwargs: dict = bootstrap_kwargs or {}
        self.optimizer: Callable = optimizer
        self.categories: np.ndarray = np.array([])
        self.int2reg: Dict[int, BinaryLogisticRegression] = {}  # Maps category index to its logistic regression

    def validate_input(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Validates the input X and y arrays and reshapes y to 2D if necessary.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector or matrix.

        Returns:
            np.ndarray: y reshaped to 2D.
        """
        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                unitests.assert_matrix_vector_match(X, y)
                y = y.reshape(-1, 1)
            elif y.ndim == 2:
                unitests.assert_feature_count(y, 1)
                unitests.assert_2d_same_rows(X, y)
            else:
                raise ValueError(f"The shape of y should be either 1 or 2. Got {y.ndim}")
            self.output_shape = y.shape
        else:
            raise TypeError(f"Got unexpected datatype for y: {type(y)}. Expected: numpy.ndarray")
        return y

    def fit(self, X: np.ndarray, y_true: np.ndarray,
            categories: Union[list[np.ndarray], str] = 'auto') -> None:
        """
        Fit the multinomial logistic regression model using one-vs-rest strategy.

        Args:
            X (np.ndarray): Feature matrix.
            y_true (np.ndarray): Target vector or matrix.
            categories (Union[list[np.ndarray], str]): Categories for encoding.
        """
        y = self.validate_input(X, y_true)
        self.p: int = X.shape[1]
        self.OHE_inst: OHE = OHE()
        y_ohe: np.ndarray = self.OHE_inst.fit_transform(X=y, categories=categories)[0]
        self.k: int = y_ohe.shape[1]

        for cat_id in range(self.k):
            dummy_var: np.ndarray = y_ohe[:, cat_id].reshape(-1, 1)
            bin_log_reg = BinaryLogisticRegression(
                bootstrap_kwargs=self.__bootstrap_kwargs,
                optimizer=self.optimizer,
                optimizer_kwargs=self.__optimizer_kwargs
            )
            bin_log_reg.fit(X, dummy_var)
            self.int2reg[cat_id] = bin_log_reg

        self.is_fit = True

    def _compute_likelihoods(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the probabilities of belonging to each category for a single observation.

        Args:
            x (np.ndarray): Feature vector.

        Returns:
            np.ndarray: Probabilities for each category.
        """
        probs: np.ndarray = np.zeros(shape=self.k, dtype=np.float64)
        for reg_id, reg in self.int2reg.items():
            probs[reg_id] = reg._compute_likelihood(x)
        return probs

    def _classify(self, x: np.ndarray):
        """
        Predicts a single observation.

        Args:
            x (np.ndarray): Feature vector.

        Returns:
            Predicted category label.
        """
        probs: np.ndarray = self._compute_likelihoods(x)
        i: int = int(np.argmax(probs))
        cat = self.OHE_inst.feature_encoders_[0].unique_categories_[i]
        return cat

    def get_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        For each observation of X, calculates the likelihoods of belonging to each found category.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probability matrix (n_samples, n_categories).
        """
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)
        n: int = X.shape[0]
        probs = np.array([self._compute_likelihoods(X[i]) for i in range(n)])
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts multiple observations.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted category labels, shape matches training y.
        """
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)
        n: int = X.shape[0]
        y_pred: np.ndarray = np.array([self._classify(X[i]) for i in range(n)]).reshape(self.output_shape)
        return y_pred