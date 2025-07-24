import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models (classifiers and regressors).

    This class defines the common interface that all supervised learning models
    should implement, ensuring consistency across different algorithms.

    Attributes:
        is_fit (bool): Flag indicating whether the model has been trained on data.
    """

    def __init__(self) -> None:
        """
        Initialize the base model.

        Sets up the basic state tracking for model training status.
        """
        self.is_fit: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the training data.

        This method trains the model using the provided feature matrix and target values.
        Implementations should update model parameters and set self.is_fit = True upon
        successful training.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector or matrix of shape (n_samples,) or (n_samples, n_outputs).

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input data.

        This method generates predictions using the trained model. Should only be
        called after the model has been fitted to training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,) or (n_samples, n_outputs).

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
            RuntimeError: If called before the model has been fitted (should be implemented
                         by subclasses as needed).
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model performance on test data.

        Default implementation computes the coefficient of determination (R²) for
        regression tasks. Subclasses may override this method to provide more
        appropriate scoring metrics.

        Args:
            X (np.ndarray): Test feature matrix of shape (n_samples, n_features).
            y (np.ndarray): True target values of shape (n_samples,) or (n_samples, n_outputs).

        Returns:
            float: Model performance score (higher is better).

        Raises:
            RuntimeError: If the model hasn't been fitted yet.
        """
        if not self.is_fit:
            raise RuntimeError("Model must be fitted before scoring.")

        y_pred = self.predict(X)
        
        # Compute R² score as default metric
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)