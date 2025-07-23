import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all classifiers and regressors.

    This class defines the interface that all models should implement.
    """

    def __init__(self) -> None:
        """
        Initialize the base model.
        """
        self.model: BaseModel = self

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector or matrix of shape (n_samples,) or (n_samples, n_outputs).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        pass