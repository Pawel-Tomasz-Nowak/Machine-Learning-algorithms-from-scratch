import numpy as np
import sys
import os
from scipy.stats import multivariate_normal as mvn

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.base_model import BaseModel
import tests.unit_tests as unitests
from utils.bootstrap import Bootstrap


class Label:
    """
    Class representing a labeled class with its statistical parameters.
    
    Encapsulates the class label, mean vector, covariance matrix,
    and multivariate normal distribution for likelihood computation.
    """

    def __init__(self, label: int, label_priori: float, mean: np.ndarray, cov_mat: np.ndarray) -> None:
        """
        Initialize a labeled class.

        Args:
            label (int): Class label.
            label_priori (float): A priori probability for the class.
            mean (np.ndarray): Mean vector for the class.
            cov_mat (np.ndarray): Covariance matrix for the class.
        """
        self.label: int = label
        self.label_priori: float = label_priori
        self.mean: np.ndarray = mean
        self.cov_mat: np.ndarray = cov_mat
        
        # Create multivariate normal distribution
        self.mvn = mvn(mean, cov_mat)

    def compute_likelihood(self, x: np.ndarray) -> float:
        """
        Compute the likelihood for a single observation.

        Args:
            x (np.ndarray): Single observation vector.

        Returns:
            float: Likelihood value (prior * conditional probability).
        """
        likelihood: float = self.label_priori * self.mvn.pdf(x)
        return likelihood


class GaussianDiscriminator(BaseModel):
    """
    Gaussian Discriminant Analysis classifier.
    
    Implements various forms of GDA including tied covariance matrices,
    diagonal covariance matrices, and MAP estimation with bootstrap
    parameter estimation for robust statistics.
    """

    def __init__(self, **bootstrap_kwargs) -> None:
        """
        Initialize the Gaussian Discriminant Analysis classifier.

        Args:
            **bootstrap_kwargs: Optional bootstrap parameters:
                - boot_n (int): Number of bootstrap samples (required)
                - frac (float): Fraction of data in each bootstrap sample (default: 1.0)
        """
        super().__init__()
        self.bootstrap_kwargs: dict = bootstrap_kwargs or {}

    def _estimate_vector_mean(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the class vector mean using bootstrap estimation.

        Args:
            X (np.ndarray): Feature matrix for a specific class.

        Returns:
            np.ndarray: Bootstrap estimate of mean vector.
        """
        def mean_vector_function(X: np.ndarray) -> np.ndarray:
            """Mean vector estimation function for bootstrap."""
            return X.mean(axis=0)

        # Bootstrap estimation of mean vector
        bootstrap_vector_mean = Bootstrap(mean_vector_function, **self.bootstrap_kwargs)
        mean_vector: np.ndarray = bootstrap_vector_mean.estimate(X)

        return mean_vector
    
    def _estimate_covariance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the covariance matrix using bootstrap estimation.

        Args:
            X (np.ndarray): Feature matrix for a specific class.

        Returns:
            np.ndarray: Bootstrap estimate of covariance matrix.
        """
        def covariance_matrix_function(X: np.ndarray) -> np.ndarray:
            """Covariance matrix estimation function for bootstrap."""
            # Handle single observation case
            if X.shape[0] == 1:
                return self.regularization * np.eye(X.shape[1])
            
            # Calculate the mean
            vector_mean: np.ndarray = X.mean(axis=0).reshape(1, -1)

            # Compute covariance matrix manually
            centered_data: np.ndarray = X - vector_mean
            covariance_matrix: np.ndarray = (centered_data.T @ centered_data) / (X.shape[0] - 1)
            
            # Add regularization to diagonal for numerical stability
            covariance_matrix += self.regularization * np.eye(X.shape[1])

            return covariance_matrix

        # Bootstrap estimation of covariance matrix
        bootstrap_cov_mat = Bootstrap(covariance_matrix_function, **self.bootstrap_kwargs)
        estimated_cov_matrix: np.ndarray = bootstrap_cov_mat.estimate(X)
        
        return estimated_cov_matrix
    
    def _estimate_prior_probability(self, y: np.ndarray, target_class: int) -> float:
        """
        Estimate the prior probability for target class using bootstrap estimation.

        Args:
            y (np.ndarray): Label vector, shape (n_samples, 1).
            target_class (int): Target class to estimate prior probability for.

        Returns:
            float: Bootstrap estimate of prior probability for target class.
        """
        def prior_probability_function(y: np.ndarray) -> float:
            """Prior probability estimation function for bootstrap."""
            class_mask: np.ndarray = (y == target_class)[:, 0]
            n_class: int = np.sum(class_mask)
            n_total: int = y.shape[0]
            return float(n_class / n_total)
        
        # Bootstrap estimation of prior probability
        bootstrap_prior = Bootstrap(prior_probability_function, **self.bootstrap_kwargs)
        estimated_prior: float = float(bootstrap_prior.estimate(y=y))
        
        return estimated_prior

    def fit(self, X: np.ndarray, y_true: np.ndarray,
            method: str = "GDA",
            lambd_map: float = 0,
            regularization: float = 1e-6) -> None:
        """
        Fit the Gaussian Discriminant Analysis model.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            y_true (np.ndarray): Target labels, shape (n_samples, 1).
            method (str): Method for estimating covariance matrices:
                - "GDA": General Gaussian discriminant analysis
                - "Tied": Tied covariance matrices (same for all classes)
                - "MAP": MAP estimation of covariance matrix
                - "Tied DGA": Tied and diagonal covariance matrices
            lambd_map (float): Lambda parameter for MAP estimation.
            regularization (float): Regularization term for numerical stability.

        Raises:
            AssertionError: If input arrays are invalid or regularization is non-positive.
        """
        # Input validation
        unitests.assert_2d_same_rows(X, y_true)
        unitests.assert_feature_count(y_true, 1)
        assert regularization > 0, "Regularization term must be positive"
        
        # Store parameters
        self.regularization: float = regularization
        self.p: int = X.shape[1]  # Number of features

        # Find unique classes
        self.unique_classes: np.ndarray = np.unique(y_true, axis=0)
        self.n_classes: int = self.unique_classes.shape[0]

        # Initialize class labels dictionary
        self.int2label: dict[int, Label] = {}

        # Track cumulative prior probabilities to ensure they sum to 1
        cumulative_prior: float = 0.0

        # Estimate parameters for each class
        for i, class_label in enumerate(self.unique_classes):
            # Extract observations for current class
            class_mask: np.ndarray = (y_true == class_label)[:, 0]
            X_class: np.ndarray = X[class_mask]
            
            # Check for insufficient data
            if X_class.shape[0] == 0:
                raise ValueError(f"No observations found for class {class_label}")

            # Estimate prior probability for current class
            if i == self.n_classes - 1:
                # For the last class, use remaining probability to ensure sum = 1
                class_prior: float = max(1.0 - cumulative_prior, 0.0)
            else:
                class_prior = self._estimate_prior_probability(y_true, class_label[0])
                cumulative_prior += class_prior

            # Estimate class parameters
            vector_mean: np.ndarray = self._estimate_vector_mean(X_class)
            covariance_matrix: np.ndarray = self._estimate_covariance_matrix(X_class)

            # Create label object and store
            label_object = Label(class_label[0], class_prior, vector_mean, covariance_matrix)
            self.int2label[i] = label_object

        # Mark model as fitted
        self.is_fit = True

    def predict_probabilities(self, X: np.ndarray, normalize: bool = False) -> np.ndarray:
        """
        Predict class probabilities for input data.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            normalize (bool): Whether to normalize likelihoods to sum to 1.

        Returns:
            np.ndarray: Probability matrix, shape (n_samples, n_classes).

        Raises:
            AssertionError: If model is not fitted or X has wrong feature count.
        """
        # Input validation
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)

        # Compute likelihoods for each observation and class
        likelihoods: np.ndarray = np.apply_along_axis(
            func1d=lambda x: np.array([
                label.compute_likelihood(x) for label in self.int2label.values()
            ]),
            axis=1,
            arr=X
        )

        if not normalize:
            return likelihoods
        
        # Normalize likelihoods to get probabilities
        normalization_terms: np.ndarray = likelihoods.sum(axis=1).reshape(-1, 1)
        
        probabilities: np.ndarray = likelihoods / normalization_terms

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples, 1).

        Raises:
            AssertionError: If model is not fitted or X has wrong feature count.
        """
        # Input validation
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)

        # Get likelihoods and find class with maximum likelihood
        likelihoods: np.ndarray = self.predict_probabilities(X)

        # Find the class with highest likelihood
        predicted_class_indices: np.ndarray = likelihoods.argmax(axis=1)
   
        # Map indices back to class labels
        predicted_labels: np.ndarray = np.array([
            self.int2label[idx].label for idx in predicted_class_indices
        ]).reshape(-1, 1)

        return predicted_labels