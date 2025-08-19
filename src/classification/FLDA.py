import numpy as np
import sys
import os
from typing import Dict
from scipy.linalg import fractional_matrix_power

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.base_model import BaseModel
import tests.unit_tests as unitests
from utils.bootstrap import Bootstrap


class FisherLinearDiscriminant(BaseModel):
    """
    Fisher Linear Discriminant Analysis (FLDA) for dimensionality reduction and classification.
    
    FLDA finds the optimal linear projection that maximizes the Fisher criterion:
    the ratio of between-class scatter to within-class scatter. This creates a lower-dimensional
    space where classes are maximally separated.
    
    The algorithm solves the generalized eigenvalue problem: S_B * w = λ * S_W * w
    where S_B is the between-class scatter matrix and S_W is the within-class scatter matrix.
    """

    def __init__(self, **bootstrap_kwargs) -> None:
        """
        Initialize the Fisher Linear Discriminant Analysis model.

        Args:
            **bootstrap_kwargs: Optional bootstrap parameters for robust estimation:
                - boot_n (int): Number of bootstrap samples (required)
                - frac (float): Fraction of data in each bootstrap sample (default: 1.0)
        """
        super().__init__()
        self.bootstrap_kwargs: Dict = bootstrap_kwargs or {}

    def _compute_class_means(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute mean vector for each class.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            y (np.ndarray): Label vector, shape (n_samples,).

        Returns:
            np.ndarray: Class means matrix, shape (n_classes, n_features).
        """
        class_means: np.ndarray = np.array([
            X[y == class_label].mean(axis=0) for class_label in self.unique_classes
        ])
        return class_means

    def _compute_class_priors(self, y: np.ndarray) -> np.ndarray:
        """
        Compute prior probability for each class.

        Args:
            y (np.ndarray): Label vector, shape (n_samples,).

        Returns:
            np.ndarray: Prior probabilities, shape (n_classes,).
        """
        n: int = y.shape[0]
        priors: np.ndarray = np.array([
            np.sum(y == class_label) / n for class_label in self.unique_classes
        ])
        return priors

    def _estimate_within_class_scatter_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute within-class scatter matrix S_W using bootstrap estimation.
        
        S_W = Σ(c=1 to C) Σ(x∈class_c) (x - μ_c)(x - μ_c)^T
        
        This matrix measures the total scatter of samples around their respective class means.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            y (np.ndarray): Label vector, shape (n_samples,).

        Returns:
            np.ndarray: Within-class scatter matrix, shape (n_features, n_features).
        """
        def within_class_scatter_function(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            """Bootstrap function for within-class scatter matrix computation."""
            n_features: int = X.shape[1]
            S_W: np.ndarray = np.zeros((n_features, n_features), dtype=np.float64)
            
            # Compute class means for current bootstrap sample
            class_means: np.ndarray = self._compute_class_means(X, y)
            
            # For each observation, add its contribution to S_W
            for i, sample in enumerate(X):
                label = y[i]
                class_idx: int = self.label2int[label]
                class_mean: np.ndarray = class_means[class_idx]
                
                # Centered observation vector
                centered_sample: np.ndarray = (sample - class_mean).reshape(-1, 1)
                
                # Add outer product to scatter matrix
                S_W += centered_sample @ centered_sample.T
            
            return S_W

        # Bootstrap estimation for robust parameter estimation
        bootstrap_estimator: Bootstrap = Bootstrap(within_class_scatter_function, **self.bootstrap_kwargs)
        within_class_matrix: np.ndarray = bootstrap_estimator.estimate(X, y)
        
        return within_class_matrix

    def _estimate_between_class_scatter_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute between-class scatter matrix S_B using bootstrap estimation.
        
        S_B = Σ(c=1 to C) π_c * (μ_c - μ)(μ_c - μ)^T
        
        where π_c is the prior probability of class c, μ_c is the class mean,
        and μ is the global mean.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            y (np.ndarray): Label vector, shape (n_samples,).

        Returns:
            np.ndarray: Between-class scatter matrix, shape (n_features, n_features).
        """
        def between_class_scatter_function(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            """Bootstrap function for between-class scatter matrix computation."""
            # Global mean vector across all samples
            global_mean: np.ndarray = np.mean(X, axis=0)
            
            # Class means matrix, shape (n_classes, n_features)
            class_means: np.ndarray = self._compute_class_means(X, y)
            
            # Center class means around global mean
            centered_means: np.ndarray = class_means - global_mean
            
            # Compute outer products for each centered class mean
            outer_products: np.ndarray = np.apply_along_axis(
                func1d=lambda x: x.reshape(-1, 1) @ x.reshape(-1, 1).T,
                axis=1,
                arr=centered_means
            )  # Shape: (n_classes, n_features, n_features)
            
            # Compute class priors and reshape for broadcasting
            class_priors: np.ndarray = self._compute_class_priors(y).reshape((self.C, 1, 1))
            
            # Weight outer products by class priors
            weighted_products: np.ndarray = outer_products * class_priors
            
            # Sum across classes to get final S_B matrix
            S_B: np.ndarray = weighted_products.sum(axis=0)
            
            return S_B

        # Bootstrap estimation for robust parameter estimation
        bootstrap_estimator: Bootstrap = Bootstrap(between_class_scatter_function, **self.bootstrap_kwargs)
        between_class_matrix: np.ndarray = bootstrap_estimator.estimate(X, y)
        
        return between_class_matrix

    def _compute_discriminant_transformation(self, S_W: np.ndarray, S_B: np.ndarray) -> None:
        """
        Compute the optimal discriminant transformation matrix using whitening approach.
        
        The method solves the generalized eigenvalue problem using whitening:
        1. Compute S_W^(-1/2) (whitening matrix)
        2. Form A = S_W^(-1/2) @ S_B @ S_W^(-1/2)
        3. Find K leading eigenvectors of A
        4. Transform back: W = S_W^(-1/2) @ U

        Args:
            S_W (np.ndarray): Within-class scatter matrix, shape (n_features, n_features).
            S_B (np.ndarray): Between-class scatter matrix, shape (n_features, n_features).
        """
        # Compute S_W^(-1/2) using matrix fractional power
        S_W_inv_sqrt: np.ndarray = fractional_matrix_power(S_W, -1/2)

        # Compute whitened matrix A = S_W^(-1/2) @ S_B @ S_W^(-1/2)
        A: np.ndarray = S_W_inv_sqrt @ S_B @ S_W_inv_sqrt

        # Find eigenvalues and eigenvectors of whitened matrix
        eigenvalues: np.ndarray
        eigenvectors: np.ndarray
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Sort eigenvalues and select K largest (leading eigenvectors)
        sorted_indices: np.ndarray = np.argsort(eigenvalues)
        leading_eigenvectors: np.ndarray = eigenvectors[:, sorted_indices[-self.K:]]

        # Transform eigenvectors back to original space
        self.W: np.ndarray = S_W_inv_sqrt @ leading_eigenvectors

    def fit(self, X: np.ndarray, y: np.ndarray, K: int) -> None:
        """
        Fit the Fisher Linear Discriminant Analysis model to training data.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).
            y (np.ndarray): Label vector, shape (n_samples,).
            K (int): Number of discriminant components. Must satisfy K ≤ C-1 where C is the number of classes.

        Raises:
            AssertionError: If input arrays are invalid or K exceeds maximum allowed value.
        """
        # Input validation
        unitests.assert_matrix_vector_match(X, y)
        
        # Store model parameters
        self.p: int = X.shape[1]  # Number of features
        self.unique_classes: np.ndarray = np.unique(y)
        self.C: int = len(self.unique_classes)  # Number of classes
        self.K: int = K  # Number of discriminant components
        
        # Validate dimensionality reduction parameter
        assert isinstance(K, int) and K <= self.C - 1, \
            f"Number of discriminant components K ({K}) must be ≤ {self.C-1} (number of classes - 1)"
        assert K > 0, f"K must be positive, got {K}"
        
        # Create label-to-index mapping for efficient lookups
        self.label2int: Dict = {label: idx for idx, label in enumerate(self.unique_classes)}

        # Compute scatter matrices using bootstrap estimation
        S_W: np.ndarray = self._estimate_within_class_scatter_matrix(X, y)
        S_B: np.ndarray = self._estimate_between_class_scatter_matrix(X, y)

        # Find optimal discriminant transformation matrix
        self._compute_discriminant_transformation(S_W, S_B)

        # Estimate class means with bootstrap for robustness
        bootstrap_estimator: Bootstrap = Bootstrap(self._compute_class_means, **self.bootstrap_kwargs)
        self.class_means: np.ndarray = bootstrap_estimator.estimate(X, y)

        # Mark model as fitted
        self.is_fit = True

        # Transform class means to discriminant space for classification
        self.class_means_transformed: np.ndarray = self.transform(self.class_means)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data to the K-dimensional discriminant subspace.

        Args:
            X (np.ndarray): Feature matrix to transform, shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data in discriminant subspace, shape (n_samples, K).

        Raises:
            AssertionError: If model is not fitted or X has wrong feature count.
        """
        # Validate model state and input
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)

        # Project data onto discriminant directions
        X_transformed: np.ndarray = X @ self.W

        return X_transformed

    def _predict_single(self, x_transformed: np.ndarray):
        """
        Predict class label for a single observation in discriminant space.
        
        Uses nearest centroid classification in the transformed space.

        Args:
            x_transformed (np.ndarray): Single observation in discriminant space.

        Returns:
            Predicted class label.
        """
        # Compute Euclidean distances to all class centroids in discriminant space
        distances: np.ndarray = np.linalg.norm(
            self.class_means_transformed - x_transformed, axis=1
        )
        
        # Find the index of the closest class centroid
        min_distance_idx: int = int(np.argmin(distances))

        # Return corresponding class label
        return self.unique_classes[min_distance_idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data using nearest centroid classification.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).

        Raises:
            AssertionError: If model is not fitted or X has wrong feature count.
        """
        # Validate model state and input
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)

        # Transform data to discriminant subspace
        X_transformed: np.ndarray = self.transform(X)

        # Predict class for each transformed observation
        y_predicted: np.ndarray = np.apply_along_axis(
            func1d=self._predict_single,
            axis=1,
            arr=X_transformed
        )
        
        return y_predicted