import numpy as np
import sys
import os
from typing import Optional, Union, Dict, List, Set

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tests.unit_tests as unitests


class CategoricalDummyEncoder:
    """
    One-hot encoder for a single categorical feature.
    
    Transforms a categorical feature into multiple binary dummy variables,
    where each dummy variable represents one category. This is also known
    as one-hot encoding for a single feature.
    
    Example:
        >>> encoder = CategoricalDummyEncoder()
        >>> encoder.fit(np.array(['cat', 'dog', 'cat', 'bird']))
        >>> encoded = encoder.transform(np.array(['cat', 'bird']))
        >>> print(encoded)  # Shape: (2, 3) for ['bird', 'cat', 'dog']
    """

    def __init__(self, categories: Union[np.ndarray, str] = 'auto') -> None:
        """
        Initialize the categorical dummy variable encoder.

        Args:
            categories (Union[np.ndarray, str]): Categories to encode. If 'auto',
                                               categories will be inferred from training data.
        """
        self.categories: Union[np.ndarray, str] = categories
        self.is_fit: bool = False
        
        # Fitted attributes (initialized during fit)
        self.unique_categories_: np.ndarray = np.array([])
        self.unique_categories_set_: Set = set()

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the encoder to the categorical data.
        
        Learns the unique categories present in the training data.
        If categories were specified during initialization, validates that
        all categories in X are in the predefined set.

        Args:
            X (np.ndarray): 1D array of categorical values, shape (n_samples,).

        Raises:
            AssertionError: If X is not 1D array.
        """
        # Validate input dimensions
        unitests.assert_ndim(X, 1)

        # Learn categories from data if auto mode
        if self.categories == 'auto':
            self.unique_categories_ = np.unique(X)
            self.unique_categories_set_ = set(self.unique_categories_)
        else:
            # Use predefined categories
            self.unique_categories_ = np.asarray(self.categories)
            self.unique_categories_set_ = set(self.unique_categories_)

        # Mark as fitted
        self.is_fit = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform categorical data to dummy variables (one-hot encoding).
        
        Each category becomes a binary column. The order of columns corresponds
        to the sorted order of unique categories found during fitting.

        Args:
            X (np.ndarray): 1D array of categorical values to transform, shape (n_samples,).

        Returns:
            np.ndarray: Binary matrix of dummy variables, shape (n_samples, n_categories).
                       Each row represents one sample, each column represents one category.

        Raises:
            AssertionError: If encoder is not fitted, X is not 1D, or X contains unknown categories.
        """
        # Validate input and model state
        unitests.assert_ndim(X, 1)
        unitests.assert_fitted(self.is_fit)
        
        # Check for unknown categories
        X_categories_set: Set = set(X)
        unknown_categories = X_categories_set.difference(self.unique_categories_set_)
        assert X_categories_set.issubset(self.unique_categories_set_), \
            f'Input data contains unknown categories: {unknown_categories}'
        
        # Create dummy variables matrix
        # Broadcasting: (n_samples, 1) == (1, n_categories) -> (n_samples, n_categories)
        dummy_matrix: np.ndarray = np.array(
            X.reshape(-1, 1) == self.unique_categories_,
            dtype=np.uint8
        )

        return dummy_matrix
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the encoder and transform the data in one step.
        
        Convenience method that combines fitting and transformation for efficient workflow.
        Equivalent to calling fit() followed by transform(), but returns the transformed data
        directly.
    
        Args:
            X (np.ndarray): 1D array of categorical values, shape (n_samples,).
    
        Returns:
            np.ndarray: Binary matrix of dummy variables, shape (n_samples, n_categories).
                       Each row represents one sample, each column represents one category.
    
        Raises:
            AssertionError: If X is not 1D array.
            
        Example:
            >>> encoder = CategoricalDummyEncoder()
            >>> dummy_matrix = encoder.fit_transform(np.array(['cat', 'dog', 'cat', 'bird']))
            >>> # Equivalent to: encoder.fit(X); encoder.transform(X)
            >>> print(dummy_matrix.shape)  # (4, 3) for ['bird', 'cat', 'dog']
        """
        # Fit the encoder to learn categories
        self.fit(X)
        
        # Transform the data to dummy variables
        dummy_matrix: np.ndarray = self.transform(X)
    
        return dummy_matrix


    

class OneHotEncoder:
    """
    Multi-feature one-hot encoder for categorical data.
    
    Transforms multiple categorical features into binary dummy variables.
    Each categorical feature is encoded independently using the CategoricalDummyEncoder.
    
    Example:
        >>> # Dataset with 2 categorical features
        >>> X = np.array([['cat', 'small'], 
        ...               ['dog', 'large'],
        ...               ['cat', 'medium']])
        >>> encoder = OneHotEncoder()
        >>> encoder.fit(X)
        >>> encoded_features = encoder.transform(X)
        >>> # Returns list of 2 arrays: one for 'animal' feature, one for 'size' feature
    """

    def __init__(self) -> None:
        """Initialize the multi-feature one-hot encoder."""
        self.is_fit: bool = False
        
        # Fitted attributes (initialized during fit)
        self.n_features_: int = 0
        self.feature_encoders_: List[CategoricalDummyEncoder] = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            categories: Union[list[np.ndarray], str] = 'auto') -> None:
        """
        Fit the one-hot encoder to categorical data.
        
        Creates and fits a separate CategoricalDummyEncoder for each feature column.
        The y parameter is included for compatibility with sklearn-style transformers
        but is not used in the fitting process.

        Args:
            X (np.ndarray): Categorical feature matrix, shape (n_samples, n_features).
                          Each column represents a different categorical feature.
            y (Optional[np.ndarray]): Target values (ignored, present for consistency)
            categories: Union[list[np.ndarray], str]: Precomputed categories for each categorical feature

        Raises:
            AssertionError: If X is not 2D array.
        """
        # Validate input dimensions
        unitests.assert_ndim(X, 2)

        # Store number of features
        self.n_features_ = X.shape[1]

        # Initialize list to store encoders for each feature
        self.feature_encoders_ = []

        # Fit encoder for each categorical feature column
        for feature_idx in range(self.n_features_):
            feature_column: np.ndarray = X[:, feature_idx]

            # Create and fit encoder for current feature
            if categories == 'auto':
                feature_encoder = CategoricalDummyEncoder()
            elif isinstance(categories, list) and isinstance(categories[feature_idx], np.ndarray):
                feature_encoder = CategoricalDummyEncoder(categories[feature_idx])
            else:
                raise TypeError(f'''Unexpected datatype for list of categories. 
                                Expected list of ndarrays, got {type(categories)} of {type(categories[feature_idx])}''')
            
            feature_encoder.fit(feature_column)

            # Store fitted encoder
            self.feature_encoders_.append(feature_encoder)

        # Mark as fitted
        self.is_fit = True

    def transform(self, X: np.ndarray) -> Union[List[np.ndarray], np.ndarray]:
        """
        Transform categorical features to dummy variables.
        
        Each feature column is independently transformed to its one-hot representation.
        Returns a list of arrays, where each array contains the dummy variables
        for one original feature.

        Args:
            X (np.ndarray): Categorical feature matrix to transform, shape (n_samples, n_features).

        Returns:
            List[np.ndarray] or np.ndarray: List of dummy variable matrices. Each element is a 2D array
                            with shape (n_samples, n_categories_for_feature_i), where
                            n_categories_for_feature_i is the number of unique categories
                            in the i-th feature.
                            When the input X has only one feature, returns a single 2D aray with shape (n_samples, n_categories_for_feature)

        Raises:
            AssertionError: If encoder is not fitted or X has wrong number of features.
            
        Example:
            >>> # If X has 2 features with 3 and 2 categories respectively:
            >>> # Returns [array(shape=(n_samples, 3)), array(shape=(n_samples, 2))]
        """
        # Validate model state and input
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.n_features_)

        # Transform each feature independently
        transformed_features: List[np.ndarray] = []
        
        for feature_idx, feature_encoder in enumerate(self.feature_encoders_):
            # Extract feature column and transform to dummy variables
            feature_column: np.ndarray = X[:, feature_idx]
            feature_dummies: np.ndarray = feature_encoder.transform(feature_column)
            
            # Add to results
            transformed_features.append(feature_dummies)

        if len(transformed_features) == 1:
            return transformed_features[0]
        
        return transformed_features
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Union[List[np.ndarray], np.ndarray]:
        """
        Fit the encoder and transform the data in one step.
        
        Convenience method that combines fitting and transformation for efficient workflow.
        Equivalent to calling fit() followed by transform(), but returns the transformed data
        directly. The y parameter is included for sklearn API compatibility.
    
        Args:
            X (np.ndarray): Categorical feature matrix, shape (n_samples, n_features).
                          Each column represents a different categorical feature.
            y (Optional[np.ndarray]): Target values (ignored, present for API compatibility).
    
        Returns:
            List[np.ndarray] or np.ndarray: List of dummy variable matrices. Each element is a 2D array
                            with shape (n_samples, n_categories_for_feature_i), where
                            n_categories_for_feature_i is the number of unique categories
                            in the i-th feature.
                            When the input X has only one feature, returns a single 2D aray with shape (n_samples, n_categories_for_feature)
    
        Raises:
            AssertionError: If X is not 2D array.
            
        Example:
            >>> encoder = OneHotEncoder()
            >>> transformed_features = encoder.fit_transform(X_train)
            >>> # Equivalent to: encoder.fit(X_train); encoder.transform(X_train)
        """
        # Always fit first (even if already fitted, to handle new data patterns)
        self.fit(X, y)
        
        # Transform the fitted data
        transformed_features: Union[List[np.ndarray], np.ndarray] = self.transform(X)

        return transformed_features