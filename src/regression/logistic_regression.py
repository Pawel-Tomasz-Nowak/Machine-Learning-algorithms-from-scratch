import numpy as np
import sys
import os

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Callable, Optional, Union

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
        optimizer:Callable = AdamOptimizer,
        optimizer_kwargs: Optional[dict] = None
    ) -> None:
        """
        Initialize binary logistic regression classifier.

        Args:
            cut_off (float): Decision threshold for probability classification 
                           (objects with probability > cut_off are assigned to class 1).
            bootstrap_kwargs (Optional[dict]): Parameters for Bootstrap class:
                - boot_n (int): Number of bootstrap replications
                - frac (float): Fraction of rows for each bootstrap sample (0 < frac <= 1)
            optimizer (Callable): Optimizer class for maximizing likelihood function.
                                 Depends on the chosen optimizer, but gradient-based optimizers
                                 typically require:
                                 - lr (float): Learning rate (required for all gradient-based)
                                 - g_tol (float): Convergence tolerance for gradient norm
                                 - h (float): Step size for numerical differentiation
                                 Additional optimizer-specific parameters:
                                 * Adam: beta1, beta2, eps
                                 * AdaGrad: eps  
                                 * GDM/NAG: beta
                                 * RMSprop: beta, eps
            optimizer_kwargs (Optional[dict]): Parameters passed to the optimizer constructor.
        """
        super().__init__()
        self.cut_off: float = cut_off

        self.__optimizer_kwargs: dict = optimizer_kwargs or {}
        self.__bootstrap_kwargs: dict = bootstrap_kwargs or {}

        self.optimizer = optimizer(**self.__optimizer_kwargs)

    def _sigmoid_fun(self, b: np.ndarray, x: np.ndarray, y: int) -> float:
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
        # Input validation
        unitests.assert_2d_same_rows(X, y_true)
        unitests.assert_feature_count(y_true, 1)

        # Save feature dimensionality
        self.p: int = X.shape[1]

        # Add intercept column to feature matrix
        one_col: np.ndarray = np.ones((X.shape[0], 1), dtype=np.float32)
        X_exp: np.ndarray = np.concatenate([one_col, X], axis=1)

        # Initial parameter vector for optimization
        b0: np.ndarray = np.ones(shape=(X_exp.shape[1], 1), dtype=np.float64)

        def likelihood_estimation_function(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            """
            Estimate logistic regression parameters via likelihood maximization.

            Args:
                X (np.ndarray): Expanded feature matrix with intercept.
                y (np.ndarray): Binary target vector.

            Returns:
                np.ndarray: Optimized coefficient vector.
            """
            # Define negative log-likelihood objective function
            negative_log_likelihood: Callable = lambda b: (
                (-1) * np.sum([
                    np.log(self._sigmoid_fun(b, X[i], y[i, 0]))
                    for i in range(X.shape[0])
                ])
            )

            return self.optimizer.optimize(negative_log_likelihood, b0)

        # Bootstrap parameter estimation for stability
        bootstrap_estimator = Bootstrap(likelihood_estimation_function, **self.__bootstrap_kwargs)
        self.coefficients_: np.ndarray = bootstrap_estimator.estimate(X_exp, y_true)
        self.is_fit = True

    def _compute_likelihood(self, x:np.ndarray) -> float:
        lin_comb: float = self.coefficients_[0] + (x @ self.coefficients_[1:])[0]
        
        # Compute classification probability
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
        # Compute classification probability
        p: float = self._compute_likelihood(x)

        return 1 if p > self.cut_off else 0

    def predict_probabilities(self, X:np.ndarray) ->np.ndarray:
        """
        For a given input data, return the probabiltiies of each record being classified as 1

        """
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)

        probs:np.ndarray = np.array(
            [self._compute_likelihood(x) for x in X]
        )


        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted binary class labels of shape (n_samples, 1).
        """
        # Input validation
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)

        y_pred: np.ndarray = np.array([
            self._classify(X[i]) for i in range(X.shape[0])
        ]).reshape(-1, 1)

        return y_pred
    
class MultinomialLogisticRegression(BaseModel):


    def __init__(
        self,
        bootstrap_kwargs: Optional[dict] = None,
        optimizer: Callable = AdamOptimizer,
        optimizer_kwargs: Optional[dict] = None
    ) -> None:
        """
        Initialize multinomial logistic regression classifier.

        Args:
            bootstrap_kwargs (Optional[dict]): Parameters for Bootstrap class:
                - boot_n (int): Number of bootstrap replications
                - frac (float): Fraction of rows for each bootstrap sample (0 < frac <= 1)
            optimizer (Callable): Optimizer class for maximizing likelihood function.
                                Depends on the chosen optimizer, but gradient-based optimizers
                                typically require:
                                - lr (float): Learning rate (required for all gradient-based)
                                - g_tol (float): Convergence tolerance for gradient norm
                                - h (float): Step size for numerical differentiation
                                Additional optimizer-specific parameters:
                                * Adam: beta1, beta2, eps
                                * AdaGrad: eps  
                                * GDM/NAG: beta
                                * RMSprop: beta, eps
            optimizer_kwargs (Optional[dict]): Parameters passed to the optimizer constructor.
        """
        super().__init__()

        self.__optimizer_kwargs: dict = optimizer_kwargs or {}
        self.__bootstrap_kwargs: dict = bootstrap_kwargs or {}

        self.optimizer:Callable = optimizer

        self.categories: np.ndarray = np.array([])
        
        self.int2reg: dict[int, BinaryLogisticRegression] = {} # Mapper from category index to its corresponding logistic regression


    def validate_input(self, X:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Validates the input X and y arrays and reshape y to 2D if necessary
        Specifically checks if X is 2D and y is 1d or 2d.
        

        """
        if isinstance(y, np.ndarray):
            # Check if the number of rows match depending on the shape of y
            if y.ndim == 1:
                unitests.assert_matrix_vector_match(X, y)
                y:np.ndarray = y.reshape(-1, 1) # If y is 1D, reshape it to 2D

            elif y.ndim == 2:
                unitests.assert_feature_count(y, 1)
                unitests.assert_2d_same_rows(X, y)
            else:
                raise ValueError(f"The shape of y should be either 1 or 2. Got {y.ndim}")
            
            # Save the original output shape
            self.output_shape = y.shape

        else:
            raise TypeError(f"Got unexpected datatype for y: {type(y)}. Expected: numpy.ndarray")
        
 
        return y

    def fit(self, X: np.ndarray, y_true: np.ndarray,
            categories: Union[list[np.ndarray], str] = 'auto') -> None:
        y = self.validate_input(X, y_true) # y is 2d array. (n_samples, 1)
        # Save the dimensionality of the input features
        self.p: int = X.shape[1]

        # Declare an instance of OneHotEncoder
        self.OHE_inst: OHE = OHE()

 
        # Transform the output
        y_ohe: np.ndarray = self.OHE_inst.fit_transform(X = y, categories = categories)[0]
        self.k: int = y_ohe.shape[1] # Find the cardinality of y

        for cat_id in range(self.k):
            print(f"elo {cat_id}")
            dummy_var: np.ndarray = y_ohe[:, cat_id].reshape(-1, 1) # Dummy variable corresponding to a specific category
   

            # Declare the logistic regression
            BinLogReg = BinaryLogisticRegression(bootstrap_kwargs = self.__bootstrap_kwargs,
                                                 optimizer = self.optimizer,
                                                 optimizer_kwargs=self.__optimizer_kwargs)
            
            
            BinLogReg.fit(X, dummy_var) # Fit a binary logistic regression for the category

            self.int2reg[cat_id] = BinLogReg # Save the regression for later prediction

        self.is_fit = True # Mark the model as fitted

    def _compute_likelihoods(self, x:np.ndarray) -> np.ndarray:
        '''
        Computes the probabilities of belonging to each category for a single observation
        '''
        # Initialize the array for probabilties
        probs: np.ndarray = np.zeros(shape = self.k, 
                                     dtype = np.float64)
        
        for reg_id, reg in self.int2reg.items():
            prob: float = reg._compute_likelihood(x) 

            probs[reg_id] = prob


        return probs
    
    def _classify(self, x:np.ndarray):
        '''
        Predicts a single observation
        '''

        probs: np.ndarray = self._compute_likelihoods(x)

        # Find the category index with the highest likelihood
        i:int = int(np.argmax(probs))

        cat = self.OHE_inst.feature_encoders_[0].unique_categories_[i]
        
        return cat
        


    def get_probabilities(self, X:np.ndarray) -> np.ndarray:
        '''
        For each observation of X, it calculates the likelihoods of belonging to each found category
        '''
        # Check if the model is fitted
        unitests.assert_fitted(self.is_fit)
        # Validate the dimensionality of X
        unitests.assert_feature_count(X, self.p)

        n: int = X.shape[0] # Number of records

        probs = np.array( [self._compute_likelihoods(X[i]) for i in range(n)            ])

        return probs


    def predict(self, X:np.ndarray) -> np.ndarray:

        """
        Predicts multiple observations
        """
        # Check if the model is fitted
        unitests.assert_fitted(self.is_fit)
        # Validate the dimensionality of X
        unitests.assert_feature_count(X, self.p)

        n: int = X.shape[0]

        y_pred: np.ndarray = np.array([ self._classify(X[i]) for i in range(n)]).reshape(self.output_shape) # Predict the labels and reshape the results
      
        return y_pred

