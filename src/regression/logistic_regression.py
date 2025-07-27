import numpy as np
import sys
import os

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from typing import Callable, Optional

from core.base_model import BaseModel
from optimizers.adam import AdamOptimizer
import tests.unit_tests as unitests
from utils.bootstrap import bootstrap_mean




class BinaryLogisticRegression(BaseModel):
    """ Binary logistic regression"""


    def __init__(self, cut_off:float = 0.5) -> None:
        """Binary logistic regression
        
        Args:
            cut_off (float): A cut-off for probability (objects with probability > cut_off are assigned to class 1)
        """

        self.cut_off = cut_off
        self.is_fit: bool = False

  
    def _sigmoid_fun(self,b:np.ndarray, x:np.ndarray, y:int) -> float:
        p:float =  1/(1 + np.exp(-np.dot(b, x)))

        return p if y == 1 else 1-p
        
    def fit(self, X:np.ndarray, y_true:np.ndarray) -> None:
        unitests.assert_is_ndarray(X)
        unitests.assert_is_ndarray(y_true)
        unitests.assert_2d_same_rows(X, y_true)

        # Save the dimensionality of the input
        self.p = X.shape[1]

        # Add intercept and label column to X
        one_col: np.ndarray = np.ones((X.shape[0], 1), dtype=np.float32)
        X_exp: np.ndarray = np.concatenate([one_col, X, y_true], axis=1)  # expanded X for bootstrap estimation

        # Starting point for optimalization
        b0:np.ndarray = np.ones(shape = X_exp.shape[1]-1, dtype = np.float64())

        # Define a function for estimating parameters of the logistic regression
        def fun(X:np.ndarray) -> np.ndarray:
            
            # Declare an objective function
            L: Callable = lambda b: (
                (-1)*np.prod([self._sigmoid_fun(b, X[i, :-1], X[i, -1]) for i in range(X.shape[0])])
            )


            # Define an optimizer
            optimizer = AdamOptimizer(0.5)

            return optimizer.optimize(L, b0)
        

        self.coefficients_ = bootstrap_mean(fun, X_exp, 50)
        self.is_fit = True


    def _classify(self, x:np.ndarray) -> int:
        '''Classifies to a class given the computed probability'''

        # Compute the probability
        p: float = 1/(1 + np.exp(self.coefficients_[0] - np.dot(self.coefficients_[1:], x))) 

        return 1 if p > self.cut_off else 0       
        
        
    def predict(self, X:np.ndarray) -> np.ndarray:
        #Validate the input
        unitests.assert_fitted(self.is_fit)
        unitests.assert_feature_count(X, self.p)

        
        # Classify the input records
        y_pred: np.ndarray = np.apply_along_axis(
            lambda x: self._classify(x), 
            axis = 1, 
            arr = X
            ).reshape(-1,1)

        return y_pred



X: np.ndarray = np.array([[5,2,1],
                          [1,2,3],
                          [6,7,8]])

y:np.ndarray = np.array([[1],
                         [0]
                         ,[0]])

cos = BinaryLogisticRegression()
cos.fit(X, y)

print(cos.predict(X))


