import numpy as np
import sys
import os

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.base_model import BaseModel
from optimizers.adam import AdamOptimizer
import tests.unit_tests as unitests
from utils.bootstrap import Bootstrap

from typing import Callable

class GaussianDyscriminer(BaseModel):



    def __init__(self,
                 
                 **bootstrap_kwargs: dict | None) -> None:
        
        self.bootstrap_kwargs = bootstrap_kwargs or {}



    def estimate_vector_mean(self, X:np.ndarray) -> np.ndarray:
        """
        Estimating the class vector mean of records


        X - objects of the class

        """
        
        # Define an estimation function
        def mean_vector_fun(X:np.ndarray) -> np.ndarray:
            """Mean vector estimation with bootstrap replications"""
            return X.mean(axis = 0)


        # Define a bootstrap for this mean vector
        bootstrap_vector_mean: Bootstrap = Bootstrap(mean_vector_fun, 
                                                     **self.bootstrap_kwargs)

        # Estimate the vector mean with boostrap
        mean_vector:np.ndarray = bootstrap_vector_mean.estimate(X)


        return mean_vector
    
    def estimate_covariance_matrix(self, X:np.ndarray) -> np.ndarray:
        """
        Estimating the covariance matrix

        """


        def covariance_matrix_fun(X:np.ndarray) -> np.ndarray:
            """A function to be passed to bootstrap estimator"""
       
            # Calculate the mean
            vec_mean: np.ndarray = X.mean(axis = 0).reshape(1, -1)


            # formula inside a sum 
            term: Callable = lambda x: np.transpose(x.reshape(1, -1) - vec_mean)@(x - vec_mean.reshape(1, -1))


            # Find the covariance matrix esitimation
            cov_mat: np.ndarray = np.apply_along_axis(func1d = term, axis = 1, arr = X).mean(axis = 0)
            return cov_mat
        

        bootstrap_cov_mat: Bootstrap = Bootstrap(covariance_matrix_fun, **self.bootstrap_kwargs)
        cov_mat = bootstrap_cov_mat.estimate(X)
        return cov_mat


    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        # Validating the input
        unitests.assert_2d_same_rows(X, y)

        # Save the dimensionality of X
        self.p = X.shape[1]

        # Unique classes:
        self.unique_classes: np.ndarray = np.unique(y)

        # Compute conditional vector means and covariance matrix for each class 
        for c in self.unique_classes:
            # All the records from c class
            idx_c: np.ndarray = y == c

            X_c = X[idx_c]


            vector_mean: np.ndarray = self.estimate_vector_mean(X_c)
            cov_matrix: np.ndarray = self.estimate_covariance_matrix(X_c)



        
    
        pass

    def predict(self, X:np.ndarray) -> np.ndarray:
        pass


n = 5
p = 2

X = np.random.randint(0, 5, 
size = [n, p])

piwo = GaussianDyscriminer()
print(piwo.estimate_vector_mean(X))


