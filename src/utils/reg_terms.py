import numpy as np


class L1Regularization:
    """
    L1 regularization (Lasso penalty) class.
    
    Parameters:
        lambda_reg (float): Regularization strength parameter.
    """
    
    def __init__(self, lambda_reg: float = 1.0) -> None:
        """
        Initialize L1 regularization.
        
        Args:
            lambda_reg (float): Regularization strength parameter.
            
        Raises:
            AssertionError: If lambda_reg is not positive.
        """
        assert lambda_reg > 0, "Regularization parameter must be positive"
        self.lambda_reg: float = lambda_reg
    
    def compute_penalty(self, params: np.ndarray) -> float:
        """
        Compute L1 penalty for given parameters.
        
        Args:
            params (np.ndarray): Model parameters to regularize.
            
        Returns:
            float: L1 penalty term.
        """
        penalty: float = self.lambda_reg * np.sum(np.abs(params))
        return float(penalty)


class L2Regularization:
    """
    L2 regularization (Ridge penalty) class.
    
    Parameters:
        lambda_reg (float): Regularization strength parameter.
    """
    
    def __init__(self, lambda_reg: float = 1.0) -> None:
        """
        Initialize L2 regularization.
        
        Args:
            lambda_reg (float): Regularization strength parameter.
            
        Raises:
            AssertionError: If lambda_reg is not positive.
        """
        assert lambda_reg >= 0, "Regularization parameter must be positive"
        self.lambda_reg: float = lambda_reg
    
    def compute_penalty(self, params: np.ndarray) -> float:
        """
        Compute L2 penalty for given parameters.
        
        Args:
            params (np.ndarray): Model parameters to regularize.
            
        Returns:
            float: L2 penalty term.
        """
        penalty: float = self.lambda_reg * np.sum(params ** 2)
        return float(penalty)


class ElasticNetRegularization:
    """
    Elastic Net regularization (combination of L1 and L2 penalties) class.
    
    Parameters:
        lambda_reg1 (float): L1 regularization strength parameter.
        lambda_reg2 (float): L2 regularization strength parameter.
    """
    
    def __init__(self, lambda_reg1: float = 1.0, lambda_reg2: float = 1.0) -> None:
        """
        Initialize Elastic Net regularization.
        
        Args:
            lambda_reg1 (float): L1 regularization strength parameter.
            lambda_reg2 (float): L2 regularization strength parameter.
            
        Raises:
            AssertionError: If either regularization parameter is not positive.
        """
        assert lambda_reg1 > 0, "L1 regularization parameter must be positive"
        assert lambda_reg2 > 0, "L2 regularization parameter must be positive"
        
        self.l1_regularizer = L1Regularization(lambda_reg1)
        self.l2_regularizer = L2Regularization(lambda_reg2)
    
    def compute_penalty(self, params: np.ndarray) -> float:
        """
        Compute Elastic Net penalty for given parameters.
        
        Args:
            params (np.ndarray): Model parameters to regularize.
            
        Returns:
            float: Elastic Net penalty term.
        """
        l1_penalty: float = self.l1_regularizer.compute_penalty(params)
        l2_penalty: float = self.l2_regularizer.compute_penalty(params)
        
        return l1_penalty + l2_penalty