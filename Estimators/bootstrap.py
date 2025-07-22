import numpy as np
from typing import Callable


def bootstrap_mean(f: Callable[[np.ndarray], np.ndarray], X: np.ndarray, boot_n:int, frac:float = 1) -> np.ndarray:
    """
    Performs bootstrap mean estimation of `f` based on `X` array
    By default, all bootstrap samples are of sized n, where n = X.shape[0], however this could be any fraction denoted by `frac`,
    of n

    Args:
        f (Callable[[np.ndarray], np.ndarray]): a statistics to be estimated
        X (np.ndarray): a dataset
        boot_n (int): a number of bootstrap replications
        frac (float): a fraction of rows determining the size of each bootstrap sample
    """
    n: int = X.shape[0] # Number of observations

    boot_size: int = int(frac * n) # A size of each bootstrap replication


    boot_samples:np.ndarray = np.random.choice(n, [boot_n, boot_size], replace = True)

    estimates: np.ndarray = np.apply_along_axis(lambda idxs:    f(X[idxs]), axis = 1, arr = boot_samples)


    return np.mean(estimates, axis = 0)


def bootstrap_se(f: Callable[[np.ndarray], np.ndarray], X: np.ndarray, boot_n:int, frac:float = 1) -> np.ndarray:
    """
    Performs bootstrap standard error estimation of `f` based on `X` array
    By default, all bootstrap samples are of sized n, where n = X.shape[0], however this could be any fraction denoted by `frac`,
    of n

    Args:
        f (Callable[[np.ndarray], np.ndarray]): a statistics to be estimated
        X (np.ndarray): a dataset
        boot_n (int): a number of bootstrap replications
        frac (float): a fraction of rows determining the size of each bootstrap sample
    """
    n: int = X.shape[0] # Number of observations

    boot_size: int = int(frac * n) # A size of each bootstrap replication


    boot_samples:np.ndarray = np.random.choice(n, [boot_n, boot_size], replace = True)

    estimates: np.ndarray = np.apply_along_axis(lambda idxs:    f(X[idxs]), axis = 1, arr = boot_samples)


    return np.std(estimates, axis=0, ddof=1)


def bootstrap_ci(f: Callable[[np.ndarray], np.ndarray], X: np.ndarray, boot_n:int, q1:float, q2:float, frac:float = 1) -> np.ndarray:
    """
    Performs bootstrap quantile estimation of `f` based on `X` array for confidence interval
    By default, all bootstrap samples are of sized n, where n = X.shape[0], however this could be any fraction denoted by `frac`,
    of n

    Args:
        f (Callable[[np.ndarray], np.ndarray]): a statistics to be estimated
        X (np.ndarray): a dataset
        boot_n (int): a number of bootstrap replications
        q1, q2 (float): a quantiles for the confidence interval
        frac (float): a fraction of rows determining the size of each bootstrap sample
    """
    n: int = X.shape[0] # Number of observations

    boot_size: int = int(frac * n) # A size of each bootstrap replication


    boot_samples:np.ndarray = np.random.choice(n, [boot_n, boot_size], replace = True)

    estimates: np.ndarray = np.apply_along_axis(lambda idxs:    f(X[idxs]), axis = 1, arr = boot_samples)


    return np.percentile(estimates, [q1, q2], axis=0)