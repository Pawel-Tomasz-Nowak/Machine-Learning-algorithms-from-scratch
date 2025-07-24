import numpy as np
from typing import Callable, Union


def validate_fraction(frac: float) -> None:
    """
    Validate that the fraction is in the interval (0, 1].

    Args:
        frac (float): Fraction to validate.

    Raises:
        AssertionError: If frac is not in (0, 1].
    """
    assert 0 < frac <= 1, "Fraction should be in (0; 1] interval"


def bootstrap_mean(
    f: Callable[[np.ndarray], Union[np.ndarray, float]],
    X: np.ndarray,
    boot_n: int,
    frac: float = 1.0
) -> Union[np.ndarray, float]:
    """
    Estimate the mean of a statistic using the bootstrap method.

    Args:
        f (Callable): Statistic function to estimate.
        X (np.ndarray): Dataset.
        boot_n (int): Number of bootstrap replications.
        frac (float): Fraction of rows for each bootstrap sample.

    Returns:
        Union[np.ndarray, float]: Bootstrap mean estimate of the statistic.
    """
    validate_fraction(frac)

    n: int = X.shape[0]
    boot_size: int = int(frac * n)
    boot_samples: np.ndarray = np.random.choice(n, [boot_n, boot_size], replace=True)
    estimates: np.ndarray = np.apply_along_axis(
        lambda idxs: f(X[idxs]), axis=1, arr=boot_samples
    )
    
    return np.mean(estimates, axis=0)


def bootstrap_se(
    f: Callable[[np.ndarray], Union[np.ndarray, float]],
    X: np.ndarray,
    boot_n: int,
    frac: float = 1.0
) -> Union[np.ndarray, float]:
    """
    Estimate the standard error of a statistic using the bootstrap method.

    Args:
        f (Callable): Statistic function to estimate.
        X (np.ndarray): Dataset.
        boot_n (int): Number of bootstrap replications.
        frac (float): Fraction of rows for each bootstrap sample.

    Returns:
        Union[np.ndarray, float]: Bootstrap standard error estimate of the statistic.
    """
    validate_fraction(frac)

    n: int = X.shape[0]
    boot_size: int = int(frac * n)
    boot_samples: np.ndarray = np.random.choice(n, [boot_n, boot_size], replace=True)
    estimates: np.ndarray = np.apply_along_axis(
        lambda idxs: f(X[idxs]), axis=1, arr=boot_samples
    )
    
    return np.std(estimates, axis=0, ddof=1)


def bootstrap_ci(
    f: Callable[[np.ndarray], Union[np.ndarray, float]],
    X: np.ndarray,
    boot_n: int,
    q1: float,
    q2: float,
    frac: float = 1.0
) -> Union[np.ndarray, float]:
    """
    Estimate confidence interval quantiles of a statistic using the bootstrap method.

    Args:
        f (Callable): Statistic function to estimate.
        X (np.ndarray): Dataset.
        boot_n (int): Number of bootstrap replications.
        q1 (float): Lower quantile for the confidence interval.
        q2 (float): Upper quantile for the confidence interval.
        frac (float): Fraction of rows for each bootstrap sample.

    Returns:
        Union[np.ndarray, float]: Lower and upper quantiles of the bootstrap statistic.
    """
    validate_fraction(frac)

    n: int = X.shape[0]
    boot_size: int = int(frac * n)
    boot_samples: np.ndarray = np.random.choice(n, [boot_n, boot_size], replace=True)
    estimates: np.ndarray = np.apply_along_axis(
        lambda idxs: f(X[idxs]), axis=1, arr=boot_samples
    )
    
    return np.percentile(estimates, [q1, q2], axis=0)