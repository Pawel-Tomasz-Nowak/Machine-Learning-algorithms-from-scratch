import numpy as np
import sys
import os
from typing import Callable, Union

# Add the 'src' directory to the system path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "..")))

from tests import unit_tests


class Bootstrap:
    """
    Bootstrap statistical estimation class for computing statistics with resampling.

    Parameters:
        f (Callable): Statistic function to estimate.
        boot_n (int): Number of bootstrap replications.
        frac (float): Fraction of rows for each bootstrap sample.
    """

    def __init__(
        self,
        f: Callable,
        boot_n: int = 10,
        frac: float = 1.0
    ) -> None:
        """
        Initialize the bootstrap estimator.

        Args:
            f (Callable): Statistic function to estimate.
            boot_n (int): Number of bootstrap replications.
            frac (float): Fraction of rows for each bootstrap sample.
        """
        self.validate_fraction(frac)
        
        self.f: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]] = f
        self.boot_n: int = boot_n
        self.frac: float = frac

    def validate_fraction(self, frac: float) -> None:
        """
        Validate that the fraction is in the interval (0, 1].

        Args:
            frac (float): Fraction to validate.

        Raises:
            AssertionError: If frac is not in (0, 1].
        """
        assert 0 < frac <= 1, "Fraction should be in (0; 1] interval"

    def estimate(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        statistics: Callable = np.mean,
        **statistics_params
    ) -> np.ndarray:
        """
        Estimate the specified statistic using bootstrap resampling.

        Args:
            X (np.ndarray): Dataset to resample from.
            y (np.ndarray, optional): An optional label vector with the length of X.
            statistics (Callable): Statistic to compute on bootstrap estimates.
            **statistics_params: Additional parameters for the statistics function.

        Returns:
            np.ndarray: Bootstrap estimate of the statistic.
        """
        if y is not None:
            unit_tests.assert_2d_same_rows(X, y)
        else:
            y = np.zeros(shape=(X.shape[0], 1), dtype=np.uint8)

        n: int = X.shape[0]
        boot_size: int = int(self.frac * n)
        
        # Generate bootstrap sample indices
        boot_samples: np.ndarray = np.random.choice(
            n, [self.boot_n, boot_size], replace=True
        )
        
        # Compute statistic for each bootstrap sample
        estimates: np.ndarray = np.apply_along_axis(
            lambda idxs: self.f(X[idxs], y[idxs]), axis=1, arr=boot_samples
        )
        
        # Return the specified statistic of the bootstrap estimates
        return statistics(estimates, axis=0, **statistics_params)