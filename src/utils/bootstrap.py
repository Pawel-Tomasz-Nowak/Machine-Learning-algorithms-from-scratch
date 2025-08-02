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
        
        self.f: Callable = f
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
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        statistics: Callable = np.mean,
        **statistics_params
    ):
        """
        Estimate the specified statistic using bootstrap resampling.

        Args:
            X (np.ndarray, optional): Primary dataset to resample from.
            y (np.ndarray, optional): Secondary dataset (e.g., labels) to resample from.
            statistics (Callable): Statistic to compute on bootstrap estimates.
            **statistics_params: Additional parameters for the statistics function.

        Returns:
            Bootstrap estimate of the statistic.
            
        Raises:
            AssertionError: If both X and y are None, or if they have different lengths.
        """
        # Validate that at least one dataset is provided
        assert not (X is None and y is None), "X and y can't be None at the same time"
        
        # Determine primary dataset and sample size
        if X is not None and y is not None:
            # Both datasets provided - validate they have same length
            unit_tests.assert_2d_same_rows(X, y)
            n: int = X.shape[0]
        elif X is not None:
            # Only X provided
            n: int = X.shape[0]
        else:  # y is not None and X is None
            # Only y provided
            n: int = y.shape[0]
        
        # Calculate bootstrap sample size
        boot_size: int = int(self.frac * n)
        
        # Generate bootstrap sample indices
        boot_samples: np.ndarray = np.random.choice(
            n, [self.boot_n, boot_size], replace=True
        )
        
        # Compute statistic for each bootstrap sample based on available data
        if X is not None and y is not None:
            # Both X and y available - pass both to function
            estimates: np.ndarray = np.apply_along_axis(
                lambda idxs: self.f(X[idxs], y[idxs]), axis=1, arr=boot_samples
            )
        elif X is not None and y is None:
            # Only X available - pass only X to function
            estimates: np.ndarray = np.apply_along_axis(
                lambda idxs: self.f(X[idxs]), axis=1, arr=boot_samples
            )
        else:  # X is None and y is not None
            # Only y available - pass only y to function
            estimates: np.ndarray = np.apply_along_axis(
                lambda idxs: self.f(y[idxs]), axis=1, arr=boot_samples
            )
        
        # Return the specified statistic of the bootstrap estimates
        return statistics(estimates, axis=0, **statistics_params)