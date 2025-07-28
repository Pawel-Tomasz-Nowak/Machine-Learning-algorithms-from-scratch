import numpy as np
from typing import Callable, Union


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
        f: Callable[[np.ndarray], Union[np.ndarray, float]],
        boot_n: int,
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
        
        self.f: Callable[[np.ndarray], Union[np.ndarray, float]] = f
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
        statistics: Callable = np.mean,
        **statistics_params
    ) -> Union[np.ndarray, float]:
        """
        Estimate the specified statistic using bootstrap resampling.

        Args:
            X (np.ndarray): Dataset to resample from.
            statistics (Callable): Statistic to compute on bootstrap estimates.
            **statistics_params: Additional parameters for the statistics function.

        Returns:
            Union[np.ndarray, float]: Bootstrap estimate of the statistic.
        """
        n: int = X.shape[0]
        boot_size: int = int(self.frac * n)
        
        # Generate bootstrap sample indices
        boot_samples: np.ndarray = np.random.choice(
            n, [self.boot_n, boot_size], replace=True
        )
        
        # Compute statistic for each bootstrap sample
        estimates: np.ndarray = np.apply_along_axis(
            lambda idxs: self.f(X[idxs]), axis=1, arr=boot_samples
        )
        
        # Return the specified statistic of the bootstrap estimates
        return statistics(estimates, axis=0, **statistics_params)