"""
Robust Extremum Estimation Module

This module provides a framework for robustly estimating the minimum or maximum values in a set of
measurements using various techniques. It includes a base class and several subclasses,
each implementing a different extremum estimation technique.

Classes
-------
- `RobustExtremumEstimator`: Abstract base class providing methods for estimating extrema.
- `MedianFilterExtremumEstimation`: Estimation using a median filter for robustness.
- `LOWESSExtremumEstimator`: Estimation using Locally Weighted Scatterplot Smoothing (LOWESS).
- `SplineExtremumEstimator`: Estimation using Univariate Spline interpolation.
- `RBFExtremumEstimator`: Estimation using Radial Basis Function (RBF) interpolation.

Methods
-------
- `argmin(x, y, return_value=True)`: Returns the x-value corresponding to the minimum estimated y-value.
  If return_value is True, also returns the estimated y-value at the minimum.
- `argmax(x, y, return_value=True)`: Returns the x-value corresponding to the maximum estimated y-value.
  If return_value is True, also returns the estimated y-value at the maximum.
- `estimate_robust_signal(x, y)`: Abstract method to be implemented by subclasses for specific
  extremum estimation techniques. Returns the x and y values of a smoothed version of the curve.
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import scipy
import statsmodels.api as sm

from astrafocus.utils.logger import get_logger

logger = get_logger()


class RobustExtremumEstimator(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "name" not in cls.__dict__:
            name = cls.__name__.replace("ExtremumEstimator", "").replace("_", " ")
            # Add spaces before capital letters except if there is multiple capital letters in a row, then only add one space between the second last and last capital letter
            name = "".join(
                [
                    f" {c}"
                    if c.isupper()
                    and i > 0
                    and (
                        not name[i - 1].isupper()
                        or (i + 2 < len(name) and name[i - 1].isupper() and name[i + 1].islower())
                    )
                    else c
                    for i, c in enumerate(name)
                ]
            )
            # Separate numbers from letters
            name = "".join(
                [
                    f" {c}" if c.isdigit() and i > 0 and not name[i - 1].isdigit() else c
                    for i, c in enumerate(name)
                ]
            )

            cls.name = name.strip()

    def argmin(self, x: np.ndarray, y: np.ndarray, return_value=True) -> float | tuple[float, float]:
        """
        Returns the x-value corresponding to the minimum estimated noise-resistant y-value.

        Parameters
        ----------
        x : np.ndarray
            The input x values.
        y : np.ndarray
            The input y values.
        return_value : bool
            Whether to return the estimated y-value at the minimum. Defaults to True.

        Returns
        -------
        float | Tuple[float, float]
            The x-value corresponding to the minimum estimated noise-resistant y-value.
            If return_value is True, also returns the estimated y-value at the minimum.
        """
        x_prime, y_prime = self.estimate_robust_signal(x, y)
        index_min = np.argmin(y_prime)
        if return_value:
            return x_prime[index_min], y_prime[index_min]
        else:
            return x_prime[index_min]

    def argmax(self, x: np.ndarray, y: np.ndarray, return_value=True) -> float | tuple[float, float]:
        """
        Returns the x-value corresponding to the maximum estimated noise-resistant y-value.

        Parameters
        ----------
        x : np.ndarray
            The input x values.
        y : np.ndarray
            The input y values.
        return_value : bool
            Whether to return the estimated y-value at the maximum. Defaults to True.

        Returns
        -------
        float | Tuple[float, float]
            The x-value corresponding to the maximum estimated noise-resistant y-value.
            If return_value is True, also returns the estimated y-value at the maximum.
        """
        x_prime, y_prime = self.estimate_robust_signal(x, y)
        index_max = np.argmax(y_prime)
        if return_value:
            return x_prime[index_max], y_prime[index_max]
        else:
            return x_prime[index_max]

    @abstractmethod
    def estimate_robust_signal(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Abstract method to be implemented by subclasses for robust extremum signal estimation.
        """
        pass

    @staticmethod
    def sort(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sorts the x and y values by the x values."""
        sorted_indices = np.argsort(x)
        return x[sorted_indices], y[sorted_indices]


class MedianFilterExtremumEstimation(RobustExtremumEstimator):
    def __init__(self, size=10):
        self.size = size

    def estimate_robust_signal(self, x, y):
        """
        Estimates the robust signal using a median filter.

        Parameters
        ----------
        x : np.ndarray
            The input x values.
        y : np.ndarray
            The input y values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of x and y values of the smoothed curve.
        """
        estimated_values = scipy.ndimage.median_filter(y, size=self.size)
        return x, estimated_values


class LOWESSExtremumEstimator(RobustExtremumEstimator):
    def __init__(self, frac=0.5, it=3):
        self.frac = frac
        self.it = it

    def estimate_robust_signal(self, x, y):
        """
        Estimates the robust signal using Locally Weighted Scatterplot Smoothing (LOWESS).

        Parameters
        ----------
        x : np.ndarray
            The input x values.
        y : np.ndarray
            The input y values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of x and y values of the smoothed curve.
        """
        lowess = sm.nonparametric.lowess(endog=y, exog=x, frac=self.frac, it=self.it)
        return lowess[:, 0], lowess[:, 1]


class SplineExtremumEstimator(RobustExtremumEstimator):
    def __init__(self, k=2):
        self.k = k

    def estimate_robust_signal(self, x, y):
        """
        Estimates the robust signal using Univariate Spline interpolation.

        Parameters
        ----------
        x : np.ndarray
            The input x values.
        y : np.ndarray
            The input y values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of x and y values of the smoothed curve.
        """
        x_fine = np.linspace(x.min(), x.max(), 100)
        spline = scipy.interpolate.UnivariateSpline(x, y, k=self.k)

        estimated_values = spline(x_fine)
        return x_fine, estimated_values


class RBFExtremumEstimator(RobustExtremumEstimator):
    def __init__(self, kernel="linear", smoothing=20):
        self.kernel = kernel
        self.smoothing = smoothing

    def estimate_robust_signal(self, x, y):
        """
        Estimates the robust signal using Radial Basis Function (RBF) interpolation.

        Parameters
        ----------
        x : np.ndarray
            The input x values.
        y : np.ndarray
            The input y values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of x and y values of the smoothed curve.
        """
        x_fine = np.linspace(x.min(), x.max(), 100)

        rbf_interp = scipy.interpolate.RBFInterpolator(
            x.reshape(-1, 1), y, kernel=self.kernel, smoothing=self.smoothing
        )
        estimated_values = rbf_interp(x_fine.reshape(-1, 1))

        return x_fine, estimated_values


class ExtremumEstimators(Enum):
    """Enum mapping string keys to extremum estimator classes.

    Examples
    --------
    >>> from astrafocus.extremum_estimators import ExtremumEstimators
    >>> ExtremumEstimators.list()
    >>> ExtremumEstimators.lowess
    >>> ExtremumEstimators.from_name("spline")  # Get the class by fuzzy matching
    """

    median = MedianFilterExtremumEstimation
    lowess = LOWESSExtremumEstimator
    spline = SplineExtremumEstimator
    rbf = RBFExtremumEstimator

    @classmethod
    def from_name(cls, key: str):
        """Get an ExtremumEstimator by fuzzy matching the key. Returns lowess if not found."""
        key = key.lower()
        for member in cls:
            if key in member.name.lower():
                return member.value

        logger.warning(f"Extremum estimator '{key}' not found. Using 'lowess' instead.")
        return cls.lowess.value

    @classmethod
    def list(cls):
        """List all available extremum estimators."""
        return [member.name.lower() for member in cls]
