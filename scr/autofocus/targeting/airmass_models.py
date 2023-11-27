"""
Astronomical Airmass Calculation Module

This module provides functions for calculating astronomical airmass and related parameters.

Functions
---------
find_airmass_threshold_crossover(airmass_threshold=1.2, airmass_model)
    Find the zenith angle cutoff based on an airmass model and a threshold value.

zenith_angle(altitude)
    Calculate the zenith angle from the given altitude.

plane_parallel_atmosphere(zenith_angle)
    Calculate airmass in a plane-parallel atmosphere.

pickering_interpolative(zenith_angle)
    Calculate airmass using the Pickering interpolative formula.

rosenberg_interpolative(zenith_angle)
    Calculate airmass using the Rosenberg interpolative formula.

Example Usage
-------------
>>> import airmass_models

>>> cutoff = airmass_models.find_airmass_threshold_crossover(
    airmass_threshold=1.2, airmass_models.plane_parallel_atmosphere
)
>>> zenith = airmass_models.zenith_angle(altitude)

>>> airmass_pickering = airmass_models.pickering_interpolative(zenith)
>>> airmass_rosenberg = airmass_models.rosenberg_interpolative(zenith)


Sources
-------
https://en.wikipedia.org/wiki/Air_mass_(astronomy)
"""
from typing import Callable, Optional

import numpy as np


def plane_parallel_atmosphere(zenith_angle):
    """
    Calculate the airmass in a plane-parallel atmosphere model.

    Parameters
    ----------
    zenith_angle : float
        Zenith angle in radians.

    Returns
    -------
    float
        The airmass value for the given zenith angle.

    Examples
    --------
    >>> plane_parallel_atmosphere(0)
    1.0
    """
    airmass = 1 / np.cos(zenith_angle)
    return airmass


def pickering_interpolative(zenith_angle):
    """
    Calculate the airmass using the Pickering interpolative formula.

    Parameters
    ----------
    zenith_angle : float
        Zenith angle in radians.

    Returns
    -------
    float
        The airmass value calculated using the Pickering formula.

    Examples
    --------
    >>> pickering_interpolative(0)
    1.000000196171337
    """
    altitude_deg = (np.pi / 2 - zenith_angle) * 180 / np.pi
    return 1 / np.sin((altitude_deg + 244 / (165 + 47 * altitude_deg**1.1)) * np.pi / 180)


def rosenberg_interpolative(zenith_angle):
    """
    Calculate the airmass using the Rosenberg interpolative formula.

    Parameters
    ----------
    zenith_angle : float
        Zenith angle in radians.

    Returns
    -------
    float
        The airmass value calculated using the Rosenberg formula.

    Examples
    --------
    >>> rosenberg_interpolative(0)
    0.9999995824576546
    """
    return 1 / (np.cos(zenith_angle) + 0.025 * np.exp(-11 * np.cos(zenith_angle)))


def zenith_angle(altitude):
    """
    Calculate the zenith angle from the given altitude.

    Parameters
    ----------
    altitude : float
        Altitude angle in radians.

    Returns
    -------
    float
        Zenith angle in radians.

    Examples
    --------
    >>> zenith_angle(0) * 180/np.pi
    90.0
    """
    return np.pi / 2 - altitude


def find_airmass_threshold_crossover(
    airmass_threshold: Optional[float] = 1.2, airmass_model: Callable=plane_parallel_atmosphere
):
    """
    Find the zenith angle cutoff based on an airmass model and a threshold value.

    Parameters
    ----------
    airmass_model : Callable
        A callable function that takes zenith angles and returns airmass values.
    airmass_threshold : float, optional
        The threshold value, above which the airmass is considered too high.

    Returns
    -------
    float
        The zenith angle cutoff that corresponds to the airmass falling below the threshold.

    Notes
    -----
    This function assumes that airmass_model increases monotonically as a function of zenith angle.

    Examples
    --------
    >>> find_airmass_threshold_crossover(airmass_threshold=1.2, airmass_model=plane_parallel_atmosphere)
    0.5846852994181003
    """
    zenith_angles = np.linspace(0, 90, 181) * np.pi / 180
    last_angle = np.where(airmass_model(zenith_angles) < airmass_threshold)[0][-1]
    return zenith_angles[last_angle]
