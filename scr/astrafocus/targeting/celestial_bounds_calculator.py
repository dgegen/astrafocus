from abc import ABC, abstractmethod

import numpy as np
from astropy import units as u
from astropy.coordinates import Longitude


class CelestialBoundsCalculatorInterface(ABC):
    def get_query(self, pole_tolerance=1):
        ra_bounds, dec_bounds = self.get_bounds(pole_tolerance=pole_tolerance)

        ra_query = self.get_ra_query(ra_bounds)
        dec_query = f"dec BETWEEN {dec_bounds[0]} AND {dec_bounds[1]}"

        return f"{dec_query} AND {ra_query}"

    @staticmethod
    def get_ra_query(ra_bounds):
        if np.diff(ra_bounds) > 0:
            return f"ra BETWEEN {ra_bounds[0]} AND {ra_bounds[1]}"
        else:
            return f"(ra BETWEEEN 0 AND {ra_bounds[0]} OR ra BETWEEN {ra_bounds[1]} AND 360)"

    def get_bounds(self, pole_tolerance=1):
        ra_bounds, dec_bounds = self.get_ra_dec_bounds_at_edges()

        # Avoid querying near the poles
        if self.is_near_poles(dec_bounds, pole_tolerance):
            ra_bounds = np.array([0.0, 360.0])
            dec_bounds = self.get_pole_dec_bounds(dec_bounds, pole_tolerance)

        ra_bounds = self.get_ra_bounds(ra_bounds, dec_bounds)

        return ra_bounds, dec_bounds

    @abstractmethod
    def get_ra_dec_bounds_at_edges(self) -> tuple:
        pass

    @abstractmethod
    def ra_bounds_at_equator(self) -> np.ndarray:
        pass

    def get_ra_bounds(self, ra_bounds, dec_bounds):
        """Create an SQLite-style query based on Right Ascension bounds."""
        # Calculate ra_bounds at the equator if it is crossed
        if self.crosses_equator(dec_bounds):
            ra_bounds = self.ra_bounds_at_equator()

        # Assure that they are sorted and in the range [0, 360)
        if not np.array_equal(np.array([0.0, 360.0]), ra_bounds):
            ra_bounds = np.sort(Longitude(ra_bounds * u.deg).deg)

        # Flip RA bounds if the field of view crosses the equator, e.g. for ra_bounds = [2, 356]
        # This will be used by get_ra_query to filter values between [356, 0] or [0, 2]
        # instead of [2, 356]
        if np.diff(ra_bounds)[0] > Longitude(np.diff(np.flip(ra_bounds)) * u.deg).deg[0]:
            ra_bounds = np.flip(ra_bounds)

        return ra_bounds

    def get_pole_dec_bounds(self, dec_bounds, pole_tolerance):
        """Create an SQLite-style query based on Declination bounds."""
        if np.abs(90 - dec_bounds[0]) < pole_tolerance:
            dec_bounds[0] = -90
            dec_bounds[1] = -90 + pole_tolerance
        elif np.abs(-90 - dec_bounds[1]) < pole_tolerance:
            dec_bounds[0] = 90 - pole_tolerance
            dec_bounds[1] = 90
        return dec_bounds

    @staticmethod
    def is_near_poles(dec_bounds, pole_tolerance):
        return np.any(np.abs(dec_bounds) > 90 - pole_tolerance)

    @staticmethod
    def crosses_equator(dec_bounds):
        return np.diff(np.sign(dec_bounds)) != 0


class CelestialBoundsCalculatorWCS(CelestialBoundsCalculatorInterface):
    """
    cbc = CelestialBoundsCalculatorWCS(wcs)
    cbc.get_query()

    CelestialBoundsCalculatorWCS(wcs).get_query()

    """

    def __init__(self, wcs):
        self.wcs = wcs

    def get_ra_dec_bounds_at_edges(self):
        bounds = self.wcs.pixel_to_world(
            [0, self.wcs.pixel_shape[0], self.wcs.pixel_shape[0], 0],
            [0, 0, self.wcs.pixel_shape[1], self.wcs.pixel_shape[1]],
        )
        ra_bounds = np.array([np.min(bounds.ra.deg), np.max(bounds.ra.deg)])
        dec_bounds = np.array([np.min(bounds.dec.deg), np.max(bounds.dec.deg)])
        return ra_bounds, dec_bounds

    def ra_bounds_at_equator(self):
        left_bound = self.wcs.pixel_to_world(
            np.zeros(self.wcs.pixel_shape[1]), np.arange(self.wcs.pixel_shape[1])
        )
        right_bound = self.wcs.pixel_to_world(
            np.full(self.wcs.pixel_shape[0], self.wcs.pixel_shape[1]),
            np.arange(self.wcs.pixel_shape[1]),
        )

        ra_bounds = np.array(
            [
                np.minimum(np.min(left_bound.ra.deg), np.min(right_bound.ra.deg)),
                np.maximum(np.max(left_bound.ra.deg), np.max(right_bound.ra.deg)),
            ]
        )
        return ra_bounds
