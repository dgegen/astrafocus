from typing import Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from astrafocus.interface.telescope_specs import TelescopeSpecs


class TangentialPlaneProjector:
    """
    Examples
    --------
    central_star = central_star
    centre = SkyCoord(ra=central_star.ra, dec=central_star.dec, unit="deg", frame="icrs")

    centre = SkyCoord(ra=0, dec=0, unit="deg", frame="icrs")
    wcs = create_basic_wcs(center_ra=0, center_dec=-45, pixel_scale_arcsec=0.35)
    tangential_plane_projector = TangentialPlaneProjector(znqr, wcs)
    tangential_plane_projector.project(centre)

    """

    def __init__(self, df, wcs, mask_first=False) -> None:
        self.df = df
        self.wcs = wcs
        self.sky_coords = SkyCoord(
            ra=self.df.ra.to_numpy() * u.deg, dec=self.df.dec.to_numpy() * u.deg
        )

        self.mask_first = mask_first
        
        self.x_max = np.ceil(wcs.pixel_shape[0])
        self.y_max = np.ceil(wcs.pixel_shape[1])

    def project_on_ccd(self, centre: SkyCoord):
        pixel_coords = self.project(centre)
        filtered_coords = self.filter_coordinates_on_ccd(pixel_coords)

        return pixel_coords[filtered_coords]

    def num_stars_on_ccd(self, centre: SkyCoord):
        pixel_coords = self.project(centre)
        return np.sum(self.filter_coordinates_on_ccd(pixel_coords))

    def project(self, centre: SkyCoord):
        self.wcs.wcs.crval = [centre.ra.deg, centre.dec.deg]

        if self.mask_first:
            mask = self.get_mask()
            pixel_coords = np.array(self.wcs.world_to_pixel(self.sky_coords[mask])).T
        else:
            pixel_coords = np.array(self.wcs.world_to_pixel(self.sky_coords)).T

        return pixel_coords

    def filter_coordinates_on_ccd(self, pixel_coords: np.ndarray) -> np.ndarray:
        on_ccd_mask = np.bitwise_and(
            np.bitwise_and(0 <= pixel_coords[:, 0], pixel_coords[:, 0] <= self.x_max),
            np.bitwise_and(0 <= pixel_coords[:, 1], pixel_coords[:, 1] <= self.y_max),
        )

        return on_ccd_mask
    
    def num_stars_in_mask(self, centre):
        self.wcs.wcs.crval = [centre.ra.deg, centre.dec.deg]
        mask = self.get_mask()
        return mask

    def get_mask(self, pole_tolerance=1):
        ra_bounds, dec_bounds = self.get_ra_dec_bounds()

        # Avoid masking near the poles
        if self.is_near_poles(dec_bounds, pole_tolerance):
            return np.ones_like(self.df.dec, dtype=bool)

        dec_mask = self.df.dec.between(dec_bounds[0], dec_bounds[1])
        ra_mask = self.create_ra_mask(ra_bounds, dec_bounds)

        return np.bitwise_and(ra_mask, dec_mask)

    def get_ra_dec_bounds(self):
        bounds = self.wcs.pixel_to_world(
            [0, self.wcs.pixel_shape[0], self.wcs.pixel_shape[0], 0],
            [0, 0, self.wcs.pixel_shape[1], self.wcs.pixel_shape[1]],
        )
        ra_bounds = np.array([np.min(bounds.ra.deg), np.max(bounds.ra.deg)])
        dec_bounds = np.array([np.min(bounds.dec.deg), np.max(bounds.dec.deg)])
        return ra_bounds, dec_bounds

    def create_ra_mask(self, ra_bounds, dec_bounds):
        """Create a mask based on Right Ascension bounds."""
        # Calculate ra_bounds at equator if the equator is crossed
        if self.crosses_equator(dec_bounds):
            ra_bounds = self.ra_bound_at_equator()

        # Calculate the RA bounds at the equator if the field of view crosses the equator
        if np.diff(np.sign(ra_bounds)) != 0:
            return np.bitwise_or(
                self.df.ra.between(ra_bounds[-1], 360),
                self.df.ra.between(0, ra_bounds[0]),
            )
        else:
            return self.df.ra.between(ra_bounds[0], ra_bounds[1])

    def ra_bound_at_equator(self):
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

    @staticmethod
    def is_near_poles(dec_bounds, pole_tolerance):
        return np.any(np.abs(dec_bounds) > 90 - pole_tolerance)

    @staticmethod
    def crosses_equator(dec_bounds):
        return np.diff(np.sign(dec_bounds)) != 0

    @staticmethod
    def create_basic_wcs(
        center_ra: float = 0.0,
        center_dec: float = 0.0,
        pixel_scale_arcsec: float = 0.35,
        pixel_shape: Tuple[int, int] = (2000, 2000),
    ):
        """
        Create a basic WCS object.

        Parameters
        ----------
        center_ra : float
            The right ascension of the center of the field of view in degrees.
        center_dec : float
            The declination of the center of the field of view in degrees.
        pixel_scale_arcsec : float
            The pixel scale in arcseconds.
        pixel_shape : tuple of int
            The shape of the ccd in pixels, i.e. (num_pixels_x_axis, num_pixels_y_axis).
            This is the equivalent to flip(image_shape). See WCS documentation for more information.
        """
        wcs = WCS(naxis=2)

        # Set the coordinate system and projection type
        # This specifies the coordinate type for the first axis (X-axis) as Right Ascension (RA)
        # and the coordinate type for the second axis (Y-axis) as Declination (DEC).
        # Each time using using a tangent-plane projection (TAN).
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Set the reference pixel (center of the image)
        wcs.wcs.crpix = [pixel_shape[0] / 2, pixel_shape[1] / 2]

        # Set the reference coordinates (center of the field of view)
        wcs.wcs.crval = [center_ra, center_dec]

        # Set the pixel scale
        wcs.wcs.cdelt = np.array([-pixel_scale_arcsec / 3600, pixel_scale_arcsec / 3600])

        # Set the pixel shape
        wcs.pixel_shape = (pixel_shape[0], pixel_shape[1])

        return wcs

    @classmethod
    def from_telescope_specs(cls, df, TelescopeSpecs, center_ra=None, center_dec=None, **kwargs):
        wcs = cls.create_basic_wcs(
            center_ra=center_ra or 0,
            center_dec=center_dec or 0,
            pixel_scale_arcsec=TelescopeSpecs.pixel_scale,
            pixel_shape=TelescopeSpecs.pixel_shape,
        )
        return cls(df=df, wcs=wcs, **kwargs)

    def __repr__(self) -> str:
        return f"TangentialPlaneProjector(df={self.df!r}, wcs={self.wcs!r})"
    
    def flatten_on_sky(self, central_star: SkyCoord):
        coords_on_sky = (self.project(central_star) - self.wcs.wcs.crpix)*self.wcs.wcs.cdelt        
        return coords_on_sky