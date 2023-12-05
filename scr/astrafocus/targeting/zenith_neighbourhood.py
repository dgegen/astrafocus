from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, Angle, EarthLocation, Longitude, SkyCoord
from astropy.time import Time
from astrafocus.interface.telescope_specs import TelescopeSpecs
from astrafocus.utils.logger import configure_logger

__all__ = ["ZenithNeighbourhood"]

logger = configure_logger()

RAD_TO_DEG = 180 / np.pi
DEFAULT_MAXIMAL_ZENITH_ANGLE = 10 * u.deg


class ZenithNeighbourhood:
    """
    Class representing a zenith neighbourhood for astronomical observations.

    Parameters
    ----------
    observatory_location : Optional[EarthLocation], optional
        Location of the observatory specified using astropy's EarthLocation.
    observation_time : Optional[Time], optional
        Observation time specified using astropy's Time.
    maximal_zenith_angle : float, optional
        Maximum zenith angle for the neighbourhood in degrees (default is DEFAULT_MAXIMAL_ZENITH_ANGLE).
    
    Examples
    --------
    # Zenith neighbourhood now
    >>> speculoos_geo_coords = {
        "lat": -24.627222 * u.deg, "lon": -70.404167 * u.deg, "height": 2635 * u.m
    }
    >>> zn = ZenithNeighbourhood(
        observatory_location=EarthLocation(**speculoos_geo_coords),
        maximal_zenith_angle=10 * u.deg
    )

    # Zenith neighbourhood at a specific time, crossing the 0, 360 deg boundary
    >>> import astropy
    >>> zenith_neighbourhood = ZenithNeighbourhood(
        maximal_zenith_angle=MAX_ZENITH_ANGLE,
        observation_time=astropy.time.Time("2023-11-23 00:35:54.5018")
    )
    """
    def __init__(
        self,
        observatory_location: EarthLocation,
        observation_time: Optional[Time] = None,
        maximal_zenith_angle=DEFAULT_MAXIMAL_ZENITH_ANGLE,
    ):
        """
        Initialize a ZenithNeighbourhood object.

        Parameters
        ----------
        observatory_location : Optional[EarthLocation], optional
            Location of the observatory specified using astropy's EarthLocation.
        observation_time : Optional[Time], optional
            Observation time specified using astropy's Time.
        maximal_zenith_angle : float, optional
            Maximum zenith angle for the neighbourhood in degrees (default is DEFAULT_MAXIMAL_ZENITH_ANGLE).
        """
        self.maximal_zenith_angle = ZenithNeighbourhood.check_maximal_zenith_angle(
            maximal_zenith_angle
        )
        altitude_angle = Angle(90 * u.deg) - self.maximal_zenith_angle

        self.location = observatory_location
        self.observation_time = observation_time or Time.now()
        self.validate_observatoy_location()
        self.validate_observation_time()

        self.zenith = HorizontalCoordinates(
            obstime=self.observation_time, location=self.location, alt=90 * u.deg, az=0 * u.deg
        )

        self.east = HorizontalCoordinates(
            obstime=self.observation_time,
            location=self.location,
            alt=altitude_angle,
            az=90 * u.deg,
        )
        self.north = HorizontalCoordinates(
            obstime=self.observation_time,
            location=self.location,
            alt=altitude_angle,
            az=180 * u.deg,
        )
        self.west = HorizontalCoordinates(
            obstime=self.observation_time,
            location=self.location,
            alt=altitude_angle,
            az=270 * u.deg,
        )
        self.south = HorizontalCoordinates(
            obstime=self.observation_time,
            location=self.location,
            alt=altitude_angle,
            az=0 * u.deg,
        )

        self.delta_lon_zenith = np.abs(self.east.ra.rad - self.west.ra.rad)
        self.delta_dec_zenith = np.abs(self.north.dec.rad - self.south.dec.rad)

        logger.info(
            f"Initializing ZenithNeighbourhood with "
            f"maximal_zenith_angle={self.maximal_zenith_angle}, "
            f"location={self.location}, "
            f"observation_time={self.observation_time}."
        )

        # Warn that the approximation is not valid for declinations near the poles
        if np.abs(self.zenith.dec.deg) > 90 - self.maximal_zenith_angle.deg:
            logger.warning(
                f"The neighbourhood approximation is not yet tested for declinations near the poles. "
                f"Zenith declination: {self.zenith.dec.deg}, "
                f"maximal zenith angle: {self.maximal_zenith_angle.deg}"
            )

    def get_ra_bounds(self, n, south=None, north=None):
        """
        Calculate exact RA bounds between specified declinations.

        Parameters
        ----------
        n : int
            Number of subdivisions.
        south : Optional[float], optional
            Southern declination in radians (default is None).
        north : Optional[float], optional
            Northern declination in radians (default is None).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Declinations and corresponding RA bounds.
        """        
        if south is None:
            south = self.south.dec.rad
        if north is None:
            north = self.north.dec.rad
        logger.info(
            "Calculate exact RA bounds between declinations "
            f"{south*RAD_TO_DEG:5.2f} and {north*RAD_TO_DEG:5.2f}."
        )

        # declinations = np.linspace(*np.sort([south, north]), n)
        declinations = np.sort(np.linspace(south, north, n))

        delta_ras = np.array([self.delta_phi_of_theta(theta) for theta in declinations])
        ra_bound_coords = np.array(
            [self.zenith.ra.rad - delta_ras / 2, self.zenith.ra.rad + delta_ras / 2]
        ).T

        # Wrap ra_bound_coords to [0, 2*pi]
        ra_bound_coords = Longitude(ra_bound_coords * u.rad).rad

        # Where delta_ras is NaN, the zenith is near the pole, where there are no ra bounds.
        ra_bound_coords[np.isnan(delta_ras), :] = np.array([0, 2 * np.pi])

        return declinations, ra_bound_coords

    def get_constant_approximation_shards(self, n_sub_div=20):
        """Calculate constant shard-wise RA bounds."""        
        dec_bounds = np.sort([self.south.dec.deg, self.north.dec.deg])
        dec_bounds = np.array([np.floor(dec_bounds[0]), np.ceil(dec_bounds[-1])])
        n_shreds = int(dec_bounds[-1] - dec_bounds[0])
        south, north = dec_bounds * np.pi / 180

        logger.info(
            "Calculate constant shard wise RA bounds between declinations "
            f"{int(dec_bounds[0])} and {int(dec_bounds[1])}."
        )

        constant_declinations, ra_bounds = self.get_constant_approximation(
            n_approx=n_shreds, n_sub_div=n_sub_div, south=south, north=north
        )

        return constant_declinations, ra_bounds

    def get_constant_approximation_shards_deg(self, n_sub_div):
        """
        Calculate constant approximation in degrees.

        Parameters
        ----------
        n_sub_div : int
            Number of subdivisions for approximation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Constant declinations and corresponding RA bounds in degrees.
        """        
        approx_dec, approx_ra = self.get_constant_approximation_shards(n_sub_div=n_sub_div)
        approx_dec_deg = np.array(np.round(approx_dec * RAD_TO_DEG), dtype=int)
        approx_ra_deg = approx_ra * RAD_TO_DEG
        return approx_dec_deg, approx_ra_deg

    def get_constant_approximation(self, n_approx=3, n_sub_div=20, south=None, north=None):
        """Calculate constant approximation."""
        declinations, ra_bound_coords = self.get_ra_bounds(
            n_approx * n_sub_div + 1, south=south, north=north
        )

        constant_approximation_values = np.array(
            [
                self.constant_approximation_ra_bounds(
                    ra_bound_coords[i * n_sub_div : n_sub_div * (i + 1), :]
                )
                for i in range(0, n_approx)
            ]
        )

        constant_approx_declinations = np.zeros((n_approx, 2))
        constant_approx_declinations[:, 0] = declinations[0:-1:n_sub_div]
        constant_approx_declinations[:-1, 1] = constant_approx_declinations[1:, 0]
        constant_approx_declinations[-1, 1] = declinations[-1]

        return constant_approx_declinations, constant_approximation_values

    def delta_phi_of_theta(self, dec):
        """ """
        # r*phi is length of something, so I think it should be self.delta_lon_zenith
        # return np.cos(self.zenith.dec.rad) / np.cos(dec) * self.delta_lon_zenith
        # return np.arcsin(np.cos(self.zenith.dec.rad) / np.cos(dec) * np.sin(self.delta_lon_zenith))
        # arccos(np.sin(theta_1)*np.sin(theta_2) + np.cos(theta_1)*np.cos(theta_2)*np.cos(np.abs(phi_1-phi_2))
        return np.arccos(
            (
                np.sin(self.zenith.dec.rad) ** 2
                - np.sin(dec) ** 2
                + np.cos(self.zenith.dec.rad) ** 2 * np.cos(np.abs(self.delta_lon_zenith))
            )
            / np.cos(dec) ** 2
        )

    @staticmethod
    def check_maximal_zenith_angle(maximal_zenith_angle):
        if isinstance(maximal_zenith_angle, (int, float)):
            maximal_zenith_angle = Angle(float(maximal_zenith_angle) * u.deg)
        elif isinstance(maximal_zenith_angle, u.Quantity):
            maximal_zenith_angle = Angle(maximal_zenith_angle)
        if not isinstance(maximal_zenith_angle, Angle):
            raise ValueError(
                "maximal_zenith_angle must be of type astropy.coordinates.angles.Angle"
            )
        if maximal_zenith_angle < 0 * u.deg or maximal_zenith_angle > 90 * u.deg:
            raise ValueError("maximal_zenith_angle must be in the range [0, 90] deg")
        return maximal_zenith_angle

    @staticmethod
    def constant_approximation_ra_bounds(phi_bound_coords):
        max_ind = np.argmax(
            # Angular distance between the bounds
            np.min(
                np.minimum(
                    np.diff(phi_bound_coords, axis=1),
                    Longitude(np.diff(phi_bound_coords, axis=1) * u.rad).rad,
                )
            )
        )
        offset_west = phi_bound_coords[max_ind, 0]
        offset_east = phi_bound_coords[max_ind, 1]
        return np.array([offset_west, offset_east])

    def __repr__(self) -> str:
        return (
            f"ZenithNeighbourhood(maximal_zenith_angle={self.maximal_zenith_angle}, "
            f"location={self.location}, observation_time={self.observation_time})"
        )

    @classmethod
    def from_telescope_specs(
        cls, telescope_specs: TelescopeSpecs, observation_time=None, maximal_zenith_angle=None
    ):
        logger.info(f"Initialising from config of telescope {telescope_specs.name}.")
        return cls(
            observatory_location=telescope_specs.observatory_location,
            observation_time=observation_time,
            maximal_zenith_angle=(
                maximal_zenith_angle
                if maximal_zenith_angle is not None
                else telescope_specs.find_airmass_threshold_crossover()
            ),
        )

    def validate_observatoy_location(self):
        if not isinstance(self.location, EarthLocation):
            raise ValueError(
                "observatory_location must be of type astropy.coordinates.EarthLocation"
            )

    def validate_observation_time(self):
        if not isinstance(self.observation_time, Time):
            raise ValueError("observation_time must be of type astropy.time.Time")


class HorizontalCoordinates:
    """
    Class representing horizontal coordinates.

    Examples
    --------
    south = HorizontalCoordinates(
        obstime=obs_time, location=location, alt=90 * u.deg - maximal_zenith_angle, az=0 * u.deg
    )
    """

    def __init__(self, *args, **kwargs):
        self.altaz = AltAz(*args, **kwargs)
        self.skycoord = SkyCoord(self.altaz)
        self.icrs = self.skycoord.transform_to("icrs")

    @property
    def alt(self):
        return self.altaz.alt

    @property
    def az(self):
        return self.altaz.az

    @property
    def ra(self):
        return self.icrs.ra

    @property
    def dec(self):
        return self.icrs.dec

    def __repr__(self):
        return f"HorizontalCoordinates(alt={self.alt}, az={self.az})"


class ApproximateZenith:
    """
    Class representing an approximate zenith position.

    Parameters
    ----------
    observatory_location : EarthLocation
        Location of the observatory specified using astropy's EarthLocation.
    observation_time : Optional[Time], optional
        Observation time specified using astropy's Time (default is None).

    Examples
    --------
    >>> speculoos_geo_coords = {
    ...     "lat": -24.627222 * u.deg, "lon": -70.404167 * u.deg, "height": 2635 * u.m
    ... }
    >>> observatory_location = EarthLocation(**speculoos_geo_coords)
    >>> approximate_zenith = ApproximateZenith(observatory_location)

    # Check the approximate zenith position
    >>> print(approximate_zenith.zenith)
    SkyCoord (ICRS): (ra, dec) in deg
    """    
    def __init__(
        self, observatory_location: EarthLocation, observation_time: Optional[Time] = None
    ):
        """
        Initialize an ApproximateZenith object.

        Parameters
        ----------
        observatory_location : EarthLocation
            Location of the observatory specified using astropy's EarthLocation.
        observation_time : Optional[Time], optional
            Observation time specified using astropy's Time (default is None).
        """        
        self.location = observatory_location
        self.observation_time = observation_time or Time.now()

        self.validate_observatoy_location()
        self.validate_observation_time()

        self.zenith = self.get_approximate_zenith_icrs()

    def get_approximate_zenith_icrs(self) -> SkyCoord:
        """
        DEC_Z ~ obs_latitude
        RA_Z ~ local_sidereal_time
        """

        local_sidereal_time = self.get_local_sidereal_time()
        ra = local_sidereal_time.deg
        dec = self.location.lat.deg

        return SkyCoord(ra, dec, frame="icrs", unit="deg")

    def get_local_sidereal_time(self):
        """
        LST.hms format this into hh:mm:ss
        making it comparable to online sources like https://sidereal.app/calculate
        """
        observing_time = Time(self.observation_time, scale="utc", location=self.location)
        local_sidereal_time = observing_time.sidereal_time("mean")

        return local_sidereal_time

    def validate_observatoy_location(self):
        if not isinstance(self.location, EarthLocation):
            raise ValueError(
                "observatory_location must be of type astropy.coordinates.EarthLocation"
            )

    def validate_observation_time(self):
        if not isinstance(self.observation_time, Time):
            raise ValueError("observation_time must be of type astropy.time.Time")
