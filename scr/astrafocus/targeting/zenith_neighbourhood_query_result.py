import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from astrafocus.targeting.zenith_angle_calculator import ZenithAngleCalculator
from astrafocus.targeting.tangential_plane_projector import TangentialPlaneProjector

DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 180 / np.pi

G_MAG_RANGE = (6, 12)  # 14
J_MAG_RANGE = (6, 12)  # 14

FOV_HEIGHT = 11.666666 / 60
FOV_WIDTH = 11.6666666 / 60


class ZenithNeighbourhoodQueryResult(pd.DataFrame):
    """
    Class representing the result of a zenith neighbourhood query.

    Examples
    --------
    >>> from astrafocus.targeting.zenith_neighbourhood import ZenithNeighbourhood
    >>> from astrafocus.targeting.zenith_neighbourhood_query import ZenithNeighbourhoodQuery

    # Zenith neighbourhood now
    >>> speculoos_geo_coords = {
        "lat": -24.627222 * u.deg, "lon": -70.404167 * u.deg, "height": 2635 * u.m
    }
    >>> zn = ZenithNeighbourhood(
        observatory_location=EarthLocation(**speculoos_geo_coords),
        maximal_zenith_angle=10 * u.deg
    )

    # Perform query of approximation
    >>> db_path = "path_to/database.db"
    >>> zenith_neighbourhood_query = ZenithNeighbourhoodQuery(
        db_path=db_path, zenith_neighbourhood=zenith_neighbourhood
    )
    >>> df = zenith_neighbourhood_query.query_shardwise(n_sub_div=20)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.ZenithQueryResult
        # self.maximal_zenith_angle

    def mask_by_magnitude(self, g_mag_range=G_MAG_RANGE, j_mag_range=J_MAG_RANGE):
        """Mask the DataFrame based on magnitude ranges.

        Parameters
        ----------
        g_mag_range : Tuple[float, float], optional
            Range for the G magnitude (default is G_MAG_RANGE).
        j_mag_range : Tuple[float, float], optional
            Range for the J magnitude (default is J_MAG_RANGE).

        Returns
        -------
        ZenithNeighbourhoodQueryResult
            Masked result based on magnitude ranges.
        """
        mask = np.bitwise_and(
            self["phot_g_mean_mag"].between(*g_mag_range), self["j_m"].between(*j_mag_range)
        )
        return ZenithNeighbourhoodQueryResult(self[mask].reset_index(drop=True))
    
    def get_sky_coord_of_select_star(self, ind):
        return SkyCoord(**self[["ra", "dec"]].iloc[ind].to_dict(), unit="deg", frame="icrs")

    def add_zenith_angle_and_cartesian_coordinates(self, zenith_neighbourhood):
        """
        Add zenith angle and Cartesian coordinates to the DataFrame.

        Parameters
        ----------
        zenith_neighbourhood : ZenithNeighbourhood
            Zenith neighbourhood object.
        """
        ZenithAngleCalculator.add_zenith_angle_and_cartesian_coordinates(
            self, zenith_neighbourhood.zenith
        )

    def add_cartesian_coordinates(self):
        """Add Cartesian coordinates to the DataFrame."""
        ZenithAngleCalculator.add_cartesian_coordinates(self)

    def add_zenith_angle_fast(self, zenith_neighbourhood):
        ZenithAngleCalculator.add_zenith_angle_fast(self, zenith_neighbourhood.zenith)

    def angular_distance(self, ra, deg):
        return (
            np.arccos(
                np.dot(
                    self.loc[:, ["x", "y", "z"]].to_numpy(),
                    np.array(self.spherical_to_cartesian(ra * DEG_TO_RAD, deg * DEG_TO_RAD)),
                ).reshape(-1, 1),
            )
        ) * RAD_TO_DEG

    def determine_stars_in_neighbourhood(self, height=FOV_HEIGHT, width=FOV_WIDTH):
        """Determine the number of stars in the neighbourhood of all stars in the DataFrame."""
        n = np.array(
            [
                self.determine_stars_in_neighbourhood_of_star(ind, height=height, width=width)
                for ind in range(self.shape[0])
            ]
        )
        self.loc[:, "n"] = n
        return n

    def determine_stars_in_neighbourhood_of_star(self, ind, height=FOV_HEIGHT, width=FOV_WIDTH):
        """Determine the number of stars in the neighbourhood of a single stars."""
        return np.sum(self.find_stars_in_neighbourhood(ind, height=height, width=width))

    def find_stars_in_neighbourhood(self, ind, height=FOV_HEIGHT, width=FOV_WIDTH):
        """Find all stars in the neighbourhood of a specific stars."""
        theta_u, theta_l, phi_l, phi_r = ZenithNeighbourhoodQueryResult.find_bounds_star(
            self.ra.iloc[ind], self.dec.iloc[ind], height=height, width=width
        )
        # bf.query('@theta_l <= dec <= @theta_u and @phi_l <= ra <= @phi_r')
        mask = np.bitwise_and(self.dec.between(theta_l, theta_u), self.ra.between(phi_l, phi_r))

        return mask

    @staticmethod
    def find_bounds_star(phi_c, theta_c, height=FOV_HEIGHT, width=FOV_WIDTH):
        # Calculate the upper and lower bounds for theta
        theta_u = theta_c + height / 2
        theta_l = theta_c - height / 2

        # Calculate the covering for phi
        delta_phi = np.maximum(
            ZenithNeighbourhoodQueryResult.inverse_ra_dec(width, theta_u),
            ZenithNeighbourhoodQueryResult.inverse_ra_dec(width, theta_l),
        )

        # Calculate the left and right bounds for phi
        phi_l = phi_c - delta_phi / 2
        phi_r = phi_c + delta_phi / 2

        return theta_u, theta_l, phi_l, phi_r

    @staticmethod
    def inverse_ra_dec(delta_tau, theta):
        result = (
            np.arccos(
                (np.cos(delta_tau * DEG_TO_RAD) - np.sin(theta * DEG_TO_RAD) ** 2)
                / np.cos(theta * DEG_TO_RAD) ** 2
            )
            * RAD_TO_DEG
        )
        return result

    @staticmethod
    def spherical_to_cartesian(ra, dec):
        """Convert spherical coordinates to Cartesian coordinates."""
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return x, y, z
