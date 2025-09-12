from typing import Callable
import numpy as np
from astropy.coordinates import Angle, EarthLocation
from astropy.time import Time

from astrafocus.sql.local_gaia_database_query import LocalGaiaDatabaseQuery
from astrafocus.sql.shardwise_query import ShardwiseQuery
from astrafocus.targeting.zenith_angle_calculator import ZenithAngleCalculator
from astrafocus.targeting.zenith_neighbourhood import ZenithNeighbourhood
from astrafocus.targeting.airmass_models import find_airmass_threshold_crossover, plane_parallel_atmosphere
from astrafocus.targeting.zenith_neighbourhood_query_result import (
    ZenithNeighbourhoodQueryResult,
)
from astrafocus.utils.logger import get_logger

logger = get_logger()


class ZenithNeighbourhoodQuery:
    """
    Class for querying a database based on a zenith neighbourhood.

    Parameters
    ----------
    db_path : str
        Path to the database.
    zenith_neighbourhood : ZenithNeighbourhood
        Zenith neighbourhood object.
    maximal_number_of_stars : int, optional
        Maximum number of stars to be considered in the query (default is 1 000 000).
        This parameter is needed to prevent excessive queries that could lead to
        memory issues or long processing times.

    Examples
    --------
    zenith_neighbourhood_query = ZenithNeighbourhoodQuery(
        db_path="path_to/database.db",
        zenith_neighbourhood=zenith_neighbourhood
    )
    """

    def __init__(
        self,
        db_path: str,
        zenith_neighbourhood: ZenithNeighbourhood,
        maximal_number_of_stars: int | None = 1_000_000,
    ):
        """
        Initialize a ZenithNeighbourhoodQuery object.

        Parameters
        ----------
        db_path : str
            Path to the database.
        zenith_neighbourhood : ZenithNeighbourhood
            Zenith neighbourhood object.
        """
        self.zenith_neighbourhood = zenith_neighbourhood
        self.db_path = db_path
        self.maximal_number_of_stars = maximal_number_of_stars

    def query_full(
        self,
        n_sub_div=20,
        zenith_angle_strict=True,
        min_phot_g_mean_mag: float | None = None,
        max_phot_g_mean_mag: float | None = None,
        min_j_m: float | None = None,
        max_j_m: float | None = None,
    ) -> ZenithNeighbourhoodQueryResult:
        """Query the smallest rectangle that covers the whole patch.

        Parameters
        ----------
        n_sub_div : int, optional
            Number of subdivisions for approximation (default is 20).
        zenith_angle_strict : bool, optional
            If True, filter results based on zenith angle (default is True).
        min_phot_g_mean_mag : float, optional
            The minimum GAIA mean magnitude to query (default is None).
        max_phot_g_mean_mag : float, optional
            The maximum GAIA mean magnitude to query (default is None).
        min_j_m : float, optional
            The minimum J-band magnitude to query (default is None).
        max_j_m : float, optional
            The maximum J-band magnitude to query (default is None).

        Returns
        -------
        ZenithNeighbourhoodQueryResult
            Result of the query.
        """
        approx_dec, approx_ra = self.zenith_neighbourhood.get_constant_approximation_shards_deg(
            n_sub_div=n_sub_div
        )
        dec_min, dec_max = np.min(approx_dec), np.max(approx_ra)
        ra_min, ra_max = np.min(approx_ra), np.max(approx_ra)

        database_query = LocalGaiaDatabaseQuery(db_path=self.db_path)

        if self.maximal_number_of_stars is not None:
            number_of_stars = database_query.count_query(
                min_dec=dec_min,
                max_dec=dec_max,
                min_ra=ra_min,
                max_ra=ra_max,
                min_phot_g_mean_mag=min_phot_g_mean_mag,
                max_phot_g_mean_mag=max_phot_g_mean_mag,
                min_j_m=min_j_m,
                max_j_m=max_j_m,
            )
            if number_of_stars > self.maximal_number_of_stars:
                self._reduce_maximal_zenith_angle()
                return self.query_full(
                    n_sub_div=n_sub_div,
                    zenith_angle_strict=zenith_angle_strict,
                    min_phot_g_mean_mag=min_phot_g_mean_mag,
                    max_phot_g_mean_mag=max_phot_g_mean_mag,
                    min_j_m=min_j_m,
                    max_j_m=max_j_m,
                )

        result_df = database_query(
            min_dec=dec_min,
            max_dec=dec_max,
            min_ra=ra_min,
            max_ra=ra_max,
            min_phot_g_mean_mag=min_phot_g_mean_mag,
            max_phot_g_mean_mag=max_phot_g_mean_mag,
            min_j_m=min_j_m,
            max_j_m=max_j_m,
        )

        if zenith_angle_strict:
            result_df = self.filter_df_by_zenith_angle(result_df)

        return ZenithNeighbourhoodQueryResult(result_df)

    def query_shardwise(
        self,
        n_sub_div=20,
        zenith_angle_strict=True,
        min_phot_g_mean_mag: float | None = None,
        max_phot_g_mean_mag: float | None = None,
        min_j_m: float | None = None,
        max_j_m: float | None = None,
    ) -> ZenithNeighbourhoodQueryResult:
        """
        Query the database shard-wise, only searching each shard as far as needed.

        Parameters
        ----------
        n_sub_div : int, optional
            Number of subdivisions for approximation (default is 20).
        zenith_angle_strict : bool, optional
            If True, filter results based on zenith angle (default is True).
        min_phot_g_mean_mag : float, optional
            The minimum GAIA mean magnitude to query (default is None).
        max_phot_g_mean_mag : float, optional
            The maximum GAIA mean magnitude to query (default is None).
        min_j_m : float, optional
            The minimum J-band magnitude to query (default is None).
        max_j_m : float, optional
            The maximum J-band magnitude to query (default is None).

        Returns
        -------
        ZenithNeighbourhoodQueryResult
            Result of the query.
        """
        approx_dec, approx_ra = self.zenith_neighbourhood.get_constant_approximation_shards_deg(
            n_sub_div=n_sub_div
        )

        database_query = ShardwiseQuery(db_path=self.db_path)

        if self.maximal_number_of_stars is not None:
            number_of_stars = database_query.count_query_with_shard_array(
                approx_dec,
                approx_ra,
                min_phot_g_mean_mag=min_phot_g_mean_mag,
                max_phot_g_mean_mag=max_phot_g_mean_mag,
                min_j_m=min_j_m,
                max_j_m=max_j_m,
            )
            if number_of_stars > self.maximal_number_of_stars:
                self._reduce_maximal_zenith_angle()
                return self.query_shardwise(
                    n_sub_div=n_sub_div,
                    zenith_angle_strict=zenith_angle_strict,
                    min_phot_g_mean_mag=min_phot_g_mean_mag,
                    max_phot_g_mean_mag=max_phot_g_mean_mag,
                    min_j_m=min_j_m,
                    max_j_m=max_j_m,
                )

        result_df = database_query.querry_with_shard_array(
            approx_dec,
            approx_ra,
            min_phot_g_mean_mag=min_phot_g_mean_mag,
            max_phot_g_mean_mag=max_phot_g_mean_mag,
            min_j_m=min_j_m,
            max_j_m=max_j_m,
        )

        if zenith_angle_strict:
            result_df = self.filter_df_by_zenith_angle(result_df)
        else:
            result_df = ZenithNeighbourhoodQueryResult(result_df)

        return result_df

    def filter_df_by_zenith_angle(self, df) -> ZenithNeighbourhoodQueryResult:
        """
        Filter DataFrame based on zenith angle.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be filtered.

        Returns
        -------
        ZenithNeighbourhoodQueryResult
            Result of the filtered DataFrame.
        """
        if not hasattr(df, "zenith_angle"):
            ZenithAngleCalculator.add_zenith_angle_fast(df=df, zenith=self.zenith_neighbourhood.zenith)

        result_df = df[df.zenith_angle < self.zenith_neighbourhood.maximal_zenith_angle].reset_index(
            drop=True
        )

        return ZenithNeighbourhoodQueryResult(result_df)

    def _reduce_maximal_zenith_angle(
        self, reduction_factor: float = 2.0, number_of_stars: int | None = None
    ):
        """Reduce the maximal zenith angle by a specified reduction factor."""
        logger.warning(
            f"The number of stars in the zenith neighborhood ({number_of_stars}) "
            f"exceeds the maximum limit of {self.maximal_number_of_stars}. "
            "The maximal zenith angle will be reduced from "
            f"{self.zenith_neighbourhood.maximal_zenith_angle} "
            f"to {self.zenith_neighbourhood.maximal_zenith_angle / reduction_factor}."
        )
        self.zenith_neighbourhood = ZenithNeighbourhood(
            observatory_location=self.zenith_neighbourhood.observatory_location,
            observation_time=self.zenith_neighbourhood.observation_time,
            maximal_zenith_angle=self.zenith_neighbourhood.maximal_zenith_angle / reduction_factor,
        )

    def __repr__(self) -> str:
        return (
            f"ZenithNeighbourhoodQuery("
            f"db_path={self.db_path}, "
            f"zenith_neighbourhood={self.zenith_neighbourhood}, "
            f"maximal_number_of_stars={self.maximal_number_of_stars}"
            ")"
        )

    @classmethod
    def from_telescope_specs(
        cls,
        telescope_specs,
        observation_time=None,
        maximal_zenith_angle=None,
        db_path=None,
    ) -> "ZenithNeighbourhoodQuery":
        """
        Create an instance of the ZenithNeighbourhoodQuery class from an instance of the
        TelescopeSpecs class.

        Parameters
        ----------
        telescope_specs : TelescopeSpecs
            An instance of the TelescopeSpecs class.
        db_path : str, optional
            The path to the database, by default None

        Example
        -------
        >>> telescope_specs = TelescopeSpecs.load_telescope_config(file_path=path_to_config_file)
        >>> zenith_neighbourhood_query = ZenithNeighbourhoodQuery.from_telescope_specs(telescope_specs)
        """
        return cls(
            db_path=db_path or telescope_specs.gaia_tmass_db_path,
            zenith_neighbourhood=ZenithNeighbourhood.from_telescope_specs(
                telescope_specs=telescope_specs,
                observation_time=observation_time,
                maximal_zenith_angle=maximal_zenith_angle,
            ),
        )

    @classmethod
    def create_from_location_and_angle(
        cls,
        db_path: str,
        observatory_location: EarthLocation,
        maximal_zenith_angle: float | int | Angle,
        maximal_number_of_stars: int | None = 1_000_000,
        observation_time: Time | None = None,
    ) -> "ZenithNeighbourhoodQuery":
        """
        Create an instance of the ZenithNeighbourhoodQuery class with specified parameters.

        This class method is an alternative constructor that creates a ZenithNeighbourhoodQuery
        instance based on the provided observatory location, maximal zenith angle,
        and optional observation time.

        Parameters
        ----------
        db_path : str
            The path to the database.
        observatory_location : EarthLocation
            Location of the observatory.
        maximal_zenith_angle : float, int, or Angle
            Maximum zenith angle for the neighbourhood in degrees.
        maximal_number_of_stars : int, optional
            Maximum number of stars to be considered in the query (default is 1 000 000).
        observation_time : Optional[Time], optional
            Observation time specified using astropy's Time. (default is None, resulting to now)

        Example
        -------
        >>> db_path = '/path/to/database'
        >>> observatory_location = EarthLocation(lat=30.0, lon=-70.0, height=1000.0)
        >>> maximal_zenith_angle = 15.0
        >>> observation_time = Time('2023-12-01T12:00:00')
        >>> zenith_neighbourhood_query = ZenithNeighbourhoodQuery.create_from_location_and_angle(
        ...     db_path, observatory_location, maximal_zenith_angle, observation_time
        ... )
        """
        return cls(
            db_path=db_path,
            zenith_neighbourhood=ZenithNeighbourhood(
                observatory_location=observatory_location,
                observation_time=observation_time,
                maximal_zenith_angle=maximal_zenith_angle,
            ),
            maximal_number_of_stars=maximal_number_of_stars,
        )

    @staticmethod
    def find_airmass_threshold_crossover(
        airmass_threshold: float | None = 1.2,
        airmass_model: Callable = plane_parallel_atmosphere,
    ):
        return find_airmass_threshold_crossover(
            airmass_threshold=airmass_threshold, airmass_model=airmass_model
        )