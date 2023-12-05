from typing import Optional, Union
import numpy as np

from astropy.coordinates import EarthLocation, Angle
from astropy.time import Time

from astrafocus.sql.local_gaia_database_query import LocalGaiaDatabaseQuery
from astrafocus.sql.shardwise_query import ShardwiseQuery
from astrafocus.targeting.zenith_neighbourhood import ZenithNeighbourhood
from astrafocus.targeting.zenith_neighbourhood_query_result import ZenithNeighbourhoodQueryResult
from astrafocus.targeting.zenith_angle_calculator import ZenithAngleCalculator


class ZenithNeighbourhoodQuery:
    """
    Class for querying a database based on a zenith neighbourhood.

    Parameters
    ----------
    db_path : str
        Path to the database.
    zenith_neighbourhood : ZenithNeighbourhood
        Zenith neighbourhood object.

    Examples
    --------
    zenith_neighbourhood_query = ZenithNeighbourhoodQuery(
        db_path="path_to/database.db",
        zenith_neighbourhood=zenith_neighbourhood
    )
    """

    def __init__(self, db_path: str, zenith_neighbourhood: ZenithNeighbourhood):
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

    def query_full(self, n_sub_div=20, zenith_angle_strict=True) -> ZenithNeighbourhoodQueryResult:
        """Query the smallest rectangle that covers the whole patch.

        Parameters
        ----------
        n_sub_div : int, optional
            Number of subdivisions for approximation (default is 20).
        zenith_angle_strict : bool, optional
            If True, filter results based on zenith angle (default is True).

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

        print(dec_min, dec_max, ra_min, ra_max)
        database_query = LocalGaiaDatabaseQuery(db_path=self.db_path)
        result_df = database_query(min_dec=dec_min, max_dec=dec_max, min_ra=ra_min, max_ra=ra_max)

        if zenith_angle_strict:
            result_df = self.filter_df_by_zenith_angle(result_df)
        else:
            result_df = ZenithNeighbourhoodQueryResult(result_df)

        return ZenithNeighbourhoodQueryResult(result_df)

    def query_shardwise(
        self, n_sub_div=20, zenith_angle_strict=True
    ) -> ZenithNeighbourhoodQueryResult:
        """
        Query the database shard-wise, only searching each shard as far as needed.

        Parameters
        ----------
        n_sub_div : int, optional
            Number of subdivisions for approximation (default is 20).
        zenith_angle_strict : bool, optional
            If True, filter results based on zenith angle (default is True).

        Returns
        -------
        ZenithNeighbourhoodQueryResult
            Result of the query.
        """
        approx_dec, approx_ra = self.zenith_neighbourhood.get_constant_approximation_shards_deg(
            n_sub_div=n_sub_div
        )

        database_query = ShardwiseQuery(db_path=self.db_path)
        result_df = database_query.querry_with_shard_array(approx_dec, approx_ra)

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
            ZenithAngleCalculator.add_zenith_angle_fast(
                df=df, zenith=self.zenith_neighbourhood.zenith
            )

        result_df = df[
            df.zenith_angle < self.zenith_neighbourhood.maximal_zenith_angle
        ].reset_index(drop=True)

        return ZenithNeighbourhoodQueryResult(result_df)

    def __repr__(self) -> str:
        return (
            f"ZenithNeighbourhoodQuery("
            f"db_path={self.db_path}, "
            f"zenith_neighbourhood={self.zenith_neighbourhood}"
            ")"
        )

    @classmethod
    def from_telescope_specs(
        cls, telescope_specs, observation_time=None, maximal_zenith_angle=None, db_path=None
    ) -> "ZenithNeighbourhoodQuery":
        """
        Create an instance of the ZenithNeighbourhoodQuery class from an instance of the TelescopeSpecs class.

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
        maximal_zenith_angle: Union[float, int, Angle],
        observation_time: Optional[Time] = None,
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
        )
