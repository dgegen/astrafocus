import numpy as np

from autofocus.sql.local_gaia_database_query import LocalGaiaDatabaseQuery
from autofocus.sql.shardwise_query import ShardwiseQuery
from autofocus.targeting.zenith_neighbourhood import ZenithNeighbourhood
from autofocus.targeting.zenith_neighbourhood_query_result import ZenithNeighbourhoodQueryResult
from autofocus.targeting.zenith_angle_calculator import ZenithAngleCalculator


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

    def query_full(self, n_sub_div=20, zenith_angle_strict=True):
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

    def query_shardwise(self, n_sub_div=20, zenith_angle_strict=True):
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

    def filter_df_by_zenith_angle(self, df):
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
    ):
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
