import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from astrafocus.utils.logger import get_logger

logger = get_logger()


class LocalGaiaDatabaseQuery:
    """
    Perform queries on the Gaia-2MASS Local Catalogue.

    Parameters
    ----------
    db_path : str
        The path to the SQLite database file.

    Examples
    --------
    >>> from astrafocus.sql.local_gaia_database_query import LocalGaiaDatabaseQuery
    >>> lgdbq = LocalGaiaDatabaseQuery("path/to/db")
    >>> lgdbq.count_query(10, 20, 30, 40, max_phot_g_mean_mag=12)
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.conn = None
        self.query_input_validator = QueryInputValidator()

        if not db_path.is_file():
            raise FileNotFoundError(f"The database file at {db_path} does not exist.")

    def __call__(
        self,
        min_dec: float,
        max_dec: float,
        min_ra: float,
        max_ra: float,
        min_phot_g_mean_mag: float | None = None,
        max_phot_g_mean_mag: float | None = None,
        min_j_m: float | None = None,
        max_j_m: float | None = None,
    ):
        start_time = time.time()

        try:
            df_result = self.query(
                min_dec,
                max_dec,
                min_ra,
                max_ra,
                min_phot_g_mean_mag=min_phot_g_mean_mag,
                max_phot_g_mean_mag=max_phot_g_mean_mag,
                min_j_m=min_j_m,
                max_j_m=max_j_m,
            )
        finally:
            # Assure that connections is closed even if there is an error
            self._close_database_connection()
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Execution time of query: {execution_time:8.3f} seconds")

        return df_result

    def query(
        self,
        min_dec: float,
        max_dec: float,
        min_ra: float,
        max_ra: float,
        *,
        min_phot_g_mean_mag: float | None = None,
        max_phot_g_mean_mag: float | None = None,
        min_j_m: float | None = None,
        max_j_m: float | None = None,
    ):
        """
        Queries the local Gaia database for astronomical data
        within a specified range of declination and right ascension.
        If min_ra < max_ra, the right ascension range is assumed to cross the 0/360 degree border.

        Parameters
        ----------
        min_dec : float
            The minimum declination value to query.
        max_dec : float
            The maximum declination value to query.
        min_ra : float
            The minimum right ascension value to query.
        max_ra : float
            The maximum right ascension value to query.
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
        pd.DataFrame
            A pandas DataFrame containing the queried astronomical data.

        Raises
        ------
        TypeError
            If any of the input values is not of type float or int.
        ValueError
            If any of the input values is not within the specified range,
            or if the order of range borders is incorrect.]

        Examples
        --------

        >>> from astrafocus.sql.local_gaia_database_query import LocalGaiaDatabaseQuery
        >>> lgdbq = LocalGaiaDatabaseQuery("path/to/db")
        >>> lgdbq.count_query(10, 20, 30, 40, max_phot_g_mean_mag=12)
        """
        self.query_input_validator(
            min_dec=min_dec,
            max_dec=max_dec,
            min_ra=min_ra,
            max_ra=max_ra,
        )
        self.query_input_validator.validate_magnitudes(
            min_phot_g_mean_mag=min_phot_g_mean_mag,
            max_phot_g_mean_mag=max_phot_g_mean_mag,
            min_j_m=min_j_m,
            max_j_m=max_j_m,
        )
        self._connect_to_database()

        relevant_shard_ids = self._determine_relevant_shards(min_dec, max_dec)

        df_total = pd.concat(
            [
                self._sql_query_of_shard(
                    shard_id,
                    min_dec,
                    max_dec,
                    min_ra,
                    max_ra,
                    min_phot_g_mean_mag=min_phot_g_mean_mag,
                    max_phot_g_mean_mag=max_phot_g_mean_mag,
                    min_j_m=min_j_m,
                    max_j_m=max_j_m,
                )
                for shard_id in relevant_shard_ids
            ],
            axis=0,
        )

        self._close_database_connection()

        return df_total.sort_values(by=["j_m"]).reset_index(drop=True)

    def count_query(
        self,
        min_dec: float,
        max_dec: float,
        min_ra: float,
        max_ra: float,
        *,
        min_phot_g_mean_mag: float | None = None,
        max_phot_g_mean_mag: float | None = None,
        min_j_m: float | None = None,
        max_j_m: float | None = None,
    ):
        """
        Counts the number of entries in the local Gaia database
        within a specified range of declination and right ascension.

        Parameters
        ----------
        min_dec : float
            The minimum declination value to query.
        max_dec : float
            The maximum declination value to query.
        min_ra : float
            The minimum right ascension value to query.
        max_ra : float
            The maximum right ascension value to query.
        min_phot_g_mean_mag : float, optional
            The minimum GAIA mean magnitude to filter results (default is None).
        max_phot_g_mean_mag : float, optional
            The maximum GAIA mean magnitude to filter results (default is None).
        min_j_m : float, optional
            The minimum J-band magnitude to filter results (default is None).
        max_j_m : float, optional
            The maximum J-band magnitude to filter results (default is None).

        Returns
        -------
        int
            The count of entries matching the query criteria.

        Raises
        ------
        TypeError
            If any of the input values is not of type float or int.
        ValueError
            If any of the input values is not within the specified range,
            or if the order of range borders is incorrect.
        """
        self.query_input_validator(
            min_dec=min_dec,
            max_dec=max_dec,
            min_ra=min_ra,
            max_ra=max_ra,
        )
        self.query_input_validator.validate_magnitudes(
            min_phot_g_mean_mag=min_phot_g_mean_mag,
            max_phot_g_mean_mag=max_phot_g_mean_mag,
            min_j_m=min_j_m,
            max_j_m=max_j_m,
        )
        self._connect_to_database()

        relevant_shard_ids = self._determine_relevant_shards(min_dec, max_dec)

        total_count = sum(
            self._count_in_shard(
                shard_id,
                min_dec,
                max_dec,
                min_ra,
                max_ra,
                min_phot_g_mean_mag=min_phot_g_mean_mag,
                max_phot_g_mean_mag=max_phot_g_mean_mag,
                min_j_m=min_j_m,
                max_j_m=max_j_m,
            )
            for shard_id in relevant_shard_ids
        )

        return total_count

    def _count_in_shard(
        self,
        shard_id: str,
        min_dec: float,
        max_dec: float,
        min_ra: float,
        max_ra: float,
        min_phot_g_mean_mag: float | None = None,
        max_phot_g_mean_mag: float | None = None,
        min_j_m: float | None = None,
        max_j_m: float | None = None,
    ) -> float:
        """Execute an SQL COUNT query on a specific shard."""
        query = self._generate_query(
            shard_id,
            min_dec,
            max_dec,
            min_ra,
            max_ra,
            min_phot_g_mean_mag=min_phot_g_mean_mag,
            max_phot_g_mean_mag=max_phot_g_mean_mag,
            min_j_m=min_j_m,
            max_j_m=max_j_m,
            command="COUNT(*)",
        )
        count_result = pd.read_sql_query(query, self.conn)
        return int(count_result.iloc[0, 0])  # type: ignore

    def _determine_relevant_shards(self, min_dec: float, max_dec: float) -> set:
        """
        Determine relevant shards based on the specified range of declination.

        Returns
        -------
        set
            A set of relevant shard IDs.
        """
        arr = np.arange(start=np.floor(min_dec), stop=np.ceil(max_dec) + 1, step=1, dtype=int)
        return {f"{arr[i]}_{arr[i + 1]}" for i in range(len(arr) - 1)}

    def _sql_query_of_shard(
        self,
        shard_id: str,
        min_dec: float,
        max_dec: float,
        min_ra: float,
        max_ra: float,
        min_phot_g_mean_mag: float | None = None,
        max_phot_g_mean_mag: float | None = None,
        min_j_m: float | None = None,
        max_j_m: float | None = None,
    ):
        """
        Execute an SQL query on a specific shard within specific declination
        and right ascension ranges.
        """
        query = self._generate_query(
            shard_id,
            min_dec,
            max_dec,
            min_ra,
            max_ra,
            min_phot_g_mean_mag=min_phot_g_mean_mag,
            max_phot_g_mean_mag=max_phot_g_mean_mag,
            min_j_m=min_j_m,
            max_j_m=max_j_m,
        )

        return pd.read_sql_query(query, self.conn)

    @staticmethod
    def _generate_query(
        shard_id: str,
        min_dec: float,
        max_dec: float,
        min_ra: float,
        max_ra: float,
        min_phot_g_mean_mag: float | None = None,
        max_phot_g_mean_mag: float | None = None,
        min_j_m: float | None = None,
        max_j_m: float | None = None,
        command: str = "*",
    ):
        query = f"SELECT {command} FROM `{shard_id}` WHERE "
        if min_ra < max_ra:
            query += f"dec BETWEEN {min_dec} AND {max_dec} AND ra BETWEEN {min_ra} AND {max_ra}"
        else:
            query = (
                f"dec BETWEEN {min_dec} AND {max_dec} "
                f"AND (ra BETWEEN {min_ra} AND 360 OR ra BETWEEN 0 AND {max_ra})"
            )

        if min_phot_g_mean_mag is not None:
            query += f" AND phot_g_mean_mag >= {min_phot_g_mean_mag}"
        if max_phot_g_mean_mag is not None:
            query += f" AND phot_g_mean_mag <= {max_phot_g_mean_mag}"
        if min_j_m is not None:
            query += f" AND j_m >= {min_j_m}"
        if max_j_m is not None:
            query += f" AND j_m <= {max_j_m}"

        return query

    def _connect_to_database(self):
        """Connect to the SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path))

    def _close_database_connection(self):
        """Close the SQLite database connection."""
        if self.conn:
            self.conn.close()


class QueryInputValidator:
    """
    Validate input parameters for the query of the Gaia-2MASS Local Catalogue.

    Parameters
    ----------
    min_dec : float
        The minimum declination value.
    max_dec : float
        The maximum declination value.
    min_ra : float
        The minimum right ascension value.
    max_ra : float
        The maximum right ascension value.

    Raises
    ------
    TypeError
        If any of the input values is not of type float or int.
    ValueError
        If any of the input values is not within the specified range.
    """

    def __call__(self, min_dec, max_dec, min_ra, max_ra):
        """
        Validate input values for declination and right ascension.

        Parameters
        ----------
        min_dec : float
            The minimum declination value.
        max_dec : float
            The maximum declination value.
        min_ra : float
            The minimum right ascension value.
        max_ra : float
            The maximum right ascension value.

        Raises
        ------
        TypeError
            If any of the input values is not of type float or int.
        ValueError
            If any of the input values is not within the specified range.
        """
        self.validate_input(min_dec, max_dec, min_ra, max_ra)

    @staticmethod
    def validate_input(min_dec, max_dec, min_ra, max_ra):
        """
        Validate input values for declination and right ascension.

        Parameters
        ----------
        min_dec : float
            The minimum declination value.
        max_dec : float
            The maximum declination value.
        min_ra : float
            The minimum right ascension value.
        max_ra : float
            The maximum right ascension value.

        Raises
        ------
        TypeError
            If any of the input values is not of type float or int.
        ValueError
            If any of the input values is not within the specified range,
            or if the order of range borders is incorrect for the declination.
        """
        # Check that all provided arguments are numbers to
        # - assure default behaviour
        # - prevent SQL injection, i.e. the inclusion of potentially malicious strings into the query
        QueryInputValidator.check_numeric(min_dec, "min_dec")
        QueryInputValidator.check_numeric(max_dec, "max_dec")
        QueryInputValidator.check_numeric(min_ra, "min_ra")
        QueryInputValidator.check_numeric(max_ra, "max_ra")

        QueryInputValidator.check_range(min_dec, -90, 90, "declination")
        QueryInputValidator.check_range(max_dec, -90, 90, "declination")
        QueryInputValidator.check_range(min_ra, 0, 360, "right ascension")
        QueryInputValidator.check_range(max_ra, 0, 360, "right ascension")

        QueryInputValidator.check_order(min_dec, max_dec, "declination")

    @staticmethod
    def validate_magnitudes(min_phot_g_mean_mag, max_phot_g_mean_mag, min_j_m, max_j_m):
        QueryInputValidator.check_numeric(min_phot_g_mean_mag, "min_phot_g_mean_mag", accept_none=True)
        QueryInputValidator.check_numeric(max_phot_g_mean_mag, "max_phot_g_mean_mag", accept_none=True)
        QueryInputValidator.check_numeric(min_j_m, "min_j_m", accept_none=True)
        QueryInputValidator.check_numeric(max_j_m, "max_j_m", accept_none=True)

        if min_phot_g_mean_mag is not None and max_phot_g_mean_mag is not None:
            QueryInputValidator.check_order(min_phot_g_mean_mag, max_phot_g_mean_mag, "Gaia mean magnitude")

        if min_j_m is not None and max_j_m is not None:
            QueryInputValidator.check_order(min_j_m, max_j_m, "J-band magnitude")

    @staticmethod
    def check_numeric(value, value_name, accept_none=False):
        """Check if a value is numeric (float or int)."""
        if accept_none and value is None:
            return
        if not isinstance(value, float | int | np.integer | np.floating):
            raise TypeError(f"{value_name} must be of type float or int")

    @staticmethod
    def check_range(value, min_value, max_value, value_name):
        """Check if a value is within a specified range."""
        if not min_value <= value <= max_value:
            raise ValueError(
                f"The {value_name} must be within the range of [{min_value}, {max_value}], got {value}"
            )

    @staticmethod
    def check_order(min_value, max_value, value_name):
        """Check if the minimum value is less than the maximum value."""
        if not min_value < max_value:
            raise ValueError(
                f"{value_name} minimum value must be less than the maximum value, "
                f"got {min_value} and {max_value}"
            )
