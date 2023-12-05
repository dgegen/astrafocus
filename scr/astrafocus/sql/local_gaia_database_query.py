import sqlite3
import time

import numpy as np
import pandas as pd

from astrafocus.utils.logger import configure_logger


class LocalGaiaDatabaseQuery:
    """
    Perform queries on the Gaia-2MASS Local Catalogue.

    Parameters
    ----------
    db_path : str
        The path to the SQLite database file.
    logger : logging.Logger, optional
        A custom logger to use for logging, by default None.
        
    Examples
    --------
    local_gaia_database_query = LocalGaiaDatabaseQuery(db_path)
    """

    def __init__(self, db_path, logger=None):
        self.db_path = db_path
        self.conn = None
        self.query_input_validator = QueryInputValidator()

        # Configure logging
        if logger is None:
            self.logger = configure_logger()
        else:
            self.logger = logger

    def __call__(self, min_dec, max_dec, min_ra, max_ra):
        start_time = time.time()

        try:
            df_result = self.query(min_dec, max_dec, min_ra, max_ra)
        finally:
            # Assure that connections is closed even if there is an error
            self._close_database_connection()
            end_time = time.time()
            execution_time = end_time - start_time
            self.logger.info(f"Execution time of query: {execution_time:8.3f} seconds")

        return df_result

    def query(self, min_dec, max_dec, min_ra, max_ra):
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
            or if the order of range borders is incorrect.
        """        
        self.query_input_validator(
            min_dec=min_dec,
            max_dec=max_dec,
            min_ra=min_ra,
            max_ra=max_ra,
        )
        self._connect_to_database()

        relevant_shard_ids = self._determine_relevant_shards(min_dec, max_dec)

        df_total = pd.concat(
            [
                self._sql_query_of_shard(shard_id, min_dec, max_dec, min_ra, max_ra)
                for shard_id in relevant_shard_ids
            ],
            axis=0,
        )

        self._close_database_connection()

        return df_total.sort_values(by=["j_m"]).reset_index(drop=True)

    def _determine_relevant_shards(self, min_dec, max_dec):
        """
        Determine relevant shards based on the specified range of declination.

        Returns
        -------
        set
            A set of relevant shard IDs.
        """
        arr = np.arange(start=np.floor(min_dec), stop=np.ceil(max_dec) + 1, step=1, dtype=int)
        return {f"{arr[i]}_{arr[i + 1]}" for i in range(len(arr) - 1)}

    def _sql_query_of_shard(self, shard_id, min_dec, max_dec, min_ra, max_ra):
        """Execute an SQL query on a specific shard within specific declination and right ascension ranges."""
        if min_ra < max_ra:
            query = (
                f"SELECT * FROM `{shard_id}` "
                f"WHERE dec BETWEEN {min_dec} AND {max_dec} AND ra BETWEEN {min_ra} AND {max_ra}"
            )
        else:
            query = (
                f"SELECT * FROM `{shard_id}` "
                f"WHERE dec BETWEEN {min_dec} AND {max_dec} "
                f"AND (ra BETWEEN {min_ra} AND 360 OR ra BETWEEN 0 AND {max_ra})"
            )
        return pd.read_sql_query(query, self.conn)

    def _connect_to_database(self):
        """Connect to the SQLite database."""        
        self.conn = sqlite3.connect(self.db_path)

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
    def check_numeric(value, value_name):
        """Check if a value is numeric (float or int)."""
        if not isinstance(value, (float, int, np.integer, np.floating)):
            raise TypeError(f"{value_name} must be of type float or int")

    @staticmethod
    def check_range(value, min_value, max_value, value_name):
        """Check if a value is within a specified range."""
        if not min_value <= value <= max_value:
            raise ValueError(
                f"The {value_name} must be within the range of "
                f"[{min_value}, {max_value}], got {value}"
            )

    @staticmethod
    def check_order(min_value, max_value, value_name):
        """Check if the minimum value is less than the maximum value."""
        if not min_value < max_value:
            raise ValueError(
                f"{value_name} minimum value must be less than the maximum value, "
                f"got {min_value} and {max_value}"
            )
