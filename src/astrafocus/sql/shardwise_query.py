import time

import numpy as np
import pandas as pd

from astrafocus.sql.local_gaia_database_query import LocalGaiaDatabaseQuery
from astrafocus.utils.logger import get_logger

logger = get_logger()


class ShardwiseQuery(LocalGaiaDatabaseQuery):
    def querry_with_shard_array(
        self,
        dec_arr,
        ra_arr,
        min_phot_g_mean_mag: float | None = None,
        max_phot_g_mean_mag: float | None = None,
        min_j_m: float | None = None,
        max_j_m: float | None = None,
    ):
        """
        Perform shardwise queries on the Gaia-2MASS Local Catalogue.
        """
        time_start = time.time()

        for dec, ra in zip(dec_arr, ra_arr):
            self.query_input_validator(
                min_dec=dec[0],
                max_dec=dec[1],
                min_ra=ra[0],
                max_ra=ra[1],
            )

        self._validate_shard_declination_array(dec_arr)

        self._connect_to_database()
        try:
            result_df = pd.concat(
                [
                    self._sql_query_of_shard(
                        shard_id=f"{dec[0]}_{dec[1]}",
                        min_dec=dec[0],
                        max_dec=dec[1],
                        min_ra=ra[0],
                        max_ra=ra[1],
                        min_phot_g_mean_mag=min_phot_g_mean_mag,
                        max_phot_g_mean_mag=max_phot_g_mean_mag,
                        min_j_m=min_j_m,
                        max_j_m=max_j_m,
                    )
                    for dec, ra in zip(dec_arr, ra_arr)
                ],
                axis=0,
            )
        finally:
            self._close_database_connection()
            execution_time = time.time() - time_start
            logger.info(f"Execution time of query: {execution_time:8.3f} seconds")

        return result_df

    def count_query_with_shard_array(
        self,
        dec_arr,
        ra_arr,
        min_phot_g_mean_mag: float | None = None,
        max_phot_g_mean_mag: float | None = None,
        min_j_m: float | None = None,
        max_j_m: float | None = None,
    ):
        """
        Count the number of stars in the Gaia-2MASS Local Catalogue using shardwise queries.
        """
        time_start = time.time()

        for dec, ra in zip(dec_arr, ra_arr):
            self.query_input_validator(
                min_dec=dec[0],
                max_dec=dec[1],
                min_ra=ra[0],
                max_ra=ra[1],
            )

        self._validate_shard_declination_array(dec_arr)

        self._connect_to_database()
        try:
            count = sum(
                self._count_in_shard(
                    shard_id=f"{dec[0]}_{dec[1]}",
                    min_dec=dec[0],
                    max_dec=dec[1],
                    min_ra=ra[0],
                    max_ra=ra[1],
                    min_phot_g_mean_mag=min_phot_g_mean_mag,
                    max_phot_g_mean_mag=max_phot_g_mean_mag,
                    min_j_m=min_j_m,
                    max_j_m=max_j_m,
                )
                for dec, ra in zip(dec_arr, ra_arr)
            )
        finally:
            self._close_database_connection()
            execution_time = time.time() - time_start
            logger.debug(f"Execution time of count query: {execution_time:8.3f} seconds")

        return count

    def _validate_shard_declination_array(self, dec):
        """
        Validate input arrays for declination ranges.
        """
        if not np.array_equal(dec[1:, 0], dec[:-1, 1]):
            raise ValueError(
                "The second column of dec must be equal to the first column of the next row,"
                " i.e. dec[1:, 0] == dec[:-1, 1]"
            )
        if not (
            np.array_equal(dec[:, 0], np.arange(dec[0, 0], dec[-1, 0] + 1, 1))
            and dec[-1, 1] == dec[-1, 0] + 1
        ):
            raise ValueError(
                "Dec must be a sequential range starting from dec[0, 0] to dec[-1, 0] with a step of 1."
            )
