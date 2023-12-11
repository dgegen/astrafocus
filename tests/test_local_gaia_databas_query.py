import unittest
import logging
import sqlite3
import pandas as pd

from astrafocus.sql.local_gaia_database_query import QueryInputValidator
from astrafocus.sql.local_gaia_database_query import LocalGaiaDatabaseQuery

from utils import load_config


class TestQueryInputValidator(unittest.TestCase):
    def setUp(self):
        self.validator = QueryInputValidator()

    def test_valid_input(self):
        # Valid input within specified ranges
        self.assertIsNone(self.validator.validate_input(10, 20, 30, 40))

    def test_invalid_type(self):
        # Invalid input type (string instead of float)
        with self.assertRaises(TypeError):
            self.validator.validate_input("not_float", 20, 30, 40)

    def test_invalid_range_declination(self):
        # Invalid declination range (below minimum)
        with self.assertRaises(ValueError):
            self.validator.validate_input(-100, 20, 30, 40)

        # Invalid declination range (above maximum)
        with self.assertRaises(ValueError):
            self.validator.validate_input(10, 200, 30, 40)

    def test_invalid_range_ra(self):
        # Invalid right ascension range (below minimum)
        with self.assertRaises(ValueError):
            self.validator.validate_input(10, 20, -10, 40)

        # Invalid right ascension range (above maximum)
        with self.assertRaises(ValueError):
            self.validator.validate_input(10, 20, 30, 400)

    def test_invalid_order_declination(self):
        # Invalid declination order (min greater than max)
        with self.assertRaises(ValueError):
            self.validator.validate_input(30, 20, 30, 40)


class TestLocalGaiaDatabaseQuery(unittest.TestCase):
    def setUp(self):
        config = load_config()
        self.db_path = config["path_to_gaia_tmass_db"]

        self.logger = logging.getLogger(__name__)

    def tearDown(self):
        try:
            # Clean up the test database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DROP TABLE IF EXISTS `1_2`")
        except Exception as e:
            print(f"Error during tearDown: {e}")

    def test_determine_relevant_shards(self):
        # Test _determine_relevant_shards method
        query_obj = LocalGaiaDatabaseQuery(self.db_path, logger=self.logger)
        shards = query_obj._determine_relevant_shards(10, 20)
        self.assertEqual(
            shards,
            {
                "10_11",
                "11_12",
                "12_13",
                "13_14",
                "14_15",
                "15_16",
                "16_17",
                "17_18",
                "18_19",
                "19_20",
            },
        )

    def test_sql_query_of_shard(self):
        # Test _sql_query_of_shard method
        query_obj = LocalGaiaDatabaseQuery(self.db_path, logger=self.logger)
        query_obj._connect_to_database()
        df = query_obj._sql_query_of_shard("20_21", 20, 21, 23, 23.5)
        query_obj._close_database_connection()
        self.assertIsInstance(df, pd.DataFrame)

    def test_query(self):
        query_obj = LocalGaiaDatabaseQuery(self.db_path, logger=self.logger)
        df = query_obj(20, 22.2, 23, 23.5)
        self.assertIsInstance(df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
