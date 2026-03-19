import sqlite3

from astroquery.gaia import Gaia

from astrafocus.sql.local_gaia_database_query import LocalGaiaDatabaseQuery


class RemoteGaiaDatabaseQuery(LocalGaiaDatabaseQuery):
    def populate_database(self, dec_min, dec_max, limit=100_000_000_000):
        """
        Query Gaia-2MASS data for a declination range, using local DB cache.
        """
        # Fetch from Gaia
        query = f"""
            SELECT TOP {limit} gaia.ra, gaia.dec, gaia.pmra, gaia.pmdec,
                phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                tmass.j_m, tmass.h_m, tmass.ks_m
            FROM gaiadr2.gaia_source AS gaia
            INNER JOIN gaiadr2.tmass_best_neighbour AS tmass_match ON tmass_match.source_id = gaia.source_id
            INNER JOIN gaiadr1.tmass_original_valid AS tmass ON tmass.tmass_oid = tmass_match.tmass_oid
            WHERE gaia.dec BETWEEN {dec_min} AND {dec_max}
        """
        job = Gaia.launch_job(query)
        r = job.get_results()
        df = r.to_pandas()
        df["dec_min"] = dec_min
        df["dec_max"] = dec_max

        # Store in DB
        conn = sqlite3.connect(self.db_path)
        df.to_sql("gaia_tmass", conn, if_exists="append", index=False)
        conn.close()
