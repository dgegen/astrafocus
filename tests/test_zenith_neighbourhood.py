import unittest

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from astrafocus.targeting.zenith_neighbourhood import ApproximateZenith, ZenithNeighbourhood


class TestGetZenithICRS(unittest.TestCase):
    def test_zenith_vs_approximate_zenith(self):
        speculoos_geo_coords = {
            "lat": -24.627222 * u.deg,
            "lon": -70.404167 * u.deg,
            "height": 2635 * u.m,
        }
        observatory_location = EarthLocation(**speculoos_geo_coords)

        observation_time = Time.now()

        # Calculate zenith coordinates
        zenith_icrs = ZenithNeighbourhood(
            observatory_location=observatory_location, observation_time=observation_time
        ).zenith
        zenith_ra, zenith_dec = zenith_icrs.ra.deg, zenith_icrs.dec.deg

        # Approximate coordinates
        approximate_zenith_icrs = ApproximateZenith(
            observatory_location=observatory_location, observation_time=observation_time
        ).zenith

        approximate_zenith_ra = approximate_zenith_icrs.ra.deg
        approximate_zenith_dec = approximate_zenith_icrs.dec.deg

        # Check if the calculated zenith coordinates match the expected values
        self.assertAlmostEqual(zenith_ra, approximate_zenith_ra, delta=0.4)
        self.assertAlmostEqual(zenith_dec, approximate_zenith_dec, delta=0.2)


if __name__ == "__main__":
    unittest.main()
