import unittest

import pandas as pd

from astrafocus.autofocuser import AnalyticResponseAutofocuser
from astrafocus.focus_measure_operators import FFTFocusMeasureTan2022
from astrafocus.interface.simulation import ObservationBasedDeviceSimulator
from astrafocus.star_size_focus_measure_operators import HFRStarFocusMeasure

from utils import ConfigTests


CONFIG = ConfigTests().get()

class TestAutofocuser(unittest.TestCase):
    def setUp(self):
        config = CONFIG
        path_to_fits = config["path_to_fits"]
        if path_to_fits is None:
            self.skipTest("No path to fits files provided in config.")
        self.TELESCOPE_INTERFACE = ObservationBasedDeviceSimulator(
            fits_path=path_to_fits
        )

    def test_autofocuser(self):
        ARAF = AnalyticResponseAutofocuser(
            autofocus_device_manager=self.TELESCOPE_INTERFACE,
            exposure_time=3.0,
            focus_measure_operator=HFRStarFocusMeasure,
            n_steps=(20, 5),
            n_exposures=(1, 2),
            decrease_search_range=True,
            percent_to_cut=40,
            keep_images=True,
            secondary_focus_measure_operators={"FFT": FFTFocusMeasureTan2022()},
        )

        ARAF.run()
        df = ARAF.focus_record

        # Add assertions based on your expectations
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_autofocuser_single_sweep(self):
        ARAF = AnalyticResponseAutofocuser(
            autofocus_device_manager=self.TELESCOPE_INTERFACE,
            exposure_time=3.0,
            focus_measure_operator=HFRStarFocusMeasure,
            n_steps=20,
            n_exposures=1,
            decrease_search_range=True,
            percent_to_cut=40,
            keep_images=True,
            secondary_focus_measure_operators={"FFT": FFTFocusMeasureTan2022()},
        )

        ARAF.run()
        df = ARAF.focus_record

        # Add assertions based on your expectations
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)


if __name__ == "__main__":
    config = CONFIG
    if "path_to_fits" in config and config["path_to_fits"] != "/path/to/fits":
        unittest.main()
