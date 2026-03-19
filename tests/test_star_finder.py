import warnings
from unittest.mock import patch

import numpy as np
import pytest
from photutils.utils.exceptions import NoDetectionsWarning

from astrafocus.star_finder import StarFinder

FWHM = 3.0
BRIGHTEST_PEAK = 65090  # approximate peak of the brightest injected star


class TestGetFallbackThresholds:
    def test_returns_all_when_threshold_is_none(self):
        result = StarFinder.get_fallback_thresholds(None)
        np.testing.assert_array_equal(result, StarFinder.FALLBACK_THRESHOLDS)

    def test_returns_only_values_strictly_below_threshold(self):
        threshold = 4.0
        result = StarFinder.get_fallback_thresholds(threshold)
        assert np.all(result < threshold)

    def test_returns_empty_when_threshold_at_minimum_fallback(self):
        min_fallback = StarFinder.FALLBACK_THRESHOLDS.min()
        assert len(StarFinder.get_fallback_thresholds(min_fallback)) == 0

    def test_returns_empty_when_threshold_below_all_fallbacks(self):
        assert len(StarFinder.get_fallback_thresholds(1.0)) == 0


class TestStarFinderDetection:
    def test_detects_stars(self, sky_image):
        sf = StarFinder(sky_image, fwhm=FWHM)
        assert sf.selected_stars is not None
        assert len(sf.selected_stars) > 0

    def test_selected_stars_sorted_by_flux_descending(self, sky_image):
        sf = StarFinder(sky_image, fwhm=FWHM)
        fluxes = sf.selected_stars["flux"]
        assert all(fluxes[i] >= fluxes[i + 1] for i in range(len(fluxes) - 1))

    def test_saturation_threshold_excludes_bright_stars(self, sky_image):
        saturation = BRIGHTEST_PEAK / 2
        sf = StarFinder(sky_image, fwhm=FWHM, saturation_threshold=saturation)
        assert all(sf.selected_stars["peak"] <= saturation)

    def test_max_stars_caps_number_of_results(self, sky_image):
        sf = StarFinder(sky_image, fwhm=FWHM, max_stars=3)
        assert len(sf.selected_stars) <= 3


class TestStarFinderFallback:
    def test_falls_back_when_initial_threshold_too_high(self, sky_image):
        sf = StarFinder(sky_image, fwhm=FWHM, star_find_threshold=100.0)
        assert sf.selected_stars is not None
        assert len(sf.selected_stars) > 0

    def test_fallback_is_triggered_when_stage1_finds_nothing(self, sky_image):
        original = StarFinder._dao_star_finder
        thresholds_tried = []

        def spy(cleaned_image, fwhm, threshold, brightest=None, peakmax=None):
            thresholds_tried.append(threshold)
            return (
                None
                if len(thresholds_tried) == 1
                else original(cleaned_image, fwhm, threshold, brightest=brightest, peakmax=peakmax)
            )

        with patch.object(StarFinder, "_dao_star_finder", staticmethod(spy)):
            sf = StarFinder(sky_image, fwhm=FWHM)

        assert len(thresholds_tried) >= 2, "Fallback was never triggered"
        assert thresholds_tried[1] < thresholds_tried[0], "Fallback threshold should be lower"
        assert len(sf.selected_stars) > 0

    def test_raises_when_no_sources_found_after_all_fallbacks(self, sky_image):
        with pytest.raises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            StarFinder(sky_image, fwhm=FWHM, absolute_detection_limit=1e9)


class TestFindSourcesClassMethod:
    def test_returns_sources_from_sky_image(self, sky_image):
        sources = StarFinder.find_sources(sky_image, fwhm=FWHM)
        assert sources is not None
        assert len(sources) > 0

    def test_returns_none_when_absolute_detection_limit_too_high(self, sky_image):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            sources = StarFinder.find_sources(sky_image, fwhm=FWHM, absolute_detection_limit=1e9)
        assert sources is None

    def test_accepts_precomputed_background_and_std(self, sky_image):
        import astropy.stats

        _, background, std = astropy.stats.sigma_clipped_stats(sky_image, sigma=3.0)
        sources_auto = StarFinder.find_sources(sky_image, fwhm=FWHM)
        sources_manual = StarFinder.find_sources(sky_image, fwhm=FWHM, std=std, background=background)
        assert len(sources_auto) == len(sources_manual)
