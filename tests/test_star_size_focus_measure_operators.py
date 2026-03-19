import warnings

import numpy as np
import pytest
from photutils.utils.exceptions import NoDetectionsWarning

from astrafocus.star_size_focus_measure_operators import (
    GaussianStarFocusMeasure,
    HFRStarFocusMeasure,
)

FWHM = 3.0


class TestMeasureFocus:
    @pytest.mark.parametrize("cls", [GaussianStarFocusMeasure, HFRStarFocusMeasure])
    def test_returns_finite_float(self, cls, sky_image):
        op = cls(ref_image=sky_image, fwhm=FWHM)
        result = op.measure_focus(sky_image)
        assert np.isfinite(result)

    @pytest.mark.parametrize("cls", [GaussianStarFocusMeasure, HFRStarFocusMeasure])
    def test_raises_when_no_stars_found(self, cls, sky_image):
        op = cls(absolute_detection_limit=1e9)
        with pytest.raises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            op.measure_focus(sky_image)


class TestLazyInitialisation:
    def test_star_finder_is_none_before_first_call(self):
        op = HFRStarFocusMeasure()
        assert op.star_finder is None

    def test_star_finder_initialised_on_first_measure_focus(self, sky_image):
        op = HFRStarFocusMeasure(fwhm=FWHM)
        op.measure_focus(sky_image)
        assert op.star_finder is not None

    def test_star_finder_reused_on_subsequent_calls(self, sky_image):
        op = HFRStarFocusMeasure(fwhm=FWHM)
        op.measure_focus(sky_image)
        finder_after_first = op.star_finder
        op.measure_focus(sky_image)
        assert op.star_finder is finder_after_first

    def test_get_star_finder_raises_when_uninitialised_and_no_image(self):
        op = HFRStarFocusMeasure()
        with pytest.raises(ValueError):
            op._get_star_finder(image=None)


class TestParameterPropagation:
    def test_max_stars_reaches_star_finder(self, sky_image):
        op = HFRStarFocusMeasure(ref_image=sky_image, fwhm=FWHM, max_stars=5)
        assert op.star_finder.max_stars == 5

    def test_max_stars_caps_selected_stars(self, sky_image):
        op = HFRStarFocusMeasure(ref_image=sky_image, fwhm=FWHM, max_stars=3)
        assert len(op.star_finder.selected_stars) <= 3

    def test_saturation_threshold_reaches_star_finder(self, sky_image):
        op = HFRStarFocusMeasure(ref_image=sky_image, fwhm=FWHM, saturation_threshold=10000)
        assert op.star_finder.saturation_threshold == 10000
        assert all(op.star_finder.selected_stars["peak"] <= 10000)
