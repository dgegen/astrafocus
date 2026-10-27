"""Tests of the star-size focus measures on synthetic frames with a high sky, banding and hot pixels."""

import warnings

import numpy as np
import pytest
from synthetic_frames import (
    BEST_FOCUS,
    DONUT_INNER_FRACTION,
    SEEING_SIGMA,
    SHAPE,
    STAR_POSITIONS,
    donut_radius_at,
    hot_pixels,
    make_frame,
    star_profile,
)

from astrafocus.autofocuser import NonParametricResponseAutofocuser
from astrafocus.focus_measure_operators import FocusMeasureOperator
from astrafocus.interface.simulation import CabaretDeviceSimulator
from astrafocus.models.half_flux_radius_2D import HalfFluxRadius2D
from astrafocus.star_finder import StarFinder, prepare_image
from astrafocus.star_fitter import StarFitter
from astrafocus.star_size_focus_measure_operators import (
    GaussianStarFocusMeasure,
    HFRStarFocusMeasure,
)
from astrafocus.utils.focus_curve import focus_curve_snr

FWHM = 3.0
SWEEP = np.arange(BEST_FOCUS - 1000, BEST_FOCUS + 1001, 250)


def annulus_mean_radius(outer_radius, inner_fraction=DONUT_INNER_FRACTION):
    """Flux-weighted mean radius of a uniform annulus, i.e. the HFR estimate astrafocus uses."""
    r_out, r_in = outer_radius, inner_fraction * outer_radius
    return 2 / 3 * (r_out**3 - r_in**3) / (r_out**2 - r_in**2)


@pytest.fixture(scope="module")
def sweep_frames():
    return {int(pos): make_frame(pos, seed=int(pos)) for pos in SWEEP}


@pytest.fixture(scope="module")
def defocused_reference():
    return make_frame(BEST_FOCUS + 300, seed=1)


class TestFocusCurve:
    @pytest.mark.parametrize("cls", [HFRStarFocusMeasure, GaussianStarFocusMeasure])
    def test_minimum_at_true_focus(self, cls, sweep_frames, defocused_reference):
        op = cls(ref_image=defocused_reference, fwhm=FWHM)
        measures = np.array([op.measure_focus(sweep_frames[pos]) for pos in SWEEP])

        assert SWEEP[np.argmin(measures)] == BEST_FOCUS
        best = np.argmin(measures)
        assert np.all(np.diff(measures[: best + 1]) < 0)
        assert np.all(np.diff(measures[best:]) > 0)

    def test_curve_is_not_flat(self, sweep_frames, defocused_reference):
        op = HFRStarFocusMeasure(ref_image=defocused_reference, fwhm=FWHM)
        measures = np.array([op.measure_focus(sweep_frames[pos]) for pos in SWEEP])
        assert focus_curve_snr(SWEEP, measures) > 8
        assert measures.max() > 10 * measures.min()


class TestStarSize:
    SIGMA = 2.0

    @pytest.fixture(scope="class")
    @classmethod
    def gaussian_frame(cls):
        return make_frame(BEST_FOCUS, sigma=cls.SIGMA, seed=3)

    def test_hfd_of_gaussian_star(self, gaussian_frame):
        # With the mean-radius estimate, HFR = sqrt(pi / 2) * sigma, so HFD = 2.51 * sigma.
        op = HFRStarFocusMeasure(ref_image=gaussian_frame, fwhm=FWHM)
        hfd = 2 * op.measure_focus(gaussian_frame)
        assert hfd == pytest.approx(2 * np.sqrt(np.pi / 2) * self.SIGMA, rel=0.03)

    def test_fwhm_of_gaussian_star(self, gaussian_frame):
        op = GaussianStarFocusMeasure(ref_image=gaussian_frame, fwhm=FWHM)
        fwhm = op.measure_focus(gaussian_frame)
        assert fwhm == pytest.approx(2 * np.sqrt(2 * np.log(2)) * self.SIGMA, rel=0.03)

    def test_does_not_depend_on_cutout_size(self, gaussian_frame):
        sizes = [
            HFRStarFocusMeasure(ref_image=gaussian_frame, fwhm=FWHM, cutout_size=cutout_size).measure_focus(
                gaussian_frame
            )
            for cutout_size in (10, 15, 40)
        ]
        assert np.ptp(sizes) < 0.02 * np.mean(sizes)

    def test_large_donut_fits_default_cutout(self, sweep_frames, defocused_reference):
        # The donut has a radius of 55 px, but the default cutout half-width is 15 px.
        position = BEST_FOCUS - 1000
        op = HFRStarFocusMeasure(ref_image=defocused_reference, fwhm=FWHM)
        hfr = op.measure_focus(sweep_frames[position])
        assert hfr == pytest.approx(annulus_mean_radius(donut_radius_at(position)), rel=0.03)

    def test_donut_hole_larger_than_cutout(self, sweep_frames):
        # The dark centre of this donut (radius 19 px) fills the first cutout (half-width 15 px).
        position = BEST_FOCUS - 1000
        x, y = STAR_POSITIONS[0]
        star_data = StarFitter.extract_star(
            sweep_frames[position], x, y, cutout_size=15, max_cutout_size=120
        )
        assert np.count_nonzero(star_data) > 1000


class TestRobustness:
    def test_star_moves_between_frames(self):
        reference = make_frame(BEST_FOCUS, seed=4)
        shifted = make_frame(BEST_FOCUS, shift=(8.0, -6.0), seed=5)
        op = HFRStarFocusMeasure(ref_image=reference, fwhm=FWHM)
        assert op.measure_focus(shifted) == pytest.approx(op.measure_focus(reference), rel=0.03)

    def test_star_finder_ignores_hot_pixels(self):
        frame = make_frame(BEST_FOCUS, seed=6)
        hot_x, hot_y, _ = hot_pixels()

        def n_on_hot_pixels(stars):
            distance = np.hypot(stars["x_centroid"][:, None] - hot_x, stars["y_centroid"][:, None] - hot_y)
            return np.count_nonzero(distance.min(axis=1) < 1.0)

        assert n_on_hot_pixels(StarFinder(frame, fwhm=FWHM, remove_hot_pixels=False).selected_stars) > 0
        stars = StarFinder(frame, fwhm=FWHM).selected_stars
        assert n_on_hot_pixels(stars) == 0

        true_positions = np.array(STAR_POSITIONS)
        distance = np.hypot(
            stars["x_centroid"][:, None] - true_positions[:, 0],
            stars["y_centroid"][:, None] - true_positions[:, 1],
        )
        assert np.all(distance.min(axis=0) < 1.0)

    def test_hot_pixel_is_not_a_star(self):
        image = np.random.default_rng(7).normal(3000.0, 50.0, size=(100, 100))
        image[50, 50] += 20_000
        with pytest.raises(ValueError):
            StarFitter.extract_star(image, x=50, y=50)

    def test_star_finder_with_sky_gradient(self):
        frame = make_frame(BEST_FOCUS, seed=10) + np.linspace(0, 1500, SHAPE[1])[None, :]
        op = HFRStarFocusMeasure(ref_image=frame, fwhm=FWHM)
        assert op.measure_focus(frame) == pytest.approx(np.sqrt(np.pi / 2) * SEEING_SIGMA, rel=0.05)

    def test_reference_far_from_focus(self, sweep_frames):
        # Stars are donuts with a radius of 55 px; the star finder retries with a wider kernel.
        op = HFRStarFocusMeasure(ref_image=make_frame(BEST_FOCUS + 1000, seed=11), fwhm=FWHM)
        measures = np.array([op.measure_focus(sweep_frames[pos]) for pos in SWEEP])
        assert SWEEP[np.argmin(measures)] == BEST_FOCUS

    def test_noise_peak_does_not_hide_donuts(self):
        # With this seed, the narrow kernel finds only one noise peak on the defocused reference,
        # and the donuts are only found with the wider kernels.
        reference = CabaretDeviceSimulator.generate_image(focus_position=9_550, texp=1.0, seed=3)
        assert len(StarFinder(reference, fwhm=2.0, star_find_threshold=5.0).selected_stars) == 1
        op = HFRStarFocusMeasure(ref_image=reference)
        assert len(op.star_finder.selected_stars) >= 5
        sizes = [
            op.measure_focus(CabaretDeviceSimulator.generate_image(pos, texp=1.0, seed=pos))
            for pos in (9_600, 10_000, 10_400)
        ]
        assert sizes[1] < 0.5 * min(sizes[0], sizes[2])

    def test_combine_star_sizes_does_not_fall_back_to_static_sources(self):
        faint_star = np.array([np.nan, np.nan, 3.0, 2.0, 3.0, np.nan, np.nan])
        hot_pixel = np.full(faint_star.size, 0.75)
        focus_pos = np.arange(faint_star.size) * 100.0
        combined = HFRStarFocusMeasure().combine_star_sizes(
            np.column_stack([faint_star, hot_pixel]), focus_pos
        )
        assert np.all(np.isnan(combined))

    def test_combine_star_sizes_drops_noise_sources(self):
        focus_pos = np.linspace(9000, 11000, 20)
        star = 2 + np.abs(focus_pos - 10_000) / 50
        noise_source = 12 + np.random.default_rng(12).normal(0, 6, focus_pos.size)
        combined = HFRStarFocusMeasure().combine_star_sizes(
            np.column_stack([star, noise_source, noise_source[::-1]]), focus_pos
        )
        np.testing.assert_allclose(combined, star)

    def test_combine_star_sizes_drops_isolated_point(self):
        # As on Callisto run 2: the first frame was taken before the focuser reached its position,
        # so a star near focus is small at the far end of the sweep, and lost in the next frames.
        focus_pos = np.linspace(9000, 11000, 20)
        star = 2 + np.abs(focus_pos - 10_000) / 50
        star[1:5] = np.nan
        star[0] = 1.5
        op = HFRStarFocusMeasure()
        combined = op.combine_star_sizes(star[:, None], focus_pos)
        assert np.isnan(combined[0])
        ok = np.isfinite(combined)
        assert op.fit_focus_response_curve(focus_pos[ok], combined[ok]) == pytest.approx(10_000, abs=50)

    @pytest.mark.parametrize("lost", [[], [3, 4]])
    def test_combine_star_sizes_keeps_best_focus_at_edge(self, lost):
        focus_pos = np.linspace(9000, 11000, 20)
        star = 2 + np.abs(focus_pos - 9000) / 50
        star[lost] = np.nan
        op = HFRStarFocusMeasure()
        combined = op.combine_star_sizes(star[:, None], focus_pos)
        np.testing.assert_allclose(combined, star)
        ok = np.isfinite(combined)
        assert op.fit_focus_response_curve(focus_pos[ok], combined[ok]) == pytest.approx(9_000, abs=50)

    def test_combine_star_sizes_drops_static_and_rarely_measured_stars(self):
        star = np.array([9.0, 6.0, 3.0, 2.0, 3.0, 6.0, 9.0])
        hot_pixel = np.full(star.size, 0.75)
        rarely_measured = np.array([np.nan, np.nan, np.nan, 0.5, np.nan, np.nan, 20.0])
        star_sizes = np.column_stack([star, star * 1.1, hot_pixel, hot_pixel, rarely_measured])

        focus_pos = np.arange(star.size) * 100.0
        combined = HFRStarFocusMeasure().combine_star_sizes(star_sizes, focus_pos)
        np.testing.assert_allclose(combined, star * 1.05)

    def test_hfr_model_stores_radius(self):
        rng = np.random.default_rng(8)
        y, x = np.indices((60, 60))
        sigma = 3.0
        star_data = np.exp(-((x - 30.3) ** 2 + (y - 29.6) ** 2) / (2 * sigma**2))
        star_data += rng.normal(0, 1e-4, star_data.shape)
        star_fitter = StarFitter(HalfFluxRadius2D).fit(star_data)
        assert star_fitter.star_size == pytest.approx(np.sqrt(np.pi / 2) * sigma, rel=0.01)


class TestBrightNeighbour:
    """A star next to a much brighter one must not be measured on the bright one."""

    FAINT = (100.0, 100.0)

    @staticmethod
    def frame(faint_flux, distance, donut_radius=0.0, seed=11):
        # The neighbour is a compact star with a flux of 1e6.
        shape = (200, 300)
        x, y = TestBrightNeighbour.FAINT
        image = star_profile(shape, x, y, faint_flux, donut_radius)
        image += star_profile(shape, x + distance, y, 1e6, 0.0)
        return np.random.default_rng(seed).normal(3000.0, 50.0, size=shape) + image

    def test_invisible_star_is_not_replaced_by_neighbour(self):
        # No light in the first cutout, so it grows until it reaches the neighbour 60 px away.
        with pytest.raises(ValueError):
            StarFitter.extract_star(
                self.frame(2e3, distance=60), *self.FAINT, cutout_size=15, max_cutout_size=120
            )

    def test_faint_donut_is_not_replaced_by_neighbour(self):
        # The donut (radius 35 px) reaches the edge of the cutout until its half-width is 60 px,
        # which also takes in the neighbour, 55 px to the left.
        frame = self.frame(3e5, distance=-55, donut_radius=35.0)
        star_data = StarFitter.extract_star(frame, *self.FAINT, cutout_size=15, max_cutout_size=120)
        assert np.sum(star_data) == pytest.approx(3e5, rel=0.1)

    def test_residue_in_donut_centre_is_not_the_star(self):
        # A compact source (e.g. a cosmic ray) in the dark centre of a donut of radius 80 px. The
        # dark centre (radius 28 px) fills the first cutout.
        shape = (260, 260)
        image = star_profile(shape, 130.0, 130.0, 3e6, 80.0) + star_profile(shape, 130.0, 130.0, 2e4, 0.0)
        image += np.random.default_rng(12).normal(3000.0, 50.0, size=shape)
        star_data = StarFitter.extract_star(image, 130.0, 130.0, cutout_size=15, max_cutout_size=120)
        assert np.sum(star_data) == pytest.approx(3e6, rel=0.1)

    def test_reference_sources_are_moved_to_star_centres(self):
        # Far from focus, the star finder places sources on the donut rings (radius 55 px).
        op = HFRStarFocusMeasure(ref_image=make_frame(BEST_FOCUS + 1000, seed=13), fwhm=FWHM)
        stars = op.star_finder.selected_stars
        for x, y in STAR_POSITIONS:
            distance = np.hypot(stars["x_centroid"] - x, stars["y_centroid"] - y)
            assert np.count_nonzero(distance < 60) == 1
            assert np.min(distance) < 2.0


class TestLargeDonutSky:
    """
    One donut about 200 px across, in frames not much larger than it, as far from focus on Callisto.
    As there, the ring is about 250 above the sky, and the row banding has a standard deviation of
    about 40.
    """

    OFFSET = 1800  # focuser steps from best focus; the donut radius is 99 px
    SWEEP = np.arange(BEST_FOCUS - OFFSET, BEST_FOCUS + OFFSET + 1, 300)
    STAR_FLUX = 7e6
    BANDING_AMPLITUDE = 45.0

    @classmethod
    def frame(cls, focus_position, size, gradient=0.0, seed=1):
        centre = size / 2
        image = make_frame(
            focus_position,
            shape=(size, size),
            star_positions=((centre, centre),),
            star_flux=cls.STAR_FLUX,
            banding_amplitude=cls.BANDING_AMPLITUDE,
            seed=seed,
        )
        return image + np.linspace(0, gradient, size)[None, :]

    @pytest.mark.parametrize("gradient", [0.0, 1500.0])
    @pytest.mark.parametrize("size", [512, 384, 320])
    def test_sky_keeps_donut_light(self, size, gradient):
        radius = donut_radius_at(BEST_FOCUS + self.OFFSET)
        centre = size / 2
        data = prepare_image(self.frame(BEST_FOCUS + self.OFFSET, size, gradient)).data

        yy, xx = np.indices(data.shape)
        r = np.hypot(xx - centre, yy - centre)
        true_flux = star_profile(data.shape, centre, centre, self.STAR_FLUX, radius).sum()
        assert data[r < radius + 10].sum() == pytest.approx(true_flux, rel=0.15)
        assert abs(np.median(data[r > radius + 15])) < 3.0

    @pytest.mark.parametrize("gradient", [0.0, 1500.0])
    @pytest.mark.parametrize("size", [384, 320])
    def test_minimum_at_true_focus_in_small_frame(self, size, gradient):
        op = HFRStarFocusMeasure(ref_image=self.frame(BEST_FOCUS + 600, size, gradient, seed=2), fwhm=FWHM)
        measures = [op.measure_focus(self.frame(pos, size, gradient, seed=int(pos))) for pos in self.SWEEP]
        assert self.SWEEP[np.argmin(measures)] == BEST_FOCUS


class NoiseFocusMeasure(FocusMeasureOperator):
    """A focus measure with no focus information, like a measure dominated by the sky."""

    def __init__(self, **kwargs):
        self.rng = np.random.default_rng(9)

    def measure_focus(self, image, **kwargs):
        return 45.8 + self.rng.normal(0, 0.003)


def test_flat_curve_fails_and_resets_focuser():
    simulator = CabaretDeviceSimulator()
    initial_position = simulator.focuser.position
    autofocuser = NonParametricResponseAutofocuser(
        autofocus_device_manager=simulator,
        exposure_time=0.1,
        focus_measure_operator=NoiseFocusMeasure(),
        n_steps=(20,),
    )
    assert autofocuser.run() is False
    assert autofocuser.best_focus_position is None
    assert simulator.focuser.position == initial_position


class MostlyFailingFocusMeasure(FocusMeasureOperator):
    """A focus measure that can measure only a few frames, like stars too faint to see far from focus."""

    def __init__(self, **kwargs):
        self.n_calls = 0

    def measure_focus(self, image, **kwargs):
        self.n_calls += 1
        if self.n_calls % 5:
            raise ValueError("No star light above the noise.")
        return float(self.n_calls)


def test_too_few_measured_points_fails():
    simulator = CabaretDeviceSimulator()
    autofocuser = NonParametricResponseAutofocuser(
        autofocus_device_manager=simulator,
        exposure_time=0.1,
        focus_measure_operator=MostlyFailingFocusMeasure(),
        n_steps=(20,),
    )
    assert autofocuser.run() is False
    assert autofocuser.best_focus_position is None


def test_sweep_with_failed_positions_does_not_warn():
    autofocuser = NonParametricResponseAutofocuser(
        autofocus_device_manager=CabaretDeviceSimulator(),
        exposure_time=0.1,
        focus_measure_operator=MostlyFailingFocusMeasure(),
        n_steps=(10,),
        n_exposures=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert autofocuser._run_sweep(autofocuser.integer_linspace(*autofocuser.search_range, 10), 2)
    assert autofocuser.focus_record["focus_measure"].isna().sum() == 16
