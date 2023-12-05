from abc import abstractmethod
from typing import Optional

import astropy
import numpy as np
import scipy

from astrafocus.focus_measure_operators import FocusMeasureOperator, ImageType
from astrafocus.models.elliptical_moffat_2D import EllipticalMoffat2D
from astrafocus.models.half_flux_radius_2D import HalfFluxRadius2D
from astrafocus.star_finder import StarFinder
from astrafocus.star_fitter import StarFitter
from astrafocus.focus_measure_operators import AnalyticResponseFocusedMeasureOperator
from astrafocus.utils.logger import configure_logger

logger = configure_logger()


class StarSizeFocusMeasure(AnalyticResponseFocusedMeasureOperator):
    def __init__(
        self,
        ref_image,
        model,
        fwhm=2.0,
        star_find_threshold=5.0,
        absolute_detection_limit=0.0,
        saturation_threshold=None,
    ) -> None:
        self.star_finder = StarFinder(
            ref_image,
            fwhm=fwhm,
            star_find_threshold=star_find_threshold,
            absolute_detection_limit=absolute_detection_limit,
            saturation_threshold=saturation_threshold,
        )
        self.star_fitter = StarFitter(model)
        self.optimised_parameters = None
        logger.info(self.star_fitter)

    def measure_focus(self, image: ImageType, cutout_size=15, **kwargs) -> float:
        selected_stars = self.star_finder.selected_stars
        star_size_arr = self.star_fitter.calculate_star_sizes_of_selection(
            image,
            selected_stars=selected_stars,
            cutout_size=cutout_size,
        )
        return np.mean(star_size_arr)

    def __repr__(self) -> str:
        return (
            f"StarSizeFocusMeasure(self.star_finder={self.star_finder!r}, "
            f"star_fitter={self.star_fitter!r})"
        )


class GaussianStarFocusMeasure(StarSizeFocusMeasure):
    """
    from astrafocus.utils.load_fits_from_directory import load_fits_from_directory
    fits_directory = "path_to_fits_files"
    image_data, headers, focus_pos = load_fits_from_directory(fits_directory)

    image = image_data[0]
    gsfm = GaussianStarFocusMeasure(image, fwhm=2.0, star_find_threshold=8.0)
    gsfm.star_finder.selected_stars

    import matplotlib.pyplot as plt
    fm_vals = [gsfm.measure_focus(image) for image in image_data]
    plt.plot(focus_pos, fm_vals, ls='', marker='.'); plt.show()
    
    plot_focus_response_curve(gsfm, image_data, focus_pos, plot_name='gaussian_star.pdf')
    """

    def __init__(
        self,
        ref_image,
        fwhm=2.0,
        star_find_threshold=5.0,
        absolute_detection_limit=0.0,
        saturation_threshold=None,
    ) -> None:
        model = astropy.modeling.models.Gaussian2D
        super().__init__(
            ref_image,
            model,
            fwhm=fwhm,
            star_find_threshold=star_find_threshold,
            absolute_detection_limit=absolute_detection_limit,
            saturation_threshold=saturation_threshold,
        )

    def fit_focus_response_curve(self, focus_pos, measured_focus):
        popt, pcov = GaussianStarFocusMeasure.fit_hyperbola(focus_pos, measured_focus)
        self.optimised_parameters = popt

        return popt[-2]

    def get_focus_response_curve_fit(self, focus_pos):
        if self.optimised_parameters is None:
            return None
        predicted_focus = self.hyperbola(focus_pos, *self.optimised_parameters)
        return predicted_focus

    @staticmethod
    def fit_hyperbola(x, y):
        popt, pcov = scipy.optimize.curve_fit(
            GaussianStarFocusMeasure.hyperbola, x, y, p0=(1, 1, np.mean(x), np.min(y)) # , np.min(y)
        )
        return popt, pcov

    @staticmethod
    def hyperbola(x, a=1, b=1, x_0=0, y_0=0):
        """
        Notes
        -----
        If we want the hyperbola to be north-south opening, we need to use the parametrisation
        (y-y_0)**2/b**2 - (x-x_0)**2/a**2 = 1

        Resulting in the following parametrisation of the north opening hyperbola
        y = b * sqrt(1 + ((x-x_0)/a)**2) + y_0

        Examples
        --------
        >>> x = np.linspace(-1, 2)
        >>> plt.plot(x, hyperbola(x=x, a=1, b=1, x_0=0, y_0=-1)); plt.show()
        """
        y = b * np.sqrt(1 + ((x - x_0) / a) ** 2) + y_0

        return y


class HFRStarFocusMeasure(StarSizeFocusMeasure):
    """
    from astrafocus.utils.load_fits_from_directory import load_fits_from_directory
    fits_directory = "path_to_fits_files"
    image_data, headers, focus_pos = load_fits_from_directory(fits_directory)


    image = image_data[0]
    hfrfm = HFRStarFocusMeasure(image, fwhm=2.0, star_find_threshold=8.0)
    hfrfm.star_finder.selected_stars


    import matplotlib.pyplot as plt
    fm_vals = [hfrfm.measure_focus(image) for image in image_data]
    plt.plot(focus_pos, fm_vals/np.min(fm_vals), ls='', marker='.')
    plt.plot(focus_pos, hfrfm.linear_V_curve(focus_pos, *(-0.003, 0.003, np.mean(focus_pos), np.min(fm_vals/np.min(fm_vals)))))
    plt.show()

    plot_focus_response_curve(hfrfm, image_data, focus_pos, plot_name='HFR_star.pdf')


    hfrfm.fit_focus_response_curve(focus_pos, fm_vals/np.min(fm_vals))
    hfrfm.get_focus_response_curve_fit(focus_pos)
    focus_pos_fine = np.linspace(np.min(focus_pos), np.max(focus_pos))
    """

    def __init__(
        self,
        ref_image,
        fwhm=2.0,
        star_find_threshold=5.0,
        absolute_detection_limit=0.0,
        saturation_threshold=None,
    ) -> None:
        model = HalfFluxRadius2D
        super().__init__(
            ref_image,
            model,
            fwhm=fwhm,
            star_find_threshold=star_find_threshold,
            absolute_detection_limit=absolute_detection_limit,
            saturation_threshold=saturation_threshold,
        )

    def fit_focus_response_curve(self, focus_pos, measured_focus):
        popt, pcov = HFRStarFocusMeasure.fit_linear_V_curve(focus_pos, measured_focus)
        self.optimised_parameters = popt

        return popt[2]

    def get_focus_response_curve_fit(self, focus_pos):
        if self.optimised_parameters is None:
            return None
        predicted_focus = self.linear_V_curve(focus_pos, *self.optimised_parameters)
        return predicted_focus

    @staticmethod
    def fit_linear_V_curve(x, y):
        popt, pcov = scipy.optimize.curve_fit(
            HFRStarFocusMeasure.linear_V_curve,
            x,
            y,
            p0=(-0.1, 0.1, np.mean(x), np.min(y)),
            # loss="soft_l1",
        )
        return popt, pcov

    @staticmethod
    def linear_V_curve(x, slope_left=-1, slope_right=1, x_centre=0, intercept=0):
        """
        Examples
        --------
        >>> x = np.linspace(-2, 2, 200)
        >>> plt.plot(x, linear_V_curve(x=x)); plt.show()

        y_min = slope_left*v_centre + intercept_left = slope_right*v_centre + intercept_right
        (slope_left-slope_right)*v_centre =  intercept_right - intercept_left
        v_centre = (intercept_right-intercept_left)/(slope_left - slope_right)

        intercept_diff = (intercept_right - intercept_left)

        v_centre = intercept_diff /(slope_left - slope_right)

        slope_left*(x-v_centre) + b = slope_left*x + (b - slope_left**v_centre)
        slope_right*(x-v_centre) + b = slope_right*x + (b - v_centre*slope_right)
        intercept_diff = v_centre*(slope_left-slope_right)

        v_centre = (intercept_right - intercept_left) / (slope_left - slope_right)
        """
        y = np.where(
            x - x_centre <= 0,
            slope_left * (x - x_centre) + intercept,
            slope_right * (x - x_centre) + intercept,
        )

        return y

    @staticmethod
    def linear_V_curve_prime(x, slope_left=-1, slope_right=1, intercept_left=0, intercept_right=0):
        """
        Examples
        --------
        >>> x = np.linspace(-2, 2, 200)
        >>> plt.plot(x, linear_V_curve(x=x)); plt.show()

        slope_left*v_centre + intercept_left = slope_right*v_centre + intercept_right
        (slope_left-slope_right)*v_centre =  intercept_right - intercept_left
        v_centre = (intercept_right-intercept_left)/(slope_left - slope_right)
        """
        v_centre = (intercept_right - intercept_left) / (slope_left - slope_right)

        left_mask = x <= v_centre
        y = np.zeros_like(x)
        y[left_mask] = slope_left * x[left_mask] + intercept_left
        y[~left_mask] = slope_right * x[~left_mask] + intercept_right

        return y
