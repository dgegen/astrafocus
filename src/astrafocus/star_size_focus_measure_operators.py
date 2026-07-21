import astropy
import numpy as np
import numpy.typing as npt
import scipy

from astrafocus.focus_measure_operators import (
    AnalyticResponseFocusedMeasureOperator,
    ImageType,
)
from astrafocus.models.half_flux_radius_2D import HalfFluxRadius2D
from astrafocus.star_finder import StarFinder
from astrafocus.star_fitter import StarFitter
from astrafocus.utils.logger import get_logger

logger = get_logger()


class StarSizeFocusMeasure(AnalyticResponseFocusedMeasureOperator):
    """
    Focus measure operator that derives focus quality from the fitted size of detected stars.

    Stars are located in the image with a `StarFinder`, and `model` is fit to each detected
    star with a `StarFitter` (see `StarFitter.star_size`, e.g. FWHM). The mean fitted size
    across stars is returned as the focus measure; smaller values indicate better focus.

    Subclasses fix `model` to a specific profile (see `GaussianStarFocusMeasure`,
    `HFRStarFocusMeasure`) and implement `fit_focus_response_curve` /
    `get_focus_response_curve_fit` to locate the optimal focus position from a sweep of
    focus measures.

    Parameters
    ----------
    model : astropy.modeling.core._ModelMeta
        The model class fit to each star, passed through to `StarFitter`. Commonly
        `astropy.modeling.models.Gaussian2D` or `astrafocus.models.HalfFluxRadius2D`.
    ref_image : 2D array_like, optional
        Reference image used to initialise the `StarFinder` eagerly. If None (default), the
        `StarFinder` is instead created lazily from the first image passed to `measure_focus`.
    fwhm : float, optional
        Expected FWHM (pixels) of stars, forwarded to `StarFinder`. Default is 2.0.
    star_find_threshold : float, optional
        Initial detection threshold in units of background standard deviation, forwarded to
        `StarFinder`. Default is 5.0.
    absolute_detection_limit : float, optional
        Hard floor for detection in ADU/counts, forwarded to `StarFinder`. Default is 0.0.
    cutout_size : int, optional
        Half-width (pixels) of the square cutout extracted around each star for fitting.
        Default is 15.
    saturation_threshold : float, optional
        Maximum allowed pixel value; stars with brighter peaks are rejected. Forwarded to
        `StarFinder`. Default is None (no cap).
    max_stars : int, optional
        Maximum number of stars to use, sorted by brightness. Forwarded to `StarFinder`.
        Default is 100.
    sharpness_range : tuple of float, optional
        (sharplo, sharphi) bounds passed to DAOStarFinder, forwarded to `StarFinder`.
        Default is (0.05, 1.0).
    **kwargs
        Accepted for interface compatibility with other `FocusMeasureOperator`
        implementations; unused.

    Attributes
    ----------
    star_finder : StarFinder or None
        Locates stars in an image. None until a reference or measurement image has
        initialised it.
    star_fitter : StarFitter
        Fits `model` to individual stars and reports their size.
    cutout_size : int
        Half-width (pixels) of the cutout used when fitting each star.
    optimised_parameters : numpy.ndarray or None
        Parameters of the fitted focus response curve, set by `fit_focus_response_curve`
        (subclasses only) and consumed by `get_focus_response_curve_fit`. None until fit.

    Raises
    ------
    ValueError
        If the `StarFinder` cannot find enough stars to measure focus.
    """

    def __init__(
        self,
        model: astropy.modeling.core._ModelMeta,
        ref_image: ImageType | None = None,
        fwhm: float = 2.0,
        star_find_threshold: float = 5.0,
        absolute_detection_limit: float = 0.0,
        cutout_size: int = 15,
        saturation_threshold: float | None = None,
        max_stars: int = 100,
        sharpness_range: tuple[float, float] = (0.05, 1.0),
        **kwargs,
    ) -> None:
        self._star_finder_kwargs = {
            "fwhm": fwhm,
            "star_find_threshold": star_find_threshold,
            "absolute_detection_limit": absolute_detection_limit,
            "saturation_threshold": saturation_threshold,
            "max_stars": max_stars,
            "sharpness_range": sharpness_range,
        }
        self.star_finder = (
            StarFinder(ref_image, **self._star_finder_kwargs) if ref_image is not None else None
        )
        self.star_fitter = StarFitter(model)
        self.cutout_size = cutout_size
        self.optimised_parameters = None

    def _get_star_finder(self, image: ImageType | None = None) -> StarFinder:
        """
        Return `self.star_finder`, lazily initialising it from `image` if needed.

        Parameters
        ----------
        image : 2D array_like, optional
            Image to initialise the `StarFinder` with, if it does not already exist.
            Ignored if `self.star_finder` is already set.

        Returns
        -------
        StarFinder
            The (possibly newly created) star finder.

        Raises
        ------
        ValueError
            If `self.star_finder` is not initialised and no `image` was provided.
        """
        if self.star_finder is None:
            if image is None:
                raise ValueError("StarFinder is not initialised and no image was provided to initialise it.")
            self.star_finder = StarFinder(image, **self._star_finder_kwargs)
        return self.star_finder

    def measure_focus(self, image: ImageType, cutout_size: int | None = None, **kwargs) -> float:
        """
        Compute the mean fitted star size in `image`, used as the focus measure.

        Stars are located via `self.star_finder` (initialised from `image` on first call if
        not already set), and `self.star_fitter` is fit to a cutout around each star.

        Parameters
        ----------
        image : ImageType
            Image to measure focus in.
        cutout_size : int, optional
            Half-width (pixels) of the square cutout extracted around each star. Defaults to
            `self.cutout_size`.
        **kwargs
            Accepted for interface compatibility with `FocusMeasureOperator.measure_focus`;
            unused.

        Returns
        -------
        float
            Mean fitted star size (e.g. FWHM) across the selected stars. Smaller is better.

        Raises
        ------
        ValueError
            If the `StarFinder` could not find enough stars in `image`.
        """
        if cutout_size is None:
            cutout_size = self.cutout_size

        try:
            selected_stars = self._get_star_finder(image).selected_stars
        except ValueError:
            raise ValueError(
                "StarFinder could not find enough stars to measure focus. "
                "Adjust the star finder parameters or check image quality."
            )

        star_size_arr = self.star_fitter.calculate_star_sizes_of_selection(
            image,
            selected_stars=selected_stars,
            cutout_size=cutout_size,
        )
        return np.mean(star_size_arr)

    def __repr__(self) -> str:
        return (
            f"StarSizeFocusMeasure(star_finder={self.star_finder!r}, "
            f"star_fitter={self.star_fitter!r}, cutout_size={self.cutout_size!r})"
        )

    def __str__(self) -> str:
        return (
            f"StarSizeFocusMeasure(star_finder={self.star_finder}, "
            f"star_fitter={self.star_fitter}, cutout_size={self.cutout_size})"
        )


class GaussianStarFocusMeasure(StarSizeFocusMeasure):
    """
    `StarSizeFocusMeasure` that fits a 2D Gaussian to each star.

    Star size is the average of the fitted Gaussian's `x_fwhm`/`y_fwhm` (see
    `StarFitter.star_size`). The focus response curve (star size vs. focus position) is
    modelled as a north-opening hyperbola via `hyperbola`/`fit_hyperbola`; its vertex gives
    the estimated in-focus position.

    Parameters
    ----------
    ref_image, fwhm, star_find_threshold, absolute_detection_limit, cutout_size,
    saturation_threshold, max_stars, **kwargs
        See `StarSizeFocusMeasure`. `model` is fixed to `astropy.modeling.models.Gaussian2D`.

    Examples
    --------
    >>> from astrafocus.interface.simulation import CabaretDeviceSimulator
    >>> image = CabaretDeviceSimulator.generate_image()
    >>> gsfm = GaussianStarFocusMeasure(image, fwhm=2.0, star_find_threshold=8.0)
    >>> bool(gsfm.measure_focus(image) > 0)
    True

    Fitting a response curve to a sweep of images taken at different focus positions:

    >>> import numpy as np
    >>> focus_pos = np.array([1, 2, 3, 4, 5])
    >>> fm_vals = np.array([5.0, 3.0, 2.0, 3.0, 5.0])
    >>> best_focus = gsfm.fit_focus_response_curve(focus_pos, fm_vals)
    >>> predicted = gsfm.get_focus_response_curve_fit(focus_pos)
    """

    def __init__(
        self,
        ref_image: ImageType | None = None,
        fwhm: float = 2.0,
        star_find_threshold: float = 5.0,
        absolute_detection_limit: float = 0.0,
        cutout_size: int = 15,
        saturation_threshold: float | None = None,
        max_stars: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(
            model=astropy.modeling.models.Gaussian2D,
            ref_image=ref_image,
            fwhm=fwhm,
            star_find_threshold=star_find_threshold,
            absolute_detection_limit=absolute_detection_limit,
            cutout_size=cutout_size,
            saturation_threshold=saturation_threshold,
            max_stars=max_stars,
        )

    def fit_focus_response_curve(self, focus_pos: npt.ArrayLike, measured_focus: npt.ArrayLike) -> float:
        """Fit a north-opening hyperbola to a focus sweep and return the estimated best focus."""
        popt, pcov = GaussianStarFocusMeasure.fit_hyperbola(focus_pos, measured_focus)
        self.optimised_parameters = popt

        return popt[-2]

    def get_focus_response_curve_fit(self, focus_pos: npt.ArrayLike) -> np.ndarray | None:
        """Evaluate the fitted hyperbola from `fit_focus_response_curve` at `focus_pos`."""
        if self.optimised_parameters is None:
            return None
        predicted_focus = self.hyperbola(focus_pos, *self.optimised_parameters)
        return predicted_focus

    @staticmethod
    def fit_hyperbola(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Least-squares fit of `hyperbola` to `(x, y)` via `scipy.optimize.curve_fit`."""
        popt, pcov = scipy.optimize.curve_fit(
            GaussianStarFocusMeasure.hyperbola,
            x,
            y,
            p0=(1, 1, np.mean(x), np.min(y)),
        )
        return popt, pcov

    @staticmethod
    def hyperbola(
        x: npt.ArrayLike, a: float = 1, b: float = 1, x_0: float = 0, y_0: float = 0
    ) -> np.ndarray:
        """
        North-opening hyperbola used to model star size as a function of focus position.

            `y = b * sqrt(1 + ((x - x_0) / a)**2) + y_0`.

        Notes
        -----
        This is the north-opening branch of the hyperbola
        `(y - y_0)**2 / b**2 - (x - x_0)**2 / a**2 = 1`, solved for `y >= y_0`.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = np.linspace(-1, 2)
        >>> _ = plt.plot(x, GaussianStarFocusMeasure.hyperbola(x=x, a=1, b=1, x_0=0, y_0=-1))
        >>> plt.show()  # doctest: +SKIP
        """
        y = b * np.sqrt(1 + ((x - x_0) / a) ** 2) + y_0

        return y


class HFRStarFocusMeasure(StarSizeFocusMeasure):
    """
    `StarSizeFocusMeasure` that fits a Half Flux Radius (HFR) model to each star.

    Star size is twice the fitted `R_0` (see `HalfFluxRadius2D` and `StarFitter.star_size`).
    The focus response curve (star size vs. focus position) is modelled as a piecewise-linear
    "V" curve via `linear_V_curve`/`fit_linear_V_curve`; its centre gives the estimated
    in-focus position.

    Parameters
    ----------
    ref_image, fwhm, star_find_threshold, absolute_detection_limit, cutout_size,
    saturation_threshold, max_stars, **kwargs
        See `StarSizeFocusMeasure`. `model` is fixed to
        `astrafocus.models.HalfFluxRadius2D`.

    Examples
    --------
    >>> from astrafocus.interface.simulation import CabaretDeviceSimulator
    >>> image = CabaretDeviceSimulator.generate_image()
    >>> hfrfm = HFRStarFocusMeasure(image, fwhm=2.0, star_find_threshold=8.0)
    >>> bool(hfrfm.measure_focus(image) > 0)
    True

    Fitting a response curve to a sweep of images taken at different focus positions:

    >>> import numpy as np
    >>> focus_pos = np.array([1, 2, 3, 4, 5])
    >>> fm_vals = np.array([5.0, 3.0, 2.0, 3.0, 5.0])
    >>> best_focus = hfrfm.fit_focus_response_curve(focus_pos, fm_vals)
    >>> predicted = hfrfm.get_focus_response_curve_fit(focus_pos)
    """

    def __init__(
        self,
        ref_image: ImageType | None = None,
        fwhm: float = 2.0,
        star_find_threshold: float = 5.0,
        absolute_detection_limit: float = 0.0,
        cutout_size: int = 15,
        saturation_threshold: float | None = None,
        max_stars: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(
            model=HalfFluxRadius2D,
            ref_image=ref_image,
            fwhm=fwhm,
            star_find_threshold=star_find_threshold,
            absolute_detection_limit=absolute_detection_limit,
            cutout_size=cutout_size,
            saturation_threshold=saturation_threshold,
            max_stars=max_stars,
        )

    def fit_focus_response_curve(self, focus_pos: npt.ArrayLike, measured_focus: npt.ArrayLike) -> float:
        """
        Fit a piecewise-linear "V" curve to a focus sweep and return the estimated best focus.
        """
        popt, pcov = HFRStarFocusMeasure.fit_linear_V_curve(focus_pos, measured_focus)
        self.optimised_parameters = popt

        return popt[2]

    def get_focus_response_curve_fit(self, focus_pos: npt.ArrayLike) -> np.ndarray | None:
        """
        Evaluate the fitted "V" curve from `fit_focus_response_curve` at `focus_pos`.
        """
        if self.optimised_parameters is None:
            return None
        predicted_focus = self.linear_V_curve(focus_pos, *self.optimised_parameters)
        return predicted_focus

    @staticmethod
    def fit_linear_V_curve(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Least-squares fit of `linear_V_curve` to `(x, y)` via `scipy.optimize.curve_fit`."""
        popt, pcov = scipy.optimize.curve_fit(
            HFRStarFocusMeasure.linear_V_curve,
            x,
            y,
            p0=(-0.1, 0.1, np.mean(x), np.min(y)),
        )
        return popt, pcov

    @staticmethod
    def linear_V_curve(
        x: npt.ArrayLike,
        slope_left: float = -1,
        slope_right: float = 1,
        x_centre: float = 0,
        intercept: float = 0,
    ) -> np.ndarray:
        """
        Piecewise-linear "V" curve used to model star size as a function of focus position.

        Two linear branches meeting at `(x_centre, intercept)`: `slope_left` for
        `x <= x_centre` and `slope_right` for `x > x_centre`.

        Parameters
        ----------
        x : array_like
            Focus position(s) at which to evaluate the curve.
        slope_left : float, optional
            Slope for `x <= x_centre`. Default is -1.
        slope_right : float, optional
            Slope for `x > x_centre`. Default is 1.
        x_centre : float, optional
            Focus position at the centre of the "V" (the estimated in-focus position).
            Default is 0.
        intercept : float, optional
            Star size at `x_centre` (the minimum of the curve, assuming
            `slope_left <= 0 <= slope_right`). Default is 0.

        Returns
        -------
        array_like
            The piecewise-linear curve evaluated at `x`.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = np.linspace(-2, 2, 200)
        >>> _ = plt.plot(x, HFRStarFocusMeasure.linear_V_curve(x=x))
        >>> plt.show()  # doctest: +SKIP
        """
        y = np.where(
            x - x_centre <= 0,
            slope_left * (x - x_centre) + intercept,
            slope_right * (x - x_centre) + intercept,
        )

        return y

    @staticmethod
    def linear_V_curve_prime(
        x: npt.ArrayLike,
        slope_left: float = -1,
        slope_right: float = 1,
        intercept_left: float = 0,
        intercept_right: float = 0,
    ) -> np.ndarray:
        """
        Piecewise-linear "V" curve parametrised by separate per-branch intercepts.

        Unlike `linear_V_curve` (which is centred on a shared `x_centre`/`intercept`), this
        variant takes each branch's own y-intercept and derives the crossover point
        `v_centre = (intercept_right - intercept_left) / (slope_left - slope_right)`
        where the two lines meet.

        Parameters
        ----------
        x : array_like
            Focus position(s) at which to evaluate the curve.
        slope_left : float, optional
            Slope for `x <= v_centre`. Default is -1.
        slope_right : float, optional
            Slope for `x > v_centre`. Default is 1.
        intercept_left : float, optional
            Y-intercept of the left branch (value at `x = 0` if extended). Default is 0.
        intercept_right : float, optional
            Y-intercept of the right branch (value at `x = 0` if extended). Default is 0.

        Returns
        -------
        numpy.ndarray
            The piecewise-linear curve evaluated at `x`.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = np.linspace(-2, 2, 200)
        >>> _ = plt.plot(x, HFRStarFocusMeasure.linear_V_curve_prime(x=x))
        >>> plt.show()  # doctest: +SKIP
        """
        v_centre = (intercept_right - intercept_left) / (slope_left - slope_right)

        left_mask = x <= v_centre
        y = np.zeros_like(x)
        y[left_mask] = slope_left * x[left_mask] + intercept_left
        y[~left_mask] = slope_right * x[~left_mask] + intercept_right

        return y
