import warnings

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
from astrafocus.utils.focus_curve import focus_curve_snr
from astrafocus.utils.logger import get_logger

logger = get_logger()

# The star cutout can grow to this many times `cutout_size`, so large donuts far from focus fit.
MAX_CUTOUT_FACTOR = 8
# Stars whose size changes by less than this fraction over a sweep, or whose own size curve has a
# signal-to-noise ratio below MIN_STAR_CURVE_SNR, are not used (see `combine_star_sizes`).
MIN_RELATIVE_SIZE_CHANGE = 0.2
MIN_STAR_CURVE_SNR = 8.0


def _lowest_point(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Return `x` and `y` as arrays, the lowest point `(x_low, y_low)`, and a typical slope of the curve.

    The curve fits start at the lowest point. A fit that starts in the middle of the search range
    does not find a best focus near its edge.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    i = int(np.argmin(y))
    slope = max((np.max(y) - y[i]) / max(np.ptp(x), 1.0), 1e-6)
    return x, y, x[i], y[i], slope


def _robust_curve_fit(model, x, y, p0, y_low):
    """
    `scipy.optimize.curve_fit` with a soft-L1 loss, so that a few bad points, e.g. of overlapping
    donuts, do not move the result. Residuals larger than 5% of the lowest value `y_low` count less.
    """
    f_scale = max(0.05 * abs(y_low), 1e-3)
    return scipy.optimize.curve_fit(model, x, y, p0=p0, method="trf", loss="soft_l1", f_scale=f_scale)


class StarSizeFocusMeasure(AnalyticResponseFocusedMeasureOperator):
    """
    Focus measure operator that derives focus quality from the fitted size of detected stars.

    Stars are located in the image with a `StarFinder`, and `model` is fit to each detected
    star with a `StarFitter` (see `StarFitter.star_size`, e.g. FWHM). Before the fit, the sky is
    subtracted and only the star light is kept (see `StarFitter.extract_star`). The median
    fitted size across stars is returned as the focus measure; smaller values indicate better
    focus.

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
        Start half-width (pixels) of the square cutout extracted around each star for fitting.
        It grows up to `MAX_CUTOUT_FACTOR` times this (see `StarFitter.extract_star`). Default is 15.
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
    last_star_sizes : numpy.ndarray or None
        Size of each selected star in the last image passed to `measure_focus` (NaN if the star
        could not be measured). None before the first call.
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
        self.star_fitter = StarFitter(model)
        self.cutout_size = cutout_size
        self.star_finder = self._make_star_finder(ref_image) if ref_image is not None else None
        self.last_star_sizes = None
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
            self.star_finder = self._make_star_finder(image)
        return self.star_finder

    def _make_star_finder(self, image: ImageType) -> StarFinder:
        """
        Find the stars in `image`, and move each one to the centre of its light.

        The star finder only tries a wider kernel if the narrow one finds nothing. Far from focus,
        the narrow kernel misses the donuts but can still find a few noise peaks. So the sources of
        every kernel width in `StarFinder.FWHM_FACTORS` are used here, and a source with no star
        light above the noise (see `StarFitter.star_centre`) is dropped as noise. If no source has
        star light, the star finder's own sources are kept, and the measurements fail later.

        Far from focus, the star finder places sources on the ring of a donut, often several on the
        same donut. The cutouts must be centred on the donut's centre, where the star is at focus
        (see `StarFitter.extract_star`). So each source moves to the centre of its star, and sources
        that end up within `cutout_size` of a brighter star are dropped. Stars are ranked by the
        flux of their light, because the source fluxes of different kernel widths can't be compared.
        """
        star_finder = StarFinder(image, **self._star_finder_kwargs)
        prepared = star_finder.prepared_ref_image
        max_cutout_size = MAX_CUTOUT_FACTOR * self.cutout_size
        tables, centres, fluxes = [], [], []
        for fwhm_factor in StarFinder.FWHM_FACTORS:
            if fwhm_factor < star_finder.fwhm_factor:  # the star finder found nothing with these
                continue
            if fwhm_factor == star_finder.fwhm_factor:
                sources = star_finder.selected_stars
            else:
                sources = star_finder._find_sources(prepared, fwhm_factor * star_finder.fwhm)
            if sources is None:
                continue
            tables.append(sources)
            for x, y in zip(sources["x_centroid"], sources["y_centroid"]):
                try:
                    light, x_centre, y_centre = StarFitter._find_star(
                        prepared, x, y, self.cutout_size, max_cutout_size, on_star=True
                    )
                except ValueError:  # no star light at this source
                    centres.append((np.nan, np.nan))
                    fluxes.append(-np.inf)
                else:
                    centres.append((x_centre, y_centre))
                    fluxes.append(np.sum(light))
        if not np.isfinite(np.max(fluxes)):
            return star_finder

        stars = astropy.table.vstack(tables, metadata_conflicts="silent")
        centres = np.array(centres)
        keep = []
        for i in np.argsort(fluxes)[::-1]:  # brightest first
            if not np.isfinite(fluxes[i]) or len(keep) == self._star_finder_kwargs["max_stars"]:
                break
            if all(np.hypot(*(centres[i] - centres[j])) > self.cutout_size for j in keep):
                keep.append(i)
        stars["x_centroid"], stars["y_centroid"] = centres[:, 0], centres[:, 1]
        star_finder.selected_stars = stars[keep]
        return star_finder

    def measure_focus(self, image: ImageType, cutout_size: int | None = None, **kwargs) -> float:
        """
        Compute the median fitted star size in `image`, used as the focus measure.

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
            Median fitted star size (e.g. FWHM) across the selected stars. Smaller is better.

        Raises
        ------
        ValueError
            If the `StarFinder` could not find enough stars in `image`, or if no star could be
            measured in `image`.
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
            max_cutout_size=MAX_CUTOUT_FACTOR * cutout_size,
        )
        self.last_star_sizes = star_size_arr[0]
        if np.all(np.isnan(star_size_arr)):
            raise ValueError("No star light above the noise at any of the selected star positions.")
        return float(np.nanmedian(star_size_arr))

    def combine_star_sizes(
        self, star_sizes: npt.ArrayLike, focus_pos: npt.ArrayLike, min_points: int = 5
    ) -> np.ndarray:
        """
        Combine the star sizes of a sweep into one focus measure per frame.

        `star_sizes` has shape (n_frames, n_stars), and `focus_pos` gives the focus position of each
        frame. A real star changes size smoothly when the focus changes, so a star is only used if:

        - it was measured in at least `min_points` frames,
        - its size changes by at least `MIN_RELATIVE_SIZE_CHANGE` (10th to 90th percentile,
          relative to the median), and
        - its own size curve has a signal-to-noise ratio of at least `MIN_STAR_CURVE_SNR`
          (see `focus_curve_snr`).

        This drops hot pixels and noise. Before that, a size measured in a frame whose neighbours (in
        focus position) both have no size for that star is dropped: one such point says nothing
        about the shape of the curve, but as the lowest point it would decide the best focus, e.g.
        a frame taken before the focuser reached its position. The focus measure of a frame is the
        median size of the stars that are used, or NaN if no star is left. With fewer than
        `min_points` frames, all stars are used.

        Examples
        --------
        >>> hfrfm = HFRStarFocusMeasure()
        >>> focus_pos = np.array([100, 200, 300, 400, 500])
        >>> star = np.array([5.0, 3.0, 2.0, 3.0, 5.0])
        >>> hot_pixel = np.full(5, 0.6)
        >>> hfrfm.combine_star_sizes(np.column_stack([star, star, hot_pixel]), focus_pos).tolist()
        [5.0, 3.0, 2.0, 3.0, 5.0]
        """
        star_sizes = np.atleast_2d(np.array(star_sizes, dtype=float))  # a copy, changed below
        keep = np.ones(star_sizes.shape[1], dtype=bool)
        if len(star_sizes) >= min_points:
            order = np.argsort(np.asarray(focus_pos, dtype=float), kind="stable")
            measured = np.pad(np.isfinite(star_sizes[order]), ((1, 1), (0, 0)))
            rows, columns = np.nonzero(measured[1:-1] & ~measured[:-2] & ~measured[2:])
            star_sizes[order[rows], columns] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # stars that are never measured
                p10, median, p90 = np.nanpercentile(star_sizes, [10, 50, 90], axis=0)
            keep &= np.count_nonzero(np.isfinite(star_sizes), axis=0) >= min_points
            keep &= p90 - p10 >= MIN_RELATIVE_SIZE_CHANGE * median
            keep &= [focus_curve_snr(focus_pos, sizes) >= MIN_STAR_CURVE_SNR for sizes in star_sizes.T]
            if not np.all(keep):
                logger.info(
                    f"Not using {np.count_nonzero(~keep)} of {keep.size} stars: size does not follow focus."
                )
        if not np.any(keep):
            return np.full(len(star_sizes), np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # frames where no used star is measured
            return np.nanmedian(star_sizes[:, keep], axis=1)

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
            **kwargs,
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
        """Robust fit of `hyperbola` to `(x, y)`, starting at the lowest point (see `_lowest_point`)."""
        x, y, x_low, y_low, slope = _lowest_point(x, y)
        b = max(y_low, 1e-3)  # with y_0 = 0, the vertex is at (x_low, y_low) and the asymptotes have `slope`
        return _robust_curve_fit(GaussianStarFocusMeasure.hyperbola, x, y, (b / slope, b, x_low, 0.0), y_low)

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

    Star size is the half flux radius (HFR) in pixels, i.e. the fitted `R_0` (see
    `HalfFluxRadius2D` and `StarFitter.star_size`). The half flux diameter (HFD) is twice this.
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
            **kwargs,
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
        """Robust fit of `linear_V_curve` to `(x, y)`, starting at the lowest point (see `_lowest_point`)."""
        x, y, x_low, y_low, slope = _lowest_point(x, y)
        return _robust_curve_fit(
            HFRStarFocusMeasure.linear_V_curve, x, y, (-slope, slope, x_low, y_low), y_low
        )

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
