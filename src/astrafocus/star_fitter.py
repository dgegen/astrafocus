import astropy
import cv2
import numpy as np
from astropy.modeling import fitting
from scipy import ndimage

from astrafocus.star_finder import prepare_image

# Star light is a group of touching pixels where the smoothed image is above N_SIGMA times its
# noise, with at least MIN_PIXELS pixels and a signal-to-noise ratio of at least MIN_SNR.
N_SIGMA = 3.0
MIN_PIXELS = 3
MIN_SNR = 10.0
# The cutout grows while more than this fraction of the star flux is outside it.
OUTSIDE_FLUX_FRACTION = 0.01
# A position counts as on the star light if the light comes this close to it (pixels).
ON_STAR_DISTANCE = 3.0


class StarFitter:
    """
    A class for fitting astropy-like models to star data to calculate the size of stars.

    Parameters
    ----------
    model : astropy.modeling.Model
        The astropy model to fit to the star data.

    Attributes
    ----------
    model : astropy.modeling.Model
        The astropy model used for fitting.
    result : astropy.modeling.Model
        The result of the fitting process.
    star_size : float
        The full-width at half-maximum (FWHM) of the fitted model, used as the focus measure.

    Methods
    -------
    fit(star_data, *args, **kwargs)
        Fit the specified model to the given star_data.
    fit_source(image, star, cutout_size=15, *args, **kwargs)
        Slice out a star from an image and fit the model to it.
    integrate_source(image, star, cutout_size=15, fitter=..., *args, **kwargs)
        Fit the model to a star and integrate the fitted profile to estimate its flux.
    calculate_star_sizes_of_selection(images, selected_stars, cutout_size, *args, **kwargs)
        Fit the model to each star in a selection, across one or more images.

    Raises
    ------
    ValueError
        If `result` or `parameter_dict` is accessed before the object has been fitted.
    AttributeError
        If `star_size` is accessed but FWHM calculation is not supported for the model type.

    Examples
    --------
    >>> from astropy.modeling import models
    >>> from astrafocus.interface.simulation import CabaretDeviceSimulator
    >>> from astrafocus.star_size_focus_measure_operators import GaussianStarFocusMeasure
    >>> image = CabaretDeviceSimulator.generate_image()
    >>> gsfm = GaussianStarFocusMeasure(image)
    >>> star = gsfm.star_finder.selected_stars[0]
    >>> star_fitter = StarFitter(models.Gaussian2D)
    >>> _ = star_fitter.fit_source(image, star=star, cutout_size=15)
    >>> bool(star_fitter.star_size > 0)
    True
    """

    def __init__(
        self,
        model: astropy.modeling.core._ModelMeta,
        scale_factor: int = 1,
        fitter: astropy.modeling.fitting = fitting.LevMarLSQFitter(),
    ):
        """
        Initialize the StarFitter with the specified model.

        Parameters
        ----------
        _result : Optional[astropy.modeling.functional_models]
            Fitted model
        model : astropy.modeling.core._ModelMeta
            An astropy model like model to use for fitting.
            Commonly, models.Gaussian2D, models.EllipticMoffat2D, or models.Ring2D.
            This package further contains autofocus.models.EllipticMoffat2D and
            autofocus.models.HalfFluxRadius2D.
        """
        self.scale_factor = scale_factor
        self.model = model
        self._result: astropy.modeling.functional_models | None = None
        self._fitter = fitter if not hasattr(model, "fit") else self.model.fit

    @property
    def result(self):
        if self._result is not None:
            return self._result
        else:
            raise ValueError("No results available. The object has not been fitted yet.")

    @property
    def parameter_dict(self):
        if self._result is not None:
            return dict(zip(self._result.param_names, self._result.parameters))
        else:
            raise ValueError("No results available. The object has not been fitted yet.")

    def calculate_star_sizes_of_selection(
        self, images, selected_stars, cutout_size, max_cutout_size=None, *args, **kwargs
    ):
        """
        Fit the model to each star in `selected_stars`, in each image.

        Each image is prepared once (see `prepare_image`). Returns an array of shape
        (n_images, n_stars). A star that has no light above the noise in an image gets NaN.
        """
        if isinstance(images, np.ndarray) and images.ndim == 2:
            images = [images]
        if isinstance(selected_stars, astropy.table.row.Row):
            selected_stars = astropy.table.table.QTable(selected_stars)

        star_sizes = np.full((len(images), len(selected_stars)), np.nan)
        for i_image, image in enumerate(images):
            prepared = prepare_image(image)
            for i_star, star in enumerate(selected_stars):
                try:
                    star_sizes[i_image, i_star] = self.fit_source(
                        prepared, star, cutout_size, max_cutout_size, *args, **kwargs
                    ).star_size
                except ValueError:  # no star light above the noise
                    pass

        return star_sizes

    def fit_source(self, image, star, cutout_size=15, max_cutout_size=None, *args, **kwargs):
        """
        Fit the model to the star near `star["x_centroid"]`, `star["y_centroid"]`.

        `image` is a raw image or the result of `prepare_image` (see `extract_star`).
        """
        star_data = self.extract_star(
            image, star["x_centroid"], star["y_centroid"], cutout_size, max_cutout_size
        )
        if self.scale_factor != 1:
            star_data = cv2.resize(
                star_data,
                None,
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv2.INTER_LINEAR,
            )

        self.fit(star_data, *args, **kwargs)

        if self.scale_factor != 1:
            self.rescale_result()

        return self

    def fit(self, star_data, *args, **kwargs):
        """
        Fit the specified model to the window around the star provided in star_data.

        Parameters
        ----------
        star_data : numpy.ndarray
                A small section of the full image containing the intensity profile of the star.
        *args, **kwargs : additional arguments and keyword arguments
            Additional arguments to pass to the model constructor.
        """
        y, x = np.indices(star_data.shape)

        bounds = self.get_bounds(self.model().param_names, star_data)
        init = self.get_parameter_init(self.model().param_names, star_data)
        model = self.model(*args, **init, **({"bounds": bounds} | kwargs))

        self._result = self.fitter(model, x, y, star_data, estimate_jacobian=False)
        return self

    def fitter(self, model, x, y, star_data, *args, **kwargs):
        return self._fitter(model, x, y, star_data, *args, **kwargs)

    def rescale_result(self):
        for param_name in self._result.param_names:
            if param_name in [
                "x_mean",
                "y_mean",
                "x_0",
                "y_0",
                "x_stddev",
                "y_stddev",
                "sigma_x",
                "sigma_y",
                "R_0",
            ]:
                setattr(
                    self._result,
                    param_name,
                    getattr(self._result, param_name) / self.scale_factor,
                )

    @staticmethod
    def get_masked_star(image, star, cutout_size, max_cutout_size=None):
        """Return the sky-subtracted, noise-masked cutout around `star` (see `extract_star`)."""
        return StarFitter.extract_star(
            image, star["x_centroid"], star["y_centroid"], cutout_size, max_cutout_size
        )

    @staticmethod
    def extract_star(image, x, y, cutout_size=15, max_cutout_size=None):
        """
        Cut out the star centred near (`x`, `y`) and set all pixels that are not its light to zero.

        The star light is the group of touching pixels with the most flux (see `N_SIGMA`) whose
        centroid is within `cutout_size` of (`x`, `y`), among the groups the largest cutout can
        reach. This removes other stars, even brighter ones, and what is left of hot pixels, also
        inside the dark centre of a donut. The cutout is centred on the group, so the star can move
        between frames, but only by up to `cutout_size`. The half-width starts at `cutout_size` and
        doubles, up to `max_cutout_size`, while more than `OUTSIDE_FLUX_FRACTION` of the light is
        outside the cutout. So a large donut far from focus fits.

        (`x`, `y`) must be the centre of the star, not a point on a donut's ring; see `star_centre`.

        `image` is a raw image or the result of `prepare_image`. The kept pixels have their
        sky-subtracted values. Raises ValueError if there is no star light in the cutout.

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> image = rng.normal(3000.0, 50.0, size=(200, 200))
        >>> yy, xx = np.indices(image.shape)
        >>> image += 1e5 / (2 * np.pi * 2.0**2) * np.exp(-((xx - 100) ** 2 + (yy - 90) ** 2) / (2 * 2.0**2))
        >>> star_data = StarFitter.extract_star(image, x=95, y=95)
        >>> bool(abs(np.sum(star_data) - 1e5) < 0.05 * 1e5)
        True
        """
        return StarFitter._find_star(image, x, y, cutout_size, max_cutout_size)[0]

    @staticmethod
    def star_centre(image, x, y, cutout_size=15, max_cutout_size=None):
        """
        Return the centre (x, y) of the star whose light is at (`x`, `y`).

        Far from focus, a star finder can place a source on a donut's ring. This returns the centre
        of the whole donut, which is where the star is at focus (see `extract_star`). The light
        must come within `ON_STAR_DISTANCE` of (`x`, `y`). Raises ValueError if it does not.

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> image = rng.normal(3000.0, 50.0, size=(200, 200))
        >>> yy, xx = np.indices(image.shape)
        >>> r = np.hypot(xx - 100, yy - 90)
        >>> image += np.where((r > 10) & (r < 25), 300.0, 0.0)
        >>> x, y = StarFitter.star_centre(image, x=118, y=90, max_cutout_size=60)
        >>> bool(abs(x - 100) < 1 and abs(y - 90) < 1)
        True
        """
        return StarFitter._find_star(image, x, y, cutout_size, max_cutout_size, on_star=True)[1:]

    @staticmethod
    def _find_star(image, x, y, cutout_size=15, max_cutout_size=None, on_star=False):
        """
        Return the cutout of the star light near (`x`, `y`) and the centre of that light.

        With `on_star`, the light must come within `ON_STAR_DISTANCE` of (`x`, `y`), else its
        centroid must be within `cutout_size` of it (see `extract_star` and `star_centre`).
        """
        prepared = prepare_image(image)
        height, width = prepared.data.shape
        size = int(cutout_size)
        max_size = size if max_cutout_size is None else max(size, int(max_cutout_size))

        # 1. Choose the star light among the groups of touching pixels above the noise, in all of
        #    the area the cutout can reach. So the whole of a large donut competes with, for
        #    example, what is left of a hot pixel in its dark centre. A point on the ring can be up
        #    to a donut radius from its centre, so the area is larger with `on_star`.
        reach = 2 * max_size if on_star else max_size + size
        x0, y0 = max(int(x - reach), 0), max(int(y - reach), 0)
        x1, y1 = min(int(x + reach), width), min(int(y + reach), height)
        data = prepared.data[y0:y1, x0:x1]
        above_noise = prepared.smoothed[y0:y1, x0:x1] > N_SIGMA * prepared.smoothed_noise
        labels, n_groups = ndimage.label(above_noise, structure=np.ones((3, 3)))
        index = np.arange(1, n_groups + 1)
        group_flux = ndimage.sum_labels(data, labels, index=index)
        if on_star:
            yy, xx = np.indices(data.shape)
            pixel_distance = np.hypot(x0 + xx - x, y0 + yy - y)
            distance = np.asarray(ndimage.minimum(pixel_distance, labels, index)).reshape(-1)
            group_flux[distance > ON_STAR_DISTANCE] = -np.inf
        else:
            centres = np.array(ndimage.center_of_mass(np.abs(data), labels, index)).reshape(-1, 2)
            distance = np.hypot(x0 + centres[:, 1] - x, y0 + centres[:, 0] - y)
            group_flux[distance > cutout_size] = -np.inf

        # 2. Of the groups near the star, the one with the most flux is the star light.
        if n_groups == 0 or not np.isfinite(np.max(group_flux)):
            raise ValueError("No star light above the noise in the cutout.")
        mask = labels == np.argmax(group_flux) + 1
        n_pixels = np.count_nonzero(mask)
        flux = np.sum(data[mask])
        if n_pixels < MIN_PIXELS or flux < MIN_SNR * prepared.noise * np.sqrt(n_pixels):
            raise ValueError("No star light above the noise in the cutout.")
        light = np.where(mask, data, 0.0)
        yy, xx = np.indices(light.shape)
        x_centre, y_centre = np.sum(xx * light) / flux, np.sum(yy * light) / flux

        # 3. Centre the cutout on the star light, and grow it while part of the light is outside.
        while True:
            x_min, y_min = max(int(x_centre - size), 0), max(int(y_centre - size), 0)
            x_max, y_max = min(int(x_centre + size), x1 - x0), min(int(y_centre + size), y1 - y0)
            star_data = light[y_min:y_max, x_min:x_max]
            if np.sum(star_data) < (1 - OUTSIDE_FLUX_FRACTION) * flux and size < max_size:
                size = min(2 * size, max_size)
            else:
                return star_data, float(x0 + x_centre), float(y0 + y_centre)

    @staticmethod
    def _slice_out_star(image, x, y, size=15):
        return image[
            np.maximum(int(y - size), 0) : int(y + size),
            np.maximum(int(x - size), 0) : int(x + size),
        ]

    @property
    def star_size(self):
        """
        Return the size of the fitted star in pixels of the original image.

        - Models with `fwhm`: the FWHM.
        - 2D Gaussian: the mean of the FWHM along x and y.
        - `HalfFluxRadius2D`: the half-flux radius (HFR). The half-flux diameter (HFD) is twice this.
        """
        result = self.result
        if hasattr(result, "fwhm"):
            return result.fwhm
        elif hasattr(result, "x_fwhm") and hasattr(result, "y_fwhm"):
            return (result.x_fwhm + result.y_fwhm) / 2
        elif hasattr(result, "R_0"):  # for HalfFLuxRadius
            return float(result.R_0.value)
        else:
            raise AttributeError("Focus measure calculation is not supported for this model type.")

    def integrate_source(
        self,
        image,
        star,
        cutout_size=15,
        fitter=fitting.LevMarLSQFitter(),
        *args,
        **kwargs,
    ):
        star_data = self.get_masked_star(image, star, cutout_size)
        self.fit(star_data, fitter=fitter, *args, **kwargs)

        y, x = np.indices(star_data.shape)
        flux_estimate = self._result(x, y).sum()

        return flux_estimate

    @staticmethod
    def get_bounds(parameters, star_data) -> dict:
        bounds = dict()
        for param_name in parameters:
            if param_name in ["x_mean", "x_0"]:
                bounds[param_name] = (0, star_data.shape[1])
            elif param_name in ["y_mean", "y_0"]:
                bounds[param_name] = (0, star_data.shape[0])
            elif param_name in ["x_stddev", "sigma_x"]:
                bounds[param_name] = (0.5, star_data.shape[1])
            elif param_name in ["y_stddev", "sigma_y"]:
                bounds[param_name] = (0.5, star_data.shape[0])
            elif param_name in ["R_0"]:
                bounds[param_name] = (0.5, np.max(star_data.shape))
            # elif param_name in ["amplitude"]:
            #     bounds[param_name] = (np.maximum(0, np.min(star_data)), np.inf)

        return bounds

    @staticmethod
    def get_parameter_init(parameters, star_data, expected_star_size=5) -> dict:
        """
        Start values for the fit. If `star_data` has positive flux, the position, width and
        amplitude come from the flux-weighted moments of `star_data`. With a fixed start width,
        the fit of a large donut can converge to a single bright pixel.
        """
        x_0, y_0 = star_data.shape[1] // 2, star_data.shape[0] // 2
        sigma_x = sigma_y = expected_star_size
        amplitude = 0.9 * np.max(star_data)

        star_data = np.clip(star_data, 0, None)
        flux = np.sum(star_data)
        if flux > 0:
            y, x = np.indices(star_data.shape)
            x_0, y_0 = np.sum(x * star_data) / flux, np.sum(y * star_data) / flux
            sigma_x = max(np.sqrt(np.sum((x - x_0) ** 2 * star_data) / flux), 0.5)
            sigma_y = max(np.sqrt(np.sum((y - y_0) ** 2 * star_data) / flux), 0.5)
            amplitude = flux / (2 * np.pi * sigma_x * sigma_y)

        init = dict()
        for param_name in parameters:
            if param_name in ["x_mean", "x_0"]:
                init[param_name] = x_0
            elif param_name in ["y_mean", "y_0"]:
                init[param_name] = y_0
            elif param_name in ["x_stddev", "sigma_x"]:
                init[param_name] = sigma_x
            elif param_name in ["y_stddev", "sigma_y"]:
                init[param_name] = sigma_y
            elif param_name in ["R_0"]:
                init[param_name] = expected_star_size
            elif param_name in ["amplitude"]:
                init[param_name] = amplitude

        return init

    def __repr__(self):
        model = self.model() if self._result is None else self._result
        return (
            f"StarFitter(model={model!r}, "
            f"scale_factor={self.scale_factor}, "
            f"fitter={self._fitter.__class__.__name__})"
        )
