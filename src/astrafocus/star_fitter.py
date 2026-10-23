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
# The cutout grows while more than this fraction of the star flux is in its outer 2 pixels.
EDGE_FLUX_FRACTION = 0.01
# Maximum number of times the cutout is centred again on the star.
N_RECENTRE = 5


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
        Cut out the star near (`x`, `y`) and set all pixels that are not its light to zero.

        The star light is the group of touching pixels with the most flux (see `N_SIGMA`). This
        removes other stars and what is left of hot pixels. The cutout is centred on the group, so
        the star can move between frames. The half-width starts at `cutout_size` and doubles, up to
        `max_cutout_size`, while the group reaches the edge of the cutout or no star light is found.
        So a large donut far from focus fits, also when its dark centre fills the first cutout.

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
        prepared = prepare_image(image)
        height, width = prepared.data.shape
        size = int(cutout_size)
        max_size = size if max_cutout_size is None else max(size, int(max_cutout_size))
        n_moves = 0
        while True:
            # 1. Cut out the window around (x, y).
            x_min, y_min = max(int(x - size), 0), max(int(y - size), 0)
            x_max, y_max = min(int(x + size), width), min(int(y + size), height)
            data = prepared.data[y_min:y_max, x_min:x_max]

            # 2. Keep the group of touching pixels above the noise with the most flux.
            above_noise = prepared.smoothed[y_min:y_max, x_min:x_max] > N_SIGMA * prepared.smoothed_noise
            labels, n_groups = ndimage.label(above_noise, structure=np.ones((3, 3)))
            mask = above_noise
            if n_groups > 1:
                group_flux = ndimage.sum_labels(data, labels, index=np.arange(1, n_groups + 1))
                mask = labels == np.argmax(group_flux) + 1

            # 3. If there is no star light, look in a larger window, or give up.
            n_pixels = np.count_nonzero(mask)
            if n_pixels < MIN_PIXELS or np.sum(data[mask]) < MIN_SNR * prepared.noise * np.sqrt(n_pixels):
                if size < max_size:
                    size = min(2 * size, max_size)
                    continue
                raise ValueError("No star light above the noise in the cutout.")
            star_data = np.where(mask, data, 0.0)
            flux = np.sum(star_data)

            # 4. Centre the window on the star light. Then grow it while the light reaches its edge.
            #    Edges of the image do not count, because a larger window does not help there.
            yy, xx = np.indices(star_data.shape)
            x_new, y_new = x_min + np.sum(xx * star_data) / flux, y_min + np.sum(yy * star_data) / flux
            edge = np.zeros_like(mask)
            edge[:2], edge[-2:] = y_min > 0, y_max < height
            edge[:, :2] |= x_min > 0
            edge[:, -2:] |= x_max < width
            touches_edge = np.sum(star_data[edge]) > EDGE_FLUX_FRACTION * flux
            if np.hypot(x_new - x, y_new - y) > 0.5 and n_moves < N_RECENTRE:
                x, y, n_moves = x_new, y_new, n_moves + 1
            elif touches_edge and size < max_size:
                size = min(2 * size, max_size)
            else:
                return star_data

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
