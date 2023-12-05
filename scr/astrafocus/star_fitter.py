from typing import Optional

import cv2
import astropy
import numpy as np
from astropy.modeling import fitting, models


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
    fwhm : float
        The full-width at half-maximum (FWHM) of the fitted model.

    Methods
    -------
    fit(star_data, *args, **kwargs)
        Fit the specified model to the given star_data.
    calculate_avg_fwhm(result)
        Calculate the average FWHM from the fitted model parameters.

    Raises
    ------
    ValueError
        If the object has not been fitted yet or if FWHM calculation is not supported for the model type.

    Examples
    --------
    >>> image = image_data[0]
    >>> from astrafocus.star_size_focus_measure_operators import GaussianStarFocusMeasure
    >>> gsfm = GaussianStarFocusMeasure(image, fwhm=2.0, star_find_threshold=8.0)
    >>> gsfm.star_finder.selected_stars
    
    # initialise with a model from astropy.modeling.models,
    >>> star_fitter = StarFitter(models.Gaussian2D)
    >>> star_fitter.fit(star_data)
    >>> fwhm = star_fitter.star_size
    >>> star_fitter.fit_source(image, star=gsfm.star_finder.selected_stars, cutout_size=15)

    # Initialise model
    >>> from astrafocus.models.half_flux_radius_2D import HalfFluxRadius2D
    >>> star_fitter = StarFitter(HalfFluxRadius2D)
    >>> star_fitter.fit_source(image, star=gsfm.star_finder.selected_stars, cutout_size=15)
    
    
    >>> star_data = star_fitter.get_masked_star(image, gsfm.star_finder.selected_stars, cutout_size=15)
    >>> plt.imshow(star_data); plt.show()
    plt.imshow(star_fitter._result(*np.indices(star_data.shape))); plt.show()
    >>> 
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
            Commonly, models.Gaussian2D, models.Moffat2D, or models.Ring2D.
            This package further contains autofocus.models.EllipticMoffat2D and 
            autofocus.models.HalfFluxRadius2D.
        """
        self.scale_factor = scale_factor
        self.model = model
        self._result: Optional[astropy.modeling.functional_models] = None
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
        self, images, selected_stars, cutout_size, *args, **kwargs
    ):
        """ """
        if isinstance(images, np.ndarray) and images.ndim == 2:
            images = [images]
        if isinstance(selected_stars, astropy.table.row.Row):
            selected_stars = astropy.table.table.QTable(selected_stars)

        star_sizes = np.zeros((len(images), len(selected_stars)))
        for i_star, star in enumerate(selected_stars):
            star_sizes[:, i_star] = [
                self.fit_source(image, star, cutout_size=cutout_size, *args, **kwargs).star_size
                for image in images
            ]

        return star_sizes

    def fit_source(self, image, star, cutout_size=15, *args, **kwargs):
        star_data = self.get_masked_star(image, star, cutout_size)
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
                    self._result, param_name, getattr(self._result, param_name) / self.scale_factor
                )

    @staticmethod
    def get_masked_star(image, star, cutout_size):
        y, x = star["ycentroid"], star["xcentroid"]
        star_data = StarFitter._slice_out_star(image, x, y, size=cutout_size)
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
        Calculate the full-width at half-maximum (FWHM) of the fitted model.
        In the case of a 2d Gaussian, return the average FWHM.
        """
        result = self.result
        if hasattr(result, "fwhm"):
            return result.fwhm
        elif hasattr(result, "x_fwhm") and hasattr(result, "y_fwhm"):
            return (result.x_fwhm + result.y_fwhm) / (2 * self.scale_factor)
        elif hasattr(result, "R_0"):  # for HalfFLuxRadius
            return 2 * result.R_0 / self.scale_factor
        else:
            raise AttributeError("Focus measure calculation is not supported for this model type.")

    def integrate_source(
        self, image, star, cutout_size=15, fitter=fitting.LevMarLSQFitter(), *args, **kwargs
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
        init = dict()
        for param_name in parameters:
            if param_name in ["x_mean", "x_0"]:
                init[param_name] =  star_data.shape[1]//2
            elif param_name in ["y_mean", "y_0"]:
                init[param_name] = star_data.shape[0]//2
            elif param_name in ["x_stddev", "sigma_x"]:
                init[param_name] = expected_star_size
            elif param_name in ["y_stddev", "sigma_y"]:
                init[param_name] = expected_star_size
            elif param_name in ["R_0"]:
                init[param_name] = expected_star_size
            elif param_name in ["amplitude"]:
                init[param_name] = 0.9*np.max(star_data)

        return init

    def __repr__(self):
        model = self.model() if self._result is None else self._result
        return (
            f"StarFitter(model={model!r}, "
            f"scale_factor={self.scale_factor}, "
            "fitter={self._fitter.__class__.__name__})"
        )
