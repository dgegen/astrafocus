import numpy as np

from astropy.units import UnitsError
from astropy.modeling.core import Fittable2DModel
from astropy.modeling.parameters import Parameter
from astropy.stats import gaussian_sigma_to_fwhm


def elliptical_moffat_model(x, y):
    def model(amplitude, x_0, y_0, sx, sy, theta, background, beta):
        # https://pixinsight.com/doc/tools/DynamicPSF/DynamicPSF.html
        dx_ = x - x_0
        dy_ = y - y_0
        dx = dx_ * np.cos(theta) + dy_ * np.sin(theta)
        dy = -dx_ * np.sin(theta) + dy_ * np.cos(theta)

        return background + amplitude / np.power(1 + (dx / sx) ** 2 + (dy / sy) ** 2, beta)

    return model


class EllipticalMoffat2D(Fittable2DModel):
    """
    Two dimensional Moffat model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the model.
    x_0 : float
        x position of the maximum of the Moffat model.
    y_0 : float
        y position of the maximum of the Moffat model.
    sigma_x : float
        Core width of the Moffat model.
    sigma_y : float
        Core height of the Moffat model.
    alpha : float
        Power index of the Moffat model.

    See Also
    --------
    Gaussian2D, Box2D

    Notes
    -----
    Model formula:

    .. math::

        f(x, y) = B + A \\left(1 + amplitude * \\frac{\\left(x - x_{0}\\right)^{2} +
        \\left(y - y_{0}\\right)^{2}}{\\gamma^{2}}\\right)^{- \\alpha}

    Examples
    --------
    from astrafocus.utils.load_fits_from_directory import load_fits_from_directory
    fits_directory = "path_to_fits_files"
    image_data, headers, focus_pos = load_fits_from_directory(fits_directory)

    star_data = StarFitter.get_masked_star(image_data[0], selected_stars[0], cutout_size=15)

    model = EllipticalMoffat2D(
        amplitude=np.max(star_data),
        background=np.median(star_data),
        x_0=star_data.shape[1]/2,
        y_0=star_data.shape[0]/2,
    )
    fitter = fitting.LevMarLSQFitter()

    y, x = np.indices(star_data.shape)
    fit = fitter(model, x, y, star_data, estimate_jacobian=False)

    fig, ax = plt.subplots(figsize=(6.69, 4.14))
    ax.set_xlabel('Column pixel')
    ax.set_ylabel('Row pixel')
    ax.imshow(star_data, origin="lower")
    ax.contour(fit(*np.indices(star_data.shape)), colors="red")
    plt.show()

    """

    amplitude = Parameter(default=1, description="Amplitude (peak value) of the model")
    background = Parameter(default=0, description="Average local background value of the model")
    x_0 = Parameter(default=0, description="X position of the maximum of the Moffat model")
    y_0 = Parameter(default=0, description="Y position of the maximum of the Moffat model")
    sigma_x = Parameter(default=1, description="Core x-width of the Moffat model")
    sigma_y = Parameter(default=1, description="Core y-width of the Moffat model")
    alpha = Parameter(default=1, description="Power index of the Moffat model")
    theta = Parameter(
        default=0.0,
        description=(
            "Rotation angle either as a " "float (in radians) or a " "|Quantity| angle (optional)"
        ),
    )

    @property
    def fwhm_x(self):
        """
        Moffat full width at half maximum.
        Derivation of the formula is available in
        `this notebook by Yoonsoo Bach
        <https://nbviewer.jupyter.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb#1.2.-Moffat>`_.
        """
        return 2.0 * np.abs(self.sigma_x) * np.sqrt(2.0 ** (1.0 / self.alpha) - 1.0)
        # return 2.0 * np.abs(self.sigma_y) * np.sqrt(2.0 ** (1.0 / self.alpha) - 1.0)
        # gaussian_sigma_to_fwhm * np.mean(
        #     [params["sigma_x"], params["sigma_y"]]
        # )

    @staticmethod
    def evaluate(x, y, amplitude, background, x_0, y_0, sigma_x, sigma_y, alpha, theta):
        """Two dimensional Moffat model function."""
        # rr_gg = ((dx_) ** 2 / sigma_x + (dy_) ** 2) / sigma_y**2
        # return amplitude * (1 + rr_gg) ** (-alpha) + background

        dx_ = x - x_0
        dy_ = y - y_0

        # Perform rotation
        dx = dx_ * np.cos(theta) + dy_ * np.sin(theta)
        dy = -dx_ * np.sin(theta) + dy_ * np.cos(theta)

        return background + amplitude * np.power(
            1 + (dx / sigma_x) ** 2 + (dy / sigma_y) ** 2, -alpha
        )

    @staticmethod
    def fit_deriv(x, y, amplitude, background, x_0, y_0, sigma_x, sigma_y, alpha, theta):
        """Two dimensional Moffat model derivative with respect to parameters."""
        dx_ = x - x_0
        dy_ = y - y_0

        # Perform rotation
        dx = dx_ * np.cos(theta) + dy_ * np.sin(theta)
        dy = -dx_ * np.sin(theta) + dy_ * np.cos(theta)

        rr_gg = (dx / sigma_x) ** 2 + (dy / sigma_y) ** 2
        d_A = (1 + rr_gg) ** (-alpha)
        inner_derivative = d_A / (1 + rr_gg)  # == (1 + rr_gg) ** (-alpha - 1)

        # d_gamma = 2 * amplitude * alpha * d_A * rr_gg / (gamma * (1 + rr_gg))
        # e.g. d/dv(A (1 + ((x-x_0)/v)^2 + ((y-y_0)/w)^2)^(-a))
        # d/dv (1 + ((x Cos[t] + y Sin[t])/v)^2 + ((-x Sin[t] + y Cos[t])/w)^2)^(-a)
        d_sigma_x = 2 * amplitude * alpha * dx**2 * inner_derivative / sigma_x**3
        d_sigma_y = 2 * amplitude * alpha * dy**2 * inner_derivative / sigma_y**3

        # Here gamma changed to sigma
        d_x_0 = 2 * amplitude * alpha * (x - x_0) * inner_derivative / sigma_x**2
        d_y_0 = 2 * amplitude * alpha * (y - y_0) * inner_derivative / sigma_y**2
        d_alpha = -amplitude * d_A * np.log(1 + rr_gg)
        d_theta = (
            2
            * amplitude
            * alpha
            * inner_derivative
            * (
                (
                    (dy_ * np.cos(theta) - dx_ * np.sin(theta))
                    * (dx_ * np.cos(theta) - dy_ * np.sin(theta))
                )
                * (1 / sigma_x**2 - 1 / sigma_y**2)(
                    (dy_ * np.cos(theta) - dx_ * np.sin(theta))
                    * (dx_ * np.cos(theta) - dy_ * np.sin(theta))
                )
                / sigma_x**2
                - (
                    (dy_ * np.cos(theta) - dx_ * np.sin(theta))
                    * (dx_ * np.cos(theta) - dy_ * np.sin(theta))
                )
                / sigma_x**2
            )
        )

        d_background = 1
        return [d_A, d_background, d_x_0, d_y_0, d_sigma_x, d_sigma_y, d_alpha]

    @property
    def input_units(self):
        if self.x_0.input_unit is None:
            return None
        else:
            return {
                self.inputs[0]: self.x_0.input_unit,
                self.inputs[1]: self.y_0.input_unit,
            }

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {
            "x_0": inputs_unit[self.inputs[0]],
            "y_0": inputs_unit[self.inputs[0]],
            "gamma": inputs_unit[self.inputs[0]],
            "amplitude": outputs_unit[self.outputs[0]],
        }
