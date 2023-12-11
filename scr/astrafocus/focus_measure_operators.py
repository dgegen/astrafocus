from abc import ABC, abstractmethod
from typing import Union

import cv2
import numpy as np
import numpy.typing as npt

from astrafocus.utils.typing import ImageType

ImageType = npt.NDArray[Union[np.floating, np.integer]]


class FocusMeasureOperator(ABC):
    """
    Abstract base class for focus measure operators.

    Attributes
    ----------
    smaller_is_better : bool
        A class attribute indicating whether a smaller focus measure is considered better.

    Methods
    -------
    __call__(image: np.ndarray, **kwargs) -> float
        Compute the focus measure of the input image.

    measure_focus(image: np.ndarray, **kwargs) -> float
        Abstract method to be implemented by subclasses.
        Compute the focus measure of the input image.

    convert_to_grayscale(image: np.ndarray) -> np.ndarray
        Convert the input image to grayscale.

    validate_image(image: np.ndarray)
        Validate the input image for compatibility with focus measure algorithms.
    """

    smaller_is_better = True

    def __call__(self, image: ImageType, **kwargs) -> float:
        """
        Compute the focus measure of the input image.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        float
            Computed focus measure.
        """
        self.validate_image(image)
        return self.measure_focus(self.convert_to_grayscale(image), **kwargs)

    @abstractmethod
    def measure_focus(self, image: ImageType, **kwargs) -> float:
        """
        Abstract method to be implemented by subclasses.
        Compute the focus measure of the input image.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        float
            Computed focus measure.
        """
        pass

    @staticmethod
    def convert_to_grayscale(image: ImageType) -> ImageType:
        """Convert the input image to grayscale."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        return image

    @staticmethod
    def validate_image(image):
        """Validate the input image for compatibility with focus measure algorithms."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array.")
        if not (np.issubdtype(image.dtype, np.integer) or np.issubdtype(image.dtype, np.floating)):
            raise ValueError("Values in the numpy array must be either integers or floats.")


class AnalyticResponseFocusedMeasureOperator(FocusMeasureOperator):
    @abstractmethod
    def fit_focus_response_curve(focus_pos, measured_focus):
        pass

    @abstractmethod
    def get_focus_response_curve_fit(self, focus_pos):
        pass


class AutoCorrelationFocusMeasure(FocusMeasureOperator):
    smaller_is_better = False

    def measure_focus(self, image: ImageType, **kwargs) -> float:
        if image.dtype == bool:
            raise ValueError("Bools are not allowed")
        return float(np.sum(image[:-1, :] * image[1:, :]) - np.sum(image[:-2, :] * image[2:, :]))


class BrennerFocusMeasure(FocusMeasureOperator):
    smaller_is_better = False

    def measure_focus(self, image: ImageType, **kwargs) -> float:
        return float(np.sum(np.abs(image[:-2] - image[2:]) ** 2))


class NormalizedVarianceFocusMeasure(FocusMeasureOperator):
    smaller_is_better = False

    def measure_focus(self, image: ImageType, **kwargs) -> float:
        return image.var() / image.mean()


class FFTFocusMeasure(FocusMeasureOperator):
    """
    \mathrm{FM} &= \norm{\bm{R \phi}}_{1} \\
    \mathrm{FFT}(x, y) &= R(x, y) \exp\qty(-i\phi(x,y)) \\
    """

    smaller_is_better = False

    def measure_focus(self, image: ImageType, **kwargs) -> float:
        f = np.fft.fft2(image)
        mag = np.abs(f)
        phase = np.arctan2(np.imag(f), np.real(f))
        return np.sum(np.abs(phase * mag))


class FFTFocusMeasureTan2022(FocusMeasureOperator):
    """
    \mathrm{FM} &= \norm{\bm{R \phi}}_{1} \\
    \mathrm{FFT}(x, y) &= R(x, y) \exp\qty(-i\phi(x,y)) \\
    """

    smaller_is_better = False

    def measure_focus(self, image: ImageType, **kwargs) -> float:
        f = np.fft.fft2(image)
        mag = np.abs(f)

        n_max_y, n_max_x = np.array(image.shape) // 2

        # average of the high frequency components of the power spectrum
        beta = np.mean(mag[n_max_y // 2 : n_max_y, n_max_x // 2 : n_max_x])
        fft_without_noise = mag - beta

        return np.sum(fft_without_noise[fft_without_noise > 0])


class FFTPowerFocusMeasure(FocusMeasureOperator):
    """
    \mathrm{FM} &= \norm{\bm{R \phi}}_{1} \\
    \mathrm{FFT}(x, y) &= R(x, y) \exp\qty(-i\phi(x,y)) \\

    """

    smaller_is_better = False

    def measure_focus(self, image: ImageType, **kwargs) -> float:
        f = np.fft.fft2(image)
        mag = np.abs(f)

        n_max_y, n_max_x = np.array(image.shape) // 2

        # average of the high frequency components of the power spectrum
        beta = np.mean(mag[n_max_y // 2 : n_max_y, n_max_x // 2 : n_max_x])
        fft_without_noise = mag - beta

        return np.sum(fft_without_noise[fft_without_noise > 0])


class AbsoluteGradientFocusMeasure(FocusMeasureOperator):
    smaller_is_better = False

    def measure_focus(self, image: ImageType, **kwargs) -> float:
        return np.sum(np.abs(image[:, 1:] - image[:, :-1])) + np.sum(
            np.abs(image[1:, :] - image[:-1, :])
        )


class SquaredGradientFocusMeasure(FocusMeasureOperator):
    smaller_is_better = False

    def measure_focus(self, image: ImageType, **kwargs) -> float:
        return float(
            np.sum((image[:, 1:] - image[:, :-1]) ** 2)
            + np.sum((image[1:, :] - image[:-1, :]) ** 2)
        )


class VarianceOfLaplacianFocusMeasure(FocusMeasureOperator):
    smaller_is_better = False

    def measure_focus(self, image: ImageType, **kwargs) -> float:
        return cv2.Laplacian(image, cv2.CV_64F).var()


class LaplacianFocusMeasure(FocusMeasureOperator):
    smaller_is_better = False

    def measure_focus(self, image: ImageType, **kwargs) -> float:
        return np.sum(np.abs(cv2.Laplacian(image, cv2.CV_64F)))


class TenengradFocusMeasure(FocusMeasureOperator):
    smaller_is_better = False

    def measure_focus(self, image: ImageType, ksize=1, **kwargs) -> float:
        # Compute the gradients in the x and y directions using Sobel operators
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        tenengrad_measure = gradient_x**2 + gradient_y**2

        return np.sum(tenengrad_measure)
