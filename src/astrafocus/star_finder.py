import warnings
from typing import NamedTuple

import astropy
import astropy.stats
import cv2
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from photutils.detection import DAOStarFinder

from astrafocus.utils.logger import get_logger
from astrafocus.utils.typing import ImageType

logger = get_logger()


class PreparedImage(NamedTuple):
    """An image with the sky and hot pixels removed, and a smoothed copy (see `prepare_image`)."""

    data: np.ndarray
    noise: float
    smoothed: np.ndarray
    smoothed_noise: float


def replace_hot_pixels(image: ImageType, std: float, n_sigma: float = 5.0, ratio: float = 5.0) -> np.ndarray:
    """
    Replace hot pixels in a sky-subtracted image with the median of their 3x3 neighbourhood.

    A pixel is hot if it is more than `n_sigma` * `std` above the sky and more than `ratio` times
    the 3x3 median. A star with a FWHM of 1.3 pixels or more has a peak of less than 5 times its
    3x3 median, so stars are not changed. Clusters of up to 4 hot pixels are also replaced,
    because their 3x3 median is still the sky.

    Examples
    --------
    >>> image = np.zeros((5, 5))
    >>> image[2, 2] = 1000.0
    >>> float(replace_hot_pixels(image, std=1.0)[2, 2])
    0.0
    """
    image = np.asarray(image, dtype=np.float32)
    median = cv2.medianBlur(image, 3)
    hot = (image > n_sigma * std) & (image > ratio * np.maximum(median, 0))
    return np.where(hot, median, image)


def prepare_image(image, remove_hot_pixels=True, smoothing=2.0, sample_step=4) -> PreparedImage:
    """
    Remove the sky and hot pixels from `image`, and make a smoothed copy to find faint star light in.

    The sky is the sigma-clipped median of each row, plus that of each column of what is left. This
    removes the sky level, row banding and gradients, and the light of one star does not change it.
    The noise is the sigma-clipped standard deviation of the result. Hot pixels are then replaced
    (see `replace_hot_pixels`). The copy is smoothed with a Gaussian of standard
    deviation `smoothing` pixels. Every `sample_step`-th row and column is used for the statistics,
    to make it fast. A `PreparedImage` is returned as it is.
    """
    if isinstance(image, PreparedImage):
        return image

    data = np.asarray(image, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        _, row_sky, _ = astropy.stats.sigma_clipped_stats(data[:, ::sample_step], sigma=3.0, axis=1)
        data = data - row_sky[:, None].astype(np.float32)
        _, column_sky, _ = astropy.stats.sigma_clipped_stats(data[::sample_step], sigma=3.0, axis=0)
        data -= column_sky[None, :].astype(np.float32)
        _, _, noise = astropy.stats.sigma_clipped_stats(data[::sample_step, ::sample_step], sigma=3.0)
        if remove_hot_pixels:
            data = replace_hot_pixels(data, noise)
        smoothed = cv2.GaussianBlur(data, (0, 0), smoothing)
        _, _, smoothed_noise = astropy.stats.sigma_clipped_stats(
            smoothed[::sample_step, ::sample_step], sigma=3.0
        )
    return PreparedImage(data, float(noise), smoothed, float(smoothed_noise))


class StarFinder:
    """
    Detects stars in a reference image using a tiered-threshold DAOFIND approach.

    On construction, background statistics are estimated from `ref_image` (via
    sigma-clipped stats) and an initial source search is run at
    `star_find_threshold`. If no sources are found, the search retries at
    progressively lower thresholds from `FALLBACK_THRESHOLDS`. See
    `find_sources` for the search logic. If no sources are found even after
    all fallbacks, a `ValueError` is raised.

    Parameters
    ----------
    ref_image : 2D array_like
        The reference image to search for stars.
    fwhm : float, optional
        Full-Width at Half-Maximum (pixels) of the Gaussian kernel used by
        DAOStarFinder. Default is 3.0.
    star_find_threshold : float, optional
        Initial detection threshold in units of background standard
        deviation (sigma). Default is 4.0.
    absolute_detection_limit : float, optional
        Hard floor for detection in ADU/counts. The effective threshold is
        max(absolute_detection_limit, std * threshold). Default is 0.0.
    saturation_threshold : float, optional
        Maximum allowed pixel value; peaks above this are rejected. Default
        is None (no cap).
    max_stars : int, optional
        Maximum number of sources to keep, sorted by brightness. Default is
        50.
    sharpness_range : tuple of float, optional
        (sharplo, sharphi) bounds passed to DAOStarFinder; see
        `find_sources` for details. Default is (0.05, 1.0).
    remove_hot_pixels : bool, optional
        Replace hot pixels before detection (see `replace_hot_pixels`). Default is True.

    Attributes
    ----------
    ref_background : float
        Sigma-clipped median background level of `ref_image`.
    ref_std : float
        Sigma-clipped standard deviation of `ref_image`.
    selected_stars : astropy.table.QTable
        Table of detected sources with centroids and photometry.

    Raises
    ------
    ValueError
        If no sources are found at `star_find_threshold` or any fallback
        threshold.

    Examples
    --------
    StarFinder(ref_image)
    """

    FALLBACK_THRESHOLDS = np.array([4, 3, 2.5])

    def __init__(
        self,
        ref_image: ImageType,
        fwhm: float = 3.0,
        star_find_threshold: float = 4.0,
        absolute_detection_limit: float = 0.0,
        saturation_threshold: float | None = None,
        max_stars: int = 50,
        sharpness_range: tuple[float, float] = (0.05, 1.0),
        remove_hot_pixels: bool = True,
    ) -> None:
        self.fwhm = fwhm
        self.star_find_threshold = star_find_threshold
        self.absolute_detection_limit = absolute_detection_limit
        self.saturation_threshold = saturation_threshold
        self.max_stars = max_stars
        self.sharpness_range = sharpness_range
        self.remove_hot_pixels = remove_hot_pixels

        # Kept as attributes for users. Detection removes the sky per row and column instead
        # (see `prepare_image`), which also works with row banding and gradients.
        mean, median, std = astropy.stats.sigma_clipped_stats(ref_image, sigma=3.0)
        self.ref_background = median
        self.ref_std = std

        # Far from focus, stars are large donuts, which a wider kernel finds.
        for fwhm_factor in (1, 4, 16):
            self.selected_stars = self._find_sources(ref_image, fwhm_factor * self.fwhm)
            if self.selected_stars is not None:
                break
        if self.selected_stars is None:
            fallbacks = self.get_fallback_thresholds(self.star_find_threshold)
            min_tried = fallbacks[-1] if fallbacks.size > 0 else self.star_find_threshold
            raise ValueError(
                f"No sources found down to min_threshold={min_tried:.1f}σ. "
                "Check image quality, or adjust the star finder parameters "
                "(e.g. lower the star_find_threshold or absolute_detection_limit, or increase the fwhm)."
            )

    def _find_sources(self, ref_image, fwhm):
        return self.find_sources(
            ref_image=ref_image,
            fwhm=fwhm,
            threshold=self.star_find_threshold,
            saturation_threshold=self.saturation_threshold,
            absolute_detection_limit=self.absolute_detection_limit,
            max_stars=self.max_stars,
            sharpness_range=self.sharpness_range,
            remove_hot_pixels=self.remove_hot_pixels,
        )

    @classmethod
    def find_sources(
        cls,
        ref_image: ImageType,
        fwhm: float = 3.0,
        threshold: float = 4.0,
        std=None,
        background=None,
        saturation_threshold=None,
        absolute_detection_limit: float = 0.0,
        max_stars: int = 50,
        sharpness_range: tuple[float, float] = (0.05, 1.0),
        remove_hot_pixels: bool = True,
    ):
        """
        Detect and locate stars using a tiered-threshold DAOFIND approach.

        This method performs an initial search at the specified `threshold`.
        If no sources are found, it automatically falls back to searching at
        progressively lower thresholds defined in `FALLBACK_THRESHOLDS`. This
        is designed to maintain autofocus reliability even when stars are
        blurred (out-of-focus) or sky transparency drops.

        Parameters
        ----------
        ref_image : 2D array_like
            The background-subtracted or raw image array.
        fwhm : float, optional
            Full-Width at Half-Maximum (pixels) of the Gaussian kernel.
            Standard ground-based telescopes typically use 2.5 to 4.0.
        threshold : float, optional
            Initial detection threshold in units of background standard
            deviation (sigma). Default is 4.0.
        std : float, optional
            Background noise standard deviation. If `std` or `background` is None,
            the sky is removed per row and column instead (see `prepare_image`).
        background : float, optional
            Median background level, subtracted from `ref_image`.
        saturation_threshold : float, optional
            Maximum allowed pixel value. Peaks above this are rejected
            (useful for excluding saturated stars that bias centroids).
        absolute_detection_limit : float, optional
            The hard floor for detection in ADU/counts. The effective
            threshold is max(absolute_limit, std * threshold).
        max_stars : int, optional
            Capping limit for returned sources to optimize downstream
            processing speed. Sorted by brightness.
        sharpness_range : tuple of float, optional
            (sharplo, sharphi) bounds passed to DAOStarFinder. The default
            lower bound of 0.05 (below the photutils default of 0.2) admits
            softer/broader sources, which helps find stars on an initial
            out-of-focus reference image. Loosening it further increases the
            risk of picking up non-stellar diffuse artifacts (e.g. hot-pixel
            clusters, cosmic rays, background blobs).
        remove_hot_pixels : bool, optional
            Replace hot pixels before detection (see `replace_hot_pixels`),
            because they can pass the sharpness bounds. Default is True.

        Returns
        -------
        astropy.table.QTable or None
            A table of detected sources with centroids and photometry,
            or None if no sources meet the criteria even after fallbacks.

        Notes
        -----
        The tiered approach prevents the "Noise Explosion" problem. Searching
        immediately at 2.0 sigma on a noisy CMOS sensor can result in
        thousands of false positives, slowing down the characterization
        phase significantly. By starting higher and falling back only when
        necessary, we balance speed with sensitivity.
        """

        if std is None or background is None:
            prepared = prepare_image(ref_image, remove_hot_pixels=remove_hot_pixels)
            cleaned_image, std = prepared.data, prepared.noise
        else:
            cleaned_image = ref_image - background
            if remove_hot_pixels:
                cleaned_image = replace_hot_pixels(cleaned_image, std)

        sources = StarFinder._dao_star_finder(
            cleaned_image=cleaned_image,
            fwhm=fwhm,
            threshold=np.maximum(absolute_detection_limit, std * threshold),
            peak_max=saturation_threshold,
            n_brightest=max_stars,
            sharpness_range=sharpness_range,
        )
        if sources is not None:
            sources.sort("flux", reverse=True)
            logger.info(f"Found {len(sources)} sources at threshold={threshold:.2f}σ.")
            return sources

        for threshold in cls.get_fallback_thresholds(threshold):
            sources = StarFinder._dao_star_finder(
                cleaned_image=cleaned_image,
                fwhm=fwhm,
                threshold=np.maximum(absolute_detection_limit, std * threshold),
                peak_max=saturation_threshold,
                n_brightest=max_stars,
                sharpness_range=sharpness_range,
            )

            if sources is not None:
                sources.sort("flux", reverse=True)
                logger.info(f"Found {len(sources)} sources at fallback threshold={threshold:.1f}σ.")
                return sources

        return sources

    @classmethod
    def get_fallback_thresholds(cls, threshold: float | None) -> np.ndarray:
        if threshold is None:
            return cls.FALLBACK_THRESHOLDS
        else:
            return cls.FALLBACK_THRESHOLDS[(cls.FALLBACK_THRESHOLDS < threshold)]

    @staticmethod
    def _dao_star_finder(
        cleaned_image,
        fwhm,
        threshold,
        n_brightest=None,
        peak_max=None,
        sharpness_range=(0.05, 1.0),
    ):
        daofind = DAOStarFinder(
            fwhm=fwhm,
            threshold=threshold,
            n_brightest=n_brightest,
            peak_max=peak_max,
            sharpness_range=sharpness_range,
        )
        sources = daofind(cleaned_image)
        if sources is not None:
            # A source with no positive flux is noise.
            sources = sources[sources["flux"] > 0]
            if len(sources) == 0:
                sources = None

        return sources

    def __repr__(self):
        return (
            "StarFinder("
            f"fwhm={self.fwhm}, "
            f"star_find_threshold={self.star_find_threshold}, "
            f"absolute_detection_limit={self.absolute_detection_limit}, "
            f"saturation_threshold={self.saturation_threshold})"
        )

    def __str__(self):
        return (
            "StarFinder("
            f"fwhm={self.fwhm}, "
            f"star_find_threshold={self.star_find_threshold}, "
            f"absolute_detection_limit={self.absolute_detection_limit}, "
            f"saturation_threshold={self.saturation_threshold})"
        )
