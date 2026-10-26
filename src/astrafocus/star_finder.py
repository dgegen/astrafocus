import warnings
from typing import NamedTuple

import astropy
import astropy.stats
import cv2
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from photutils.detection import DAOStarFinder
from scipy import ndimage, stats

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


# Sources that are masked before the sky is estimated: groups of touching
# pixels where the smoothed image is above SKY_MASK_N_SIGMA times its noise, with at least
# SKY_MASK_MIN_PIXELS pixels and a bounding box no more than SKY_MASK_MAX_ASPECT times longer than
# wide. So large stars and donuts are masked, but not noise or row banding. The mask is grown by
# SKY_MASK_GROW pixels to include the wings of the stars.
SKY_MASK_N_SIGMA = 3.0
SKY_MASK_MIN_PIXELS = 100
SKY_MASK_MAX_ASPECT = 4.0
SKY_MASK_GROW = 3
# A row or column with less than this fraction of pixels outside the mask gets the sky of its
# neighbours.
SKY_MIN_FREE_FRACTION = 0.2
# Number of times the mask is made and the sky estimated again, first with a plane as the sky and
# then with the sky of each row and column. The mask only grows, so star light that is found stays
# masked.
SKY_MASK_ITERATIONS = 2


def _plane_sky(data, mask, sample_step):
    """Least-squares plane through `data`, without the pixels in `mask` and sigma-clipped at 3 sigma."""
    rows, columns = np.mgrid[0 : data.shape[0] : sample_step, 0 : data.shape[1] : sample_step]
    values = data[::sample_step, ::sample_step].ravel()
    design = np.column_stack([np.ones(values.size), columns.ravel(), rows.ravel()])
    good = ~mask[::sample_step, ::sample_step].ravel()
    if np.count_nonzero(good) < design.shape[1]:
        good[:] = True
    for _ in range(5):
        coefficients, *_ = np.linalg.lstsq(design[good], values[good], rcond=None)
        residual = values - design @ coefficients
        std = astropy.stats.mad_std(residual[good])
        clipped = good & (np.abs(residual) < 3.0 * std)
        if np.count_nonzero(clipped) < design.shape[1] or np.array_equal(clipped, good):
            break
        good = clipped
    rows, columns = np.indices(data.shape, dtype=np.float32)
    return (coefficients[0] + coefficients[1] * columns + coefficients[2] * rows).astype(np.float32)


def _line_sky(data, mask, axis, sample_step):
    """
    Sigma-clipped median of `data` along `axis` (1: of each row, 0: of each column), without the
    pixels in `mask`. Lines with too few pixels outside the mask are interpolated from the others.
    """
    index = (slice(None), slice(None, None, sample_step)) if axis == 1 else slice(None, None, sample_step)
    values, free = data[index], ~mask[index]
    _, sky, _ = astropy.stats.sigma_clipped_stats(np.where(free, values, np.nan), sigma=3.0, axis=axis)
    good = (np.mean(free, axis=axis) >= SKY_MIN_FREE_FRACTION) & np.isfinite(sky)
    if not np.any(good):
        _, sky, _ = astropy.stats.sigma_clipped_stats(values, sigma=3.0, axis=axis)
        return sky
    if not np.all(good):
        logger.debug(f"Sky of {np.count_nonzero(~good)} lines (axis {axis}) interpolated: masked by stars.")
        lines = np.arange(sky.size)
        sky = np.interp(lines, lines[good], sky[good])
    return sky


def _row_column_sky(raw, mask, sample_step):
    """`raw` minus the sky of each row, and then minus the sky of each column of what is left."""
    data = raw - _line_sky(raw, mask, 1, sample_step)[:, None].astype(np.float32)
    data -= _line_sky(data, mask, 0, sample_step)[None, :].astype(np.float32)
    return data


def _background_noise(data, mask, sample_step):
    """
    Standard deviation of the background in `data`, from the pixels below the median, outside `mask`.
    Stars only add light, so they hardly change this, also when they cover a large part of the image.
    """
    values = data[::sample_step, ::sample_step][~mask[::sample_step, ::sample_step]]
    if values.size == 0:
        values = data[::sample_step, ::sample_step].ravel()
    median, lower = np.percentile(values, [50.0, 100 * stats.norm.cdf(-1.0)])
    return float(median - lower)


def _sky_mask(smoothed, smoothed_noise):
    """Mask of the large, compact sources in `smoothed` (see `SKY_MASK_N_SIGMA`)."""
    above = smoothed > SKY_MASK_N_SIGMA * smoothed_noise
    labels, n_groups = ndimage.label(above)
    if n_groups == 0:
        return above
    n_pixels = ndimage.sum_labels(above, labels, index=np.arange(1, n_groups + 1))
    keep = np.zeros(n_groups + 1, dtype=bool)
    for i_group, (rows, columns) in enumerate(ndimage.find_objects(labels)):
        height, width = rows.stop - rows.start, columns.stop - columns.start
        compact = max(height, width) <= SKY_MASK_MAX_ASPECT * min(height, width)
        keep[i_group + 1] = n_pixels[i_group] >= SKY_MASK_MIN_PIXELS and compact
    if not np.any(keep):
        return np.zeros_like(above)
    return ndimage.binary_dilation(keep[labels], iterations=SKY_MASK_GROW)


def prepare_image(image, remove_hot_pixels=True, smoothing=2.0, sample_step=4) -> PreparedImage:
    """
    Remove the sky and hot pixels from `image`, and make a smoothed copy to find faint star light in.

    The sky is a plane fitted to the image, plus the sigma-clipped median of each row, plus that of
    each column of what is left. This removes the sky level, gradients and row banding. Large stars
    and donuts are masked before the sky is estimated (see `SKY_MASK_N_SIGMA`), so their light does
    not raise the sky, also when they fill a large part of a row. The mask is first found with only
    the plane as the sky, because the median of a row that a donut fills would remove the donut's
    light before it can be found, and is then grown with the full sky. The noise is the
    sigma-clipped standard deviation of the result. Hot pixels are then replaced (see
    `replace_hot_pixels`). The copy is smoothed with a Gaussian of standard deviation `smoothing`
    pixels. Every `sample_step`-th row and column is used for the statistics, to make it fast. A
    `PreparedImage` is returned as it is.
    """
    if isinstance(image, PreparedImage):
        return image

    raw = np.asarray(image, dtype=np.float32)
    mask = np.zeros(raw.shape, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)  # lines that are fully masked
        for row_column in (False, True):
            for _ in range(SKY_MASK_ITERATIONS):
                data = raw - _plane_sky(raw, mask, sample_step)
                if row_column:
                    data = _row_column_sky(data, mask, sample_step)
                smoothed = cv2.GaussianBlur(data, (0, 0), smoothing)
                mask |= _sky_mask(smoothed, _background_noise(smoothed, mask, sample_step))

        data = _row_column_sky(raw - _plane_sky(raw, mask, sample_step), mask, sample_step)
        _, _, noise = astropy.stats.sigma_clipped_stats(
            np.where(mask, np.nan, data)[::sample_step, ::sample_step], sigma=3.0
        )
        if remove_hot_pixels:
            data = replace_hot_pixels(data, noise)
        smoothed = cv2.GaussianBlur(data, (0, 0), smoothing)
        _, _, smoothed_noise = astropy.stats.sigma_clipped_stats(
            np.where(mask, np.nan, smoothed)[::sample_step, ::sample_step], sigma=3.0
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
