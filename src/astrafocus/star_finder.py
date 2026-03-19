import astropy
import numpy as np
from photutils.detection import DAOStarFinder

from astrafocus.utils.logger import get_logger
from astrafocus.utils.typing import ImageType

logger = get_logger()


class StarFinder:
    """
    Examples
    -------
    TargetFinder(ref_image)
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
    ) -> None:
        self.fwhm = fwhm
        self.star_find_threshold = star_find_threshold
        self.absolute_detection_limit = absolute_detection_limit
        self.saturation_threshold = saturation_threshold
        self.max_stars = max_stars

        mean, median, std = astropy.stats.sigma_clipped_stats(ref_image, sigma=3.0)
        self.ref_background = median
        self.ref_std = std

        self.selected_stars = self._find_sources(ref_image)
        if self.selected_stars is None:
            fallbacks = self.get_fallback_thresholds(self.star_find_threshold)
            min_tried = fallbacks[-1] if fallbacks.size > 0 else self.star_find_threshold
            raise ValueError(
                f"No sources found down to min_threshold={min_tried:.1f}σ. "
                "Check image quality, or adjust the star finder parameters "
                "(e.g. lower the star_find_threshold or absolute_detection_limit, or increase the fwhm)."
            )

    def _find_sources(self, ref_image):
        return self.find_sources(
            ref_image=ref_image,
            fwhm=self.fwhm,
            threshold=self.star_find_threshold,
            std=self.ref_std,
            background=self.ref_background,
            saturation_threshold=self.saturation_threshold,
            absolute_detection_limit=self.absolute_detection_limit,
            max_stars=self.max_stars,
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
            Background noise standard deviation. If None, estimated via
            sigma-clipped statistics.
        background : float, optional
            Median background level. If None, estimated via sigma-clipped
            statistics.
        saturation_threshold : float, optional
            Maximum allowed pixel value. Peaks above this are rejected
            (useful for excluding saturated stars that bias centroids).
        absolute_detection_limit : float, optional
            The hard floor for detection in ADU/counts. The effective
            threshold is max(absolute_limit, std * threshold).
        max_stars : int, optional
            Capping limit for returned sources to optimize downstream
            processing speed. Sorted by brightness.

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
            mean, median, std = astropy.stats.sigma_clipped_stats(ref_image, sigma=3.0)
            background = median

        cleaned_image = ref_image - background

        sources = StarFinder._dao_star_finder(
            cleaned_image=cleaned_image,
            fwhm=fwhm,
            threshold=np.maximum(absolute_detection_limit, std * threshold),
            peakmax=saturation_threshold,
            brightest=max_stars,
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
                peakmax=saturation_threshold,
                brightest=max_stars,
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
    def _dao_star_finder(cleaned_image, fwhm, threshold, brightest=None, peakmax=None):
        daofind = DAOStarFinder(
            fwhm=fwhm,
            threshold=threshold,
            brightest=brightest,
            peakmax=peakmax,
        )
        sources = daofind(cleaned_image)

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
