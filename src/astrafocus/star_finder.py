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

    def __init__(
        self,
        ref_image: ImageType,
        fwhm: float = 2.0,
        star_find_threshold: float = 3.0,
        absolute_detection_limit: float = 0.0,
        saturation_threshold: float | None = None,
    ) -> None:
        self.ref_image = ref_image
        self.fwhm = fwhm
        self.star_find_threshold = star_find_threshold
        self.absolute_detection_limit = absolute_detection_limit
        self.saturation_threshold = saturation_threshold or np.inf
        # - not oversaturated

        mean, median, std = astropy.stats.sigma_clipped_stats(ref_image, sigma=3.0)
        self.ref_background = median
        self.ref_std = std

        potential_targets = self._find_sources()
        self.selected_stars = self.select_target_stars(potential_targets)

    def select_target_stars(self, potential_targets):
        # This could be achieved with peakmax
        selected_stars = potential_targets[potential_targets["peak"] <= self.saturation_threshold]

        return selected_stars

    def _find_sources(self):
        return self.find_sources(
            self.ref_image,
            fwhm=self.fwhm,
            threshold=self.star_find_threshold,
            std=self.ref_std,
            background=self.ref_background,
            absolute_detection_limit=self.absolute_detection_limit,
        )

    @staticmethod
    def find_sources(
        ref_image: ImageType,
        fwhm: float = 1.0,
        threshold: float = 5.0,
        std=None,
        background=None,
        peakmax=None,
        absolute_detection_limit: float = 0.0,
    ):
        """Detect and locate stars in the reference image"""
        if std is None or background is None:
            # Calculate summary statistics without pixels that are above or below a sigma from the median
            mean, median, std = astropy.stats.sigma_clipped_stats(ref_image, sigma=3.0)
            background = median

        sources = StarFinder._dao_star_finder(
            cleaned_image=ref_image - background,
            fwhm=fwhm,
            threshold=np.maximum(absolute_detection_limit, std * threshold),
            brightest=None,
            peakmax=peakmax,
        )

        while sources is None and threshold > 0.1:
            threshold /= 2
            logger.warning(
                f"No sources found in StarFinder: {ref_image.std()}, "
                f"{fwhm}, {threshold}. Decreasing threshold and retrying."
            )
            sources = StarFinder._dao_star_finder(
                cleaned_image=ref_image - background,
                fwhm=fwhm,
                threshold=np.maximum(absolute_detection_limit, std * threshold),
                brightest=None,
                peakmax=peakmax,
            )
        if sources is None:
            raise ValueError(
                f"No sources found in StarFinder: {ref_image.std()}, {fwhm}, {threshold}."
                "Decrease threshold and check the image."
            )

        try:
            sources.sort("flux", reverse=True)
        except Exception as exc:
            raise ValueError(f"In StarFinder: {ref_image.std()}, {fwhm}, {threshold}. {exc}")

        logger.info(f"Number of sources above threshold the threshold of {threshold} is {len(sources)}")

        return sources

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
            f"ref_image={self.ref_image}, "
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
